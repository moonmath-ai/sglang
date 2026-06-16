# SPDX-License-Identifier: Apache-2.0
"""Step-level continuous batching for diffusion denoising.

The default worker path runs an entire denoising loop (all steps) for a request
inside a single ``pipeline.forward`` call, so a batch of requests must start and
finish together. This module inverts that control flow so the scheduler can keep
several *groups* of requests running simultaneously and advance them one denoise
step at a time:

* requests admitted mid-flight join at the next step boundary instead of waiting
  for a fixed batch window,
* groups finish independently (a 20-step group returns while a 50-step group
  keeps going),
* compatible running groups share a single batched denoise forward pass.

Each running unit is a **merged batch group**: the scheduler merges compatible
requests (same shape/CFG/steps) into one batched ``Req`` so the per-request
encode and decode stages are batched too (matching dynamic batching), while the
denoise loop is still driven step-by-step here. A group carries the list of
client ``identities`` and the original per-request ``Req`` objects so the
scheduler can split the batched output back to each client.

The engine is worker-agnostic: it talks to a :class:`StepRunner` (the GPU worker)
through a small protocol so the scheduling logic can be unit-tested without a
GPU. Compatibility is expressed by an opaque, hashable ``batch_key`` (keyed on
the *per-sample* shape, so groups of different sizes can still fuse).
"""

from __future__ import annotations

import dataclasses
from typing import Any, Hashable, Optional, Protocol

from sglang.multimodal_gen.runtime.utils.logging_utils import init_logger

logger = init_logger(__name__)


@dataclasses.dataclass
class DiffusionRequestState:
    """Mutable state for a merged batch group currently being denoised.

    The runner owns the heavy fields (``req``/``ctx``); the engine only reads the
    scheduling-relevant bookkeeping (``batch_key``, ``step_index``, ``num_steps``,
    ``error``). ``identities`` and ``original_reqs`` let the scheduler split the
    batched output back to each client.
    """

    identities: list[bytes | None]
    original_reqs: list[Any]
    request_id: str
    # Opaque compatibility key; states with equal keys may share a forward pass.
    batch_key: Hashable
    num_steps: int
    # Runner-owned payloads (the merged batched Req and its DenoisingContext).
    req: Any = None
    ctx: Any = None
    step_index: int = 0
    error: str | None = None

    @property
    def num_requests(self) -> int:
        return len(self.identities)

    def is_finished(self) -> bool:
        return self.error is not None or self.step_index >= self.num_steps


class StepRunner(Protocol):
    """Worker-side hooks driven by :class:`ContinuousBatchingEngine`.

    Implementations run actual model code. Recoverable failures should be
    recorded on the state via ``state.error`` rather than raised, so one bad
    group cannot tear down the whole running set. The engine still guards against
    raised exceptions.
    """

    def cb_prepare(
        self,
        identities: list[bytes | None],
        original_reqs: list[Any],
        merged_req: Any,
    ) -> DiffusionRequestState:
        """Run pre-denoise stages (batched) + denoise setup; return a ready state."""
        ...

    def cb_step(self, group: list[DiffusionRequestState]) -> None:
        """Advance every state in ``group`` by exactly one denoise step.

        All states in a group share the same ``batch_key``. The runner may fuse
        them into one forward pass or step them individually; either way it must
        increment ``state.step_index`` (or set ``state.error``) for each state.
        """
        ...

    def cb_finalize(self, state: DiffusionRequestState) -> Any:
        """Run denoise teardown + post-denoise stages (batched); return OutputBatch."""
        ...

    def cb_make_error_output(self, state: DiffusionRequestState) -> Any:
        """Build an OutputBatch carrying ``state.error`` for a failed group."""
        ...


# A finished result the engine hands back: the group state plus its (possibly
# batched) OutputBatch. The scheduler splits the output across state.identities.
FinishedResult = tuple[DiffusionRequestState, Any]


class ContinuousBatchingEngine:
    """Maintains the running set of denoising groups and advances them.

    One :meth:`step` call is a single "tick": every running group advances by one
    denoise step (compatible groups fused into shared forward passes), after which
    finished groups are finalized and returned. The scheduler calls :meth:`step`
    repeatedly, interleaving admission of new groups between ticks.
    """

    def __init__(
        self,
        runner: StepRunner,
        *,
        max_running: int,
        max_batch_size: int,
    ) -> None:
        if max_running < 1:
            raise ValueError("max_running must be >= 1")
        if max_batch_size < 1:
            raise ValueError("max_batch_size must be >= 1")
        self._runner = runner
        self._max_running = max_running
        self._max_batch_size = max_batch_size
        self._running: list[DiffusionRequestState] = []

    # -- introspection -----------------------------------------------------

    @property
    def num_running(self) -> int:
        """Number of running *requests* (summed across groups)."""
        return sum(s.num_requests for s in self._running)

    @property
    def num_running_groups(self) -> int:
        return len(self._running)

    def has_work(self) -> bool:
        return bool(self._running)

    def admit_headroom(self) -> int:
        """How many more requests may be admitted before hitting max_running."""
        return max(0, self._max_running - self.num_running)

    # -- admission ---------------------------------------------------------

    def admit(
        self,
        identities: list[bytes | None],
        original_reqs: list[Any],
        merged_req: Any,
    ) -> Optional[FinishedResult]:
        """Prepare a merged group and add it to the running set.

        Returns a ``(state, OutputBatch)`` pair if preparation failed outright or
        the group had nothing to denoise (so the caller replies immediately);
        otherwise ``None`` and the group joins the running set.
        """
        try:
            state = self._runner.cb_prepare(identities, original_reqs, merged_req)
        except Exception as e:  # pragma: no cover - defensive
            logger.error(
                "Continuous batching: cb_prepare raised for a group of %d; "
                "returning error. %s",
                len(identities),
                e,
                exc_info=True,
            )
            state = DiffusionRequestState(
                identities=identities,
                original_reqs=original_reqs,
                request_id=getattr(merged_req, "request_id", "unknown"),
                batch_key=None,
                num_steps=0,
                error=f"continuous batching prepare failed: {e}",
            )

        if state.error is not None:
            return state, self._runner.cb_make_error_output(state)

        if state.num_steps <= 0:
            return state, self._safe_finalize(state)

        self._running.append(state)
        logger.debug(
            "Continuous batching: admitted group %s (%d reqs, %d steps); "
            "running groups=%d reqs=%d",
            state.request_id,
            state.num_requests,
            state.num_steps,
            len(self._running),
            self.num_running,
        )
        return None

    # -- stepping ----------------------------------------------------------

    def step(self) -> list[FinishedResult]:
        """Advance all running groups by one denoise step; return finished ones."""
        if not self._running:
            return []

        for group in self._group_active_states():
            try:
                self._runner.cb_step(group)
            except Exception as e:  # pragma: no cover - defensive
                logger.error(
                    "Continuous batching: cb_step raised for %d group(s); "
                    "marking them failed. %s",
                    len(group),
                    e,
                    exc_info=True,
                )
                for state in group:
                    if state.error is None:
                        state.error = f"continuous batching step failed: {e}"

        return self._collect_finished()

    def _group_active_states(self) -> list[list[DiffusionRequestState]]:
        """Group not-yet-finished states by ``batch_key`` for a shared forward.

        Caps each fused sub-batch at ``max_batch_size`` *requests* (summed across
        the group's states) so a fused forward never exceeds the configured size.
        """
        # dicts preserve insertion order, so iterating values() keeps batch_keys
        # in first-seen order without a separate ordering list.
        groups: dict[Hashable, list[list[DiffusionRequestState]]] = {}
        counts: dict[Hashable, list[int]] = {}
        for state in self._running:
            if state.is_finished():
                continue
            buckets = groups.get(state.batch_key)
            if buckets is None:
                buckets = groups[state.batch_key] = [[]]
                counts[state.batch_key] = [0]
            bucket_counts = counts[state.batch_key]
            if (
                buckets[-1]
                and bucket_counts[-1] + state.num_requests > self._max_batch_size
            ):
                buckets.append([])
                bucket_counts.append(0)
            buckets[-1].append(state)
            bucket_counts[-1] += state.num_requests

        result: list[list[DiffusionRequestState]] = []
        for buckets in groups.values():
            result.extend(buckets)
        return result

    def _collect_finished(self) -> list[FinishedResult]:
        finished: list[FinishedResult] = []
        still_running: list[DiffusionRequestState] = []
        for state in self._running:
            if not state.is_finished():
                still_running.append(state)
                continue
            if state.error is not None:
                finished.append((state, self._runner.cb_make_error_output(state)))
            else:
                finished.append((state, self._safe_finalize(state)))
        self._running = still_running
        return finished

    def _safe_finalize(self, state: DiffusionRequestState) -> Any:
        try:
            return self._runner.cb_finalize(state)
        except Exception as e:  # pragma: no cover - defensive
            logger.error(
                "Continuous batching: cb_finalize raised for group %s; "
                "returning error. %s",
                state.request_id,
                e,
                exc_info=True,
            )
            state.error = f"continuous batching finalize failed: {e}"
            return self._runner.cb_make_error_output(state)
