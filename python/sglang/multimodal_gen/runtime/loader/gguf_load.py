"""Loader for GGUF-quantized diffusion transformers.

GGUF diffusion checkpoints ship as a single ``.gguf`` file holding the whole
transformer (the dominant community distribution format for Flux / SD3 / Wan).
The weights stay packed (uint8) in VRAM and are dequantized on-the-fly by
`GGUFLinearMethod`, which is what lets a large DiT fit on a small GPU.

This load path is intentionally separate from the safetensors/FSDP loader:
GGUF ``qweight`` parameters are materialized lazily from the quantized blocks
(unknown packed shape at construction time) and therefore do not fit the
meta-device + ``torch.empty_like`` flow in ``fsdp_load.py``. It mirrors the
`srt` ``GGUFModelLoader``: build the model on the target device with a
``GGUFConfig``, stream the GGUF tensors into the model, then run
``process_weights_after_loading``.

Scope: single-GPU loading (no FSDP sharding, no tensor parallelism). GGUF is a
small-VRAM path, so this matches how these checkpoints are used in practice.
"""

from __future__ import annotations

import re
from collections import defaultdict
from typing import Any, Callable, Iterator

import torch
from torch import nn

from sglang.multimodal_gen.runtime.loader.utils import (
    get_param_names_mapping,
    hf_to_custom_state_dict,
    set_default_torch_dtype,
)
from sglang.multimodal_gen.runtime.utils.logging_utils import init_logger

logger = init_logger(__name__)


# ---------------------------------------------------------------------------
# Model-specific GGUF adapters
# ---------------------------------------------------------------------------
# Community GGUF checkpoints often use the *original* model naming (e.g. the
# reference Flux `double_blocks.0.img_attn.qkv.weight`) and pack attention
# projections fused together, whereas SGLang's DiTs use diffusers naming with
# separate q/k/v linears. An adapter rewrites the GGUF tensor names to the
# model's parameter names and splits fused tensors along the output dimension.
#
# Splitting a quantized weight along the output (row) dimension is safe: GGML
# k-quant/standard blocks run along the input dimension within a row, so each
# output row is an independent, intact run of quantized bytes.
#
# Adapters that don't apply just aren't registered; checkpoints already in
# diffusers naming fall back to the model's param_names_mapping.


def _flux_gguf_iter(
    reader, inner_dim: int
) -> Iterator[tuple[str, Any, torch.Tensor]]:
    """Rewrite reference-Flux GGUF tensors to SGLang diffusers param names.

    Yields ``(model_param_name, weight_type, tensor)``. Fused qkv (double
    blocks) and fused qkv+mlp (single blocks) are split along output rows.
    """

    def t(tensor):
        return torch.tensor(tensor.data)

    # 1:1 renames (regex on the full GGUF name, suffix .weight/.bias preserved).
    SUFFIX = r"\.(weight|bias)$"
    rename_rules: list[tuple[re.Pattern, str]] = [
        # double blocks
        (re.compile(rf"^double_blocks\.(\d+)\.img_attn\.proj{SUFFIX}"),
         r"transformer_blocks.\1.attn.to_out.0.\2"),
        (re.compile(rf"^double_blocks\.(\d+)\.txt_attn\.proj{SUFFIX}"),
         r"transformer_blocks.\1.attn.to_add_out.\2"),
        (re.compile(rf"^double_blocks\.(\d+)\.img_mlp\.0{SUFFIX}"),
         r"transformer_blocks.\1.ff.net.0.proj.\2"),
        (re.compile(rf"^double_blocks\.(\d+)\.img_mlp\.2{SUFFIX}"),
         r"transformer_blocks.\1.ff.net.2.\2"),
        (re.compile(rf"^double_blocks\.(\d+)\.txt_mlp\.0{SUFFIX}"),
         r"transformer_blocks.\1.ff_context.net.0.proj.\2"),
        (re.compile(rf"^double_blocks\.(\d+)\.txt_mlp\.2{SUFFIX}"),
         r"transformer_blocks.\1.ff_context.net.2.\2"),
        (re.compile(rf"^double_blocks\.(\d+)\.img_mod\.lin{SUFFIX}"),
         r"transformer_blocks.\1.norm1.linear.\2"),
        (re.compile(rf"^double_blocks\.(\d+)\.txt_mod\.lin{SUFFIX}"),
         r"transformer_blocks.\1.norm1_context.linear.\2"),
        (re.compile(r"^double_blocks\.(\d+)\.img_attn\.norm\.query_norm\.scale$"),
         r"transformer_blocks.\1.attn.norm_q.weight"),
        (re.compile(r"^double_blocks\.(\d+)\.img_attn\.norm\.key_norm\.scale$"),
         r"transformer_blocks.\1.attn.norm_k.weight"),
        (re.compile(r"^double_blocks\.(\d+)\.txt_attn\.norm\.query_norm\.scale$"),
         r"transformer_blocks.\1.attn.norm_added_q.weight"),
        (re.compile(r"^double_blocks\.(\d+)\.txt_attn\.norm\.key_norm\.scale$"),
         r"transformer_blocks.\1.attn.norm_added_k.weight"),
        # single blocks
        (re.compile(rf"^single_blocks\.(\d+)\.linear2{SUFFIX}"),
         r"single_transformer_blocks.\1.proj_out.\2"),
        (re.compile(rf"^single_blocks\.(\d+)\.modulation\.lin{SUFFIX}"),
         r"single_transformer_blocks.\1.norm.linear.\2"),
        (re.compile(r"^single_blocks\.(\d+)\.norm\.query_norm\.scale$"),
         r"single_transformer_blocks.\1.attn.norm_q.weight"),
        (re.compile(r"^single_blocks\.(\d+)\.norm\.key_norm\.scale$"),
         r"single_transformer_blocks.\1.attn.norm_k.weight"),
        # top-level
        (re.compile(rf"^img_in{SUFFIX}"), r"x_embedder.\1"),
        (re.compile(rf"^txt_in{SUFFIX}"), r"context_embedder.\1"),
        (re.compile(rf"^time_in\.in_layer{SUFFIX}"),
         r"time_text_embed.timestep_embedder.linear_1.\1"),
        (re.compile(rf"^time_in\.out_layer{SUFFIX}"),
         r"time_text_embed.timestep_embedder.linear_2.\1"),
        (re.compile(rf"^vector_in\.in_layer{SUFFIX}"),
         r"time_text_embed.text_embedder.linear_1.\1"),
        (re.compile(rf"^vector_in\.out_layer{SUFFIX}"),
         r"time_text_embed.text_embedder.linear_2.\1"),
        (re.compile(rf"^guidance_in\.in_layer{SUFFIX}"),
         r"time_text_embed.guidance_embedder.linear_1.\1"),
        (re.compile(rf"^guidance_in\.out_layer{SUFFIX}"),
         r"time_text_embed.guidance_embedder.linear_2.\1"),
        (re.compile(rf"^final_layer\.linear{SUFFIX}"), r"proj_out.\1"),
        # final_layer.adaLN_modulation.1 is handled separately (scale/shift swap).
    ]
    # Fused tensors to split along output rows -> (regex, [(target, size), ...]).
    # size None means "the remainder" (single-block mlp).
    fused_rules: list[tuple[re.Pattern, list[tuple[str, int | None]]]] = [
        (re.compile(rf"^double_blocks\.(\d+)\.img_attn\.qkv{SUFFIX}"),
         [("transformer_blocks.{0}.attn.to_q.{1}", inner_dim),
          ("transformer_blocks.{0}.attn.to_k.{1}", inner_dim),
          ("transformer_blocks.{0}.attn.to_v.{1}", inner_dim)]),
        (re.compile(rf"^double_blocks\.(\d+)\.txt_attn\.qkv{SUFFIX}"),
         [("transformer_blocks.{0}.attn.add_q_proj.{1}", inner_dim),
          ("transformer_blocks.{0}.attn.add_k_proj.{1}", inner_dim),
          ("transformer_blocks.{0}.attn.add_v_proj.{1}", inner_dim)]),
        (re.compile(rf"^single_blocks\.(\d+)\.linear1{SUFFIX}"),
         [("single_transformer_blocks.{0}.attn.to_q.{1}", inner_dim),
          ("single_transformer_blocks.{0}.attn.to_k.{1}", inner_dim),
          ("single_transformer_blocks.{0}.attn.to_v.{1}", inner_dim),
          ("single_transformer_blocks.{0}.proj_mlp.{1}", None)]),
    ]

    # diffusers' AdaLayerNormContinuous stores the final modulation as
    # (scale, shift) whereas reference Flux stores (shift, scale). The diffusers
    # conversion swaps the two halves along the output dim for norm_out only
    # (per-block img_mod/txt_mod are not swapped). Mirror that here.
    final_mod = re.compile(rf"^final_layer\.adaLN_modulation\.1{SUFFIX}")

    def _swap_scale_shift(x: torch.Tensor) -> torch.Tensor:
        shift, scale = x.chunk(2, dim=0)
        return torch.cat([scale, shift], dim=0)

    for tensor in reader.tensors:
        name = tensor.name
        wt = tensor.tensor_type

        m = final_mod.match(name)
        if m:
            yield f"norm_out.linear.{m.group(1)}", wt, _swap_scale_shift(t(tensor))
            continue

        for pat, targets in fused_rules:
            m = pat.match(name)
            if not m:
                continue
            data = t(tensor)  # (out_rows, ...) — split along dim 0
            block, suffix = m.group(1), m.group(2)
            start = 0
            for tgt, size in targets:
                size = data.shape[0] - start if size is None else size
                yield tgt.format(block, suffix), wt, data[start : start + size].contiguous()
                start += size
            break
        else:
            for pat, repl in rename_rules:
                new = pat.sub(repl, name)
                if new != name:
                    yield new, wt, t(tensor)
                    break
            else:
                logger.warning("Flux GGUF: unmapped tensor %s; skipping.", name)


# Registry keyed by model class name -> adapter factory taking (model) and
# returning an iterator of (model_param_name, weight_type, tensor).
def _flux_adapter(model) -> Callable[[Any], Iterator[tuple[str, Any, torch.Tensor]]]:
    cfg = model.config
    inner_dim = cfg.num_attention_heads * cfg.attention_head_dim
    return lambda reader: _flux_gguf_iter(reader, inner_dim)


GGUF_ADAPTERS: dict[str, Callable] = {
    "FluxTransformer2DModel": _flux_adapter,
}


def get_gguf_adapter(model: nn.Module):
    """Return a GGUF name/split adapter for ``model``, or None for diffusers-keyed."""
    factory = GGUF_ADAPTERS.get(type(model).__name__)
    return factory(model) if factory is not None else None


def _maybe_dequantize_gguf(tensor: torch.Tensor, weight_type) -> torch.Tensor:
    """Expand a packed GGUF tensor to a dense float tensor if it is quantized.

    Used when a quantized GGUF tensor maps to a model parameter the model keeps
    in full precision (e.g. an MLP not routed through the quant config). Uses the
    gguf library's reference dequantizer (CPU, supports every GGML type).
    """
    from sglang.srt.layers.quantization.gguf import UNQUANTIZED_TYPES

    if weight_type in UNQUANTIZED_TYPES:
        return tensor
    import gguf

    dequant = gguf.quants.dequantize(tensor.numpy(), weight_type)
    return torch.from_numpy(dequant)


def _read_gguf_tensors(gguf_file: str):
    """Yield ``(name, weight_type, torch_tensor)`` for every tensor in the file.

    ``weight_type`` is the ``gguf.GGMLQuantizationType`` enum; for quantized
    tensors ``torch_tensor`` is the raw packed uint8 blocks, for unquantized
    tensors it is the float data.
    """
    import gguf

    reader = gguf.GGUFReader(gguf_file)
    for tensor in reader.tensors:
        yield tensor.name, tensor.tensor_type, torch.tensor(tensor.data)


def load_gguf_transformer(
    model_cls: type[nn.Module],
    init_params: dict[str, Any],
    gguf_file: str,
    device: torch.device,
    param_dtype: torch.dtype,
) -> nn.Module:
    """Build ``model_cls`` and load a GGUF transformer checkpoint into it.

    ``init_params['quant_config']`` is expected to be a
    `multimodal_gen` ``GGUFConfig`` so that the DiT's linear layers register
    ``qweight`` / ``qweight_type`` parameters.
    """
    from sglang.multimodal_gen.runtime.layers.linear import UnquantizedLinearMethod

    default_dtype = param_dtype or torch.bfloat16
    with set_default_torch_dtype(default_dtype), device:
        model = model_cls(**init_params)

    param_dict = dict(model.named_parameters())

    # A model-specific adapter rewrites reference-format GGUF names to the
    # model's diffusers param names (and splits fused qkv). When present the
    # names are already final, so we map them through identity; otherwise we
    # assume diffusers-keyed names and use the model's param_names_mapping.
    adapter = get_gguf_adapter(model)
    if adapter is not None:
        import gguf

        entries = adapter(gguf.GGUFReader(gguf_file))
        mapping_fn = lambda n: (n, None, None)  # noqa: E731
    else:
        entries = _read_gguf_tensors(gguf_file)
        mapping_fn = get_param_names_mapping(model.param_names_mapping)

    # Partition the GGUF tensors into the quantized-linear stream (handled via
    # the qweight/qweight_type machinery, which can carry per-shard quant types)
    # and the unquantized stream (norms, biases, embeddings, and any layer left
    # in higher precision), which reuses the standard fused-name mapping.
    # quant groups: target_base -> {shard_id_or_None: (weight_type, tensor)}
    quant_groups: dict[str, dict[Any, tuple[Any, torch.Tensor]]] = defaultdict(dict)
    unquantized_stream: list[tuple[str, torch.Tensor]] = []

    for gguf_name, weight_type, tensor in entries:
        target_name, merge_index, _ = mapping_fn(gguf_name)
        if not target_name:
            continue

        base, _, suffix = target_name.rpartition(".")
        # Any `.weight` destined for a quantized linear goes through the qweight
        # path, which also handles unquantized GGUF types (F32/F16/BF16). Biases,
        # norms, embeddings and layers left in higher precision take the plain
        # stream, keyed by their original (source) name so the fused-merge logic
        # in hf_to_custom_state_dict still applies.
        if suffix == "weight" and f"{base}.qweight" in param_dict:
            quant_groups[base][merge_index] = (weight_type, tensor)
        else:
            # Target is a plain (non-quantized) parameter. If the GGUF tensor is
            # quantized (e.g. an MLP the model keeps in full precision), expand
            # it to a dense float tensor before loading.
            unquantized_stream.append(
                (gguf_name, _maybe_dequantize_gguf(tensor, weight_type))
            )

    _load_quantized_groups(quant_groups, param_dict, device)
    _load_unquantized_stream(unquantized_stream, model, param_dict, mapping_fn, device)

    # Materialize merged/padded qweights and validate quant types.
    for _, module in model.named_modules():
        quant_method = getattr(module, "quant_method", None)
        if (
            quant_method is not None
            and not isinstance(quant_method, UnquantizedLinearMethod)
            and hasattr(quant_method, "process_weights_after_loading")
        ):
            quant_method.process_weights_after_loading(module)

    model.post_load_weights()

    for n, p in model.named_parameters():
        if p.is_meta:
            raise RuntimeError(f"Unexpected param {n} on meta device after GGUF load.")
        if isinstance(p, torch.nn.Parameter):
            p.requires_grad = False
    return model


def _load_quantized_groups(
    quant_groups: dict[str, dict[Any, tuple[Any, torch.Tensor]]],
    param_dict: dict[str, torch.nn.Parameter],
    device: torch.device,
) -> None:
    """Load each quantized linear's qweight/qweight_type via their loaders."""
    for base, shards in quant_groups.items():
        qweight = param_dict[f"{base}.qweight"]
        qweight_type = param_dict[f"{base}.qweight_type"]
        weight_loader = qweight.weight_loader

        # A single shard loads with shard_id=None so the parameter is
        # materialized in place; multiple shards (fused QKV / gate-up) accumulate
        # by shard id and are padded+concatenated on post-load.
        is_merged = len(shards) > 1
        for merge_index, (weight_type, tensor) in sorted(
            shards.items(), key=lambda kv: (kv[0] is not None, kv[0])
        ):
            shard_id = merge_index if is_merged else None
            weight_loader(
                qweight_type,
                torch.tensor(int(weight_type), device=device),
                shard_id,
            )
            weight_loader(qweight, tensor.to(device), shard_id)


def _load_unquantized_stream(
    unquantized_stream: list[tuple[str, torch.Tensor]],
    model: nn.Module,
    param_dict: dict[str, torch.nn.Parameter],
    mapping_fn,
    device: torch.device,
) -> None:
    """Load non-quantized tensors (norms, biases, embeddings, fp16 layers).

    Names are already mapped to the custom layout, but merged params (e.g. a
    fused QKV bias) still need to be concatenated, which ``hf_to_custom_state_dict``
    does for us when fed the *original* source names.
    """
    # Re-key by the original source names so the fused-merge logic applies.
    custom_param_sd, reverse_mapping = hf_to_custom_state_dict(
        iter(unquantized_stream),
        mapping_fn,
        valid_target_names=set(param_dict.keys()),
    )
    model.reverse_param_names_mapping = reverse_mapping

    for target_name, full_tensor in custom_param_sd.items():
        param = param_dict.get(target_name)
        if param is None:
            logger.warning("GGUF: no model parameter for %s; skipping.", target_name)
            continue
        full_tensor = full_tensor.to(device=device, dtype=param.dtype)
        weight_loader = getattr(param, "weight_loader", None)
        if weight_loader is not None:
            weight_loader(param, full_tensor)
        else:
            assert param.data.shape == full_tensor.shape, (
                f"GGUF shape mismatch for {target_name}: "
                f"{tuple(param.data.shape)} vs {tuple(full_tensor.shape)}"
            )
            param.data.copy_(full_tensor)
