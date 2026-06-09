import unittest
from unittest.mock import MagicMock

import torch

from sglang.multimodal_gen.runtime.distributed.cfg_policy import CFGBranch, CFGPolicy


def _policy(pos_kwargs, neg_kwargs):
    return CFGPolicy(
        branches=[
            CFGBranch("conditional", True, pos_kwargs),
            CFGBranch("unconditional", False, neg_kwargs),
        ]
    )


class TestBatchedCFG(unittest.TestCase):
    def test_supports_batched_cfg_two_branches(self):
        self.assertTrue(_policy({}, {}).supports_batched_cfg())
        self.assertFalse(CFGPolicy(branches=[CFGBranch("c", True, {})]).supports_batched_cfg())

    def test_merge_concatenates_tensors_in_branch_order(self):
        shared = torch.zeros(1, 4)
        pos = torch.ones(1, 8)
        neg = torch.full((1, 8), 2.0)
        merged = _policy(
            {"img": shared, "txt": pos}, {"img": shared, "txt": neg}
        ).batched_branch_kwargs()
        self.assertEqual(merged["img"].shape, (2, 4))  # shared duplicated
        self.assertEqual(merged["txt"].shape, (2, 8))
        # pos first, neg second
        self.assertTrue(torch.equal(merged["txt"][0], pos[0]))
        self.assertTrue(torch.equal(merged["txt"][1], neg[0]))

    def test_merge_returns_none_on_shape_mismatch(self):
        # cond/uncond prompts padded to different seq lengths -> cannot fuse.
        merged = _policy(
            {"txt": torch.ones(1, 8)}, {"txt": torch.ones(1, 16)}
        ).batched_branch_kwargs()
        self.assertIsNone(merged)

    def test_merge_returns_none_on_dtype_mismatch(self):
        merged = _policy(
            {"txt": torch.ones(1, 8, dtype=torch.float32)},
            {"txt": torch.ones(1, 8, dtype=torch.bfloat16)},
        ).batched_branch_kwargs()
        self.assertIsNone(merged)

    def test_merge_passes_through_matching_non_tensor(self):
        strategy = [[1, 2], [3, 4]]
        merged = _policy(
            {"mask": strategy, "txt": torch.ones(1, 8)},
            {"mask": strategy, "txt": torch.ones(1, 8)},
        ).batched_branch_kwargs()
        self.assertIs(merged["mask"], strategy)

    def test_merge_returns_none_on_differing_non_tensor(self):
        merged = _policy(
            {"mask": [1, 2], "txt": torch.ones(1, 8)},
            {"mask": [3, 4], "txt": torch.ones(1, 8)},
        ).batched_branch_kwargs()
        self.assertIsNone(merged)

    def test_merge_fuses_list_of_tensors_elementwise(self):
        # prompt embeds / attention masks are wrapped in a list per branch.
        pos = [torch.ones(1, 5, 16)]
        neg = [torch.zeros(1, 5, 16)]
        merged = _policy(
            {"encoder_hidden_states": pos}, {"encoder_hidden_states": neg}
        ).batched_branch_kwargs()
        fused = merged["encoder_hidden_states"]
        self.assertIsInstance(fused, list)
        self.assertEqual(fused[0].shape, (2, 5, 16))
        self.assertTrue(torch.equal(fused[0][0], pos[0][0]))
        self.assertTrue(torch.equal(fused[0][1], neg[0][0]))

    def test_merge_list_of_tensors_seq_mismatch_returns_none(self):
        merged = _policy(
            {"e": [torch.ones(1, 5, 16)]}, {"e": [torch.ones(1, 9, 16)]}
        ).batched_branch_kwargs()
        self.assertIsNone(merged)

    def test_merge_shared_nested_structure_passes_through(self):
        # STA mask grid: same object shared across branches, never walked.
        grid = [[None] * 3 for _ in range(3)]
        merged = _policy(
            {"mask_strategy": grid, "txt": torch.ones(1, 8)},
            {"mask_strategy": grid, "txt": torch.ones(1, 8)},
        ).batched_branch_kwargs()
        self.assertIs(merged["mask_strategy"], grid)

    def test_merge_returns_none_on_missing_key(self):
        merged = _policy(
            {"a": torch.ones(1, 8), "b": torch.ones(1, 8)},
            {"a": torch.ones(1, 8)},
        ).batched_branch_kwargs()
        self.assertIsNone(merged)


class TestCFGPolicyCombine(unittest.TestCase):
    def test_cfg_parallel_uses_parallel_arithmetic_order(self):
        policy = CFGPolicy()
        req = MagicMock()
        req.cfg_normalization = 0
        req.guidance_rescale = 0

        pipeline_config = MagicMock()
        pipeline_config.postprocess_cfg_noise.side_effect = lambda _, noise, __: noise

        pos = torch.tensor([1.0], dtype=torch.bfloat16)
        neg = torch.tensor([0.1], dtype=torch.bfloat16)

        serial = policy.combine([pos, neg], req, 7.0, pipeline_config)
        parallel = policy.combine(
            [pos, neg], req, 7.0, pipeline_config, cfg_parallel=True
        )

        self.assertTrue(torch.equal(serial, neg + 7.0 * (pos - neg)))
        self.assertTrue(torch.equal(parallel, 7.0 * pos + (1 - 7.0) * neg))
        self.assertFalse(torch.equal(serial, parallel))


if __name__ == "__main__":
    unittest.main()
