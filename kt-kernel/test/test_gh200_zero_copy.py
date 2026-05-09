"""Tests for the opt-in GH200 zero-copy BF16 MoE backend."""

from __future__ import annotations

import pytest
import torch


def _has_gh200_cuda_backend() -> bool:
    if not torch.cuda.is_available():
        return False
    try:
        import kt_kernel

        return hasattr(kt_kernel.kt_kernel_ext, "gh200") and kt_kernel.kt_kernel_ext.gh200.is_available()
    except Exception:
        return False


def _reference_cold_moe(hidden, topk_ids, topk_weights, gate, up, down, gpu_mask):
    hidden_f = hidden.float().cpu()
    topk_ids_cpu = topk_ids.cpu()
    topk_weights_cpu = topk_weights.float().cpu()
    gate_f = gate.float()
    up_f = up.float()
    down_f = down.float()

    batch, hidden_size = hidden_f.shape
    output = torch.zeros((batch, hidden_size), dtype=torch.float32)
    for token in range(batch):
        for route in range(topk_ids_cpu.shape[1]):
            expert_id = int(topk_ids_cpu[token, route])
            if expert_id < 0 or expert_id >= gate.shape[0] or bool(gpu_mask[expert_id]):
                continue
            gate_out = torch.matmul(gate_f[expert_id], hidden_f[token])
            up_out = torch.matmul(up_f[expert_id], hidden_f[token])
            act = torch.nn.functional.silu(gate_out) * up_out
            expert_out = torch.matmul(down_f[expert_id], act)
            output[token] += expert_out * topk_weights_cpu[token, route]
    return output.to(torch.bfloat16)


@pytest.mark.skipif(not _has_gh200_cuda_backend(), reason="CUDA-enabled kt_kernel_ext.gh200 backend is unavailable")
@pytest.mark.parametrize(
    "gpu_mask",
    [
        torch.tensor([False, False, False, False], dtype=torch.bool),
        torch.tensor([True, True, True, True], dtype=torch.bool),
        torch.tensor([False, True, False, False], dtype=torch.bool),
    ],
    ids=["cold-only", "hot-only", "mixed"],
)
def test_gh200_zero_copy_bf16_matches_reference_for_masks(gpu_mask):
    from kt_kernel.utils.gh200_zero_copy import GH200ZeroCopyMoEWrapper

    torch.manual_seed(0)
    num_experts = 4
    topk = 2
    hidden_size = 8
    intermediate_size = 4
    batch = 3

    wrapper = GH200ZeroCopyMoEWrapper(
        layer_idx=0,
        num_experts=num_experts,
        num_experts_per_tok=topk,
        hidden_size=hidden_size,
        moe_intermediate_size=intermediate_size,
        gpu_experts_mask=gpu_mask,
        cpuinfer_threads=1,
        threadpool_count=1,
        weight_path="",
        chunked_prefill_size=batch,
        method="BF16",
    )

    gate = torch.randn((num_experts, intermediate_size, hidden_size), dtype=torch.bfloat16)
    up = torch.randn((num_experts, intermediate_size, hidden_size), dtype=torch.bfloat16)
    down = torch.randn((num_experts, hidden_size, intermediate_size), dtype=torch.bfloat16)
    wrapper.load_weights_from_tensors(gate, up, down, torch.arange(num_experts, dtype=torch.long))

    hidden = torch.randn((batch, hidden_size), dtype=torch.bfloat16, device="cuda")
    topk_ids = torch.tensor([[0, 1], [2, 3], [1, 3]], dtype=torch.long, device="cuda")
    topk_weights = torch.tensor([[0.7, 0.3], [0.4, 0.6], [0.25, 0.75]], dtype=torch.float32, device="cuda")

    stream = torch.cuda.current_stream().cuda_stream
    actual = wrapper.forward(hidden, topk_ids, topk_weights, stream)
    torch.cuda.synchronize()

    expected = _reference_cold_moe(hidden, topk_ids, topk_weights, gate, up, down, gpu_mask).to(device="cuda")
    torch.testing.assert_close(actual, expected, rtol=0.08, atol=0.08)


def test_gh200_zero_copy_env_is_opt_in(monkeypatch):
    experts_mod = pytest.importorskip("kt_kernel.experts")

    class DummyNativeMoEWrapper:
        def __init__(self, **kwargs):
            self.kwargs = kwargs

    monkeypatch.delenv("KT_GH200_ZERO_COPY", raising=False)
    monkeypatch.setattr(experts_mod, "NativeMoEWrapper", DummyNativeMoEWrapper)

    wrapper = experts_mod._create_inference_wrapper(
        layer_idx=0,
        num_experts=1,
        num_experts_per_tok=1,
        hidden_size=8,
        moe_intermediate_size=4,
        gpu_experts_mask=torch.zeros(1, dtype=torch.bool),
        cpuinfer_threads=1,
        threadpool_count=1,
        weight_path="",
        chunked_prefill_size=1,
        cpu_save=False,
        max_deferred_experts_per_token=None,
        method="BF16",
    )
    assert isinstance(wrapper, DummyNativeMoEWrapper)
