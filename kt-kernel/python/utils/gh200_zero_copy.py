import os
from typing import List, Optional

import torch

from ..experts_base import BaseMoEWrapper
from .loader import BF16SafeTensorLoader

try:
    from kt_kernel_ext import gh200 as _gh200_ext
except (ImportError, AttributeError):
    _gh200_ext = None


def _truthy_env(name: str, default: bool = False) -> bool:
    value = os.getenv(name)
    if value is None:
        return default
    return value.strip().lower() in {"1", "true", "yes", "on", "enabled"}


class GH200ZeroCopyMoEWrapper(BaseMoEWrapper):
    """BF16 MoE wrapper that lets Hopper read Grace-resident expert weights directly.

    This is an opt-in GH200 backend for correctness-first validation. It preserves
    the existing KT hot/cold expert mask semantics: experts marked in
    ``gpu_experts_mask`` are skipped here because the normal GPU path owns them;
    unmasked experts are computed by CUDA kernels reading mapped host pointers.
    """

    _loader_instance = None

    def __init__(
        self,
        layer_idx: int,
        num_experts: int,
        num_experts_per_tok: int,
        hidden_size: int,
        moe_intermediate_size: int,
        gpu_experts_mask: Optional[torch.Tensor],
        cpuinfer_threads: int,
        threadpool_count: int,
        weight_path: str,
        chunked_prefill_size: int,
        cpu_save: bool = False,
        max_deferred_experts_per_token: Optional[int] = None,
        method: str = "BF16",
        numa_nodes: Optional[List[int]] = None,
    ):
        if method != "BF16":
            raise ValueError(f"GH200ZeroCopyMoEWrapper only supports BF16, got {method}.")
        if _gh200_ext is None or not _gh200_ext.is_available():
            raise RuntimeError(
                "KT_GH200_ZERO_COPY=1 requires kt_kernel_ext built with CUDA "
                "(set CPUINFER_USE_CUDA=1 and CPUINFER_CUDA_ARCHS=90 on GH200)."
            )

        super().__init__(
            layer_idx=layer_idx,
            num_experts=num_experts,
            num_experts_per_tok=num_experts_per_tok,
            hidden_size=hidden_size,
            moe_intermediate_size=moe_intermediate_size,
            gpu_experts_mask=gpu_experts_mask,
            cpuinfer_threads=cpuinfer_threads,
            threadpool_count=threadpool_count,
            weight_path=weight_path,
            chunked_prefill_size=chunked_prefill_size,
            cpu_save=cpu_save,
            max_deferred_experts_per_token=max_deferred_experts_per_token,
            method=method,
            numa_nodes=numa_nodes,
        )

        self.gate_weights: Optional[torch.Tensor] = None
        self.up_weights: Optional[torch.Tensor] = None
        self.down_weights: Optional[torch.Tensor] = None

        self._host_registrations: list[int] = []
        self._gate_base_device_ptr = 0
        self._up_base_device_ptr = 0
        self._down_base_device_ptr = 0
        self._gate_device_ptrs = [0] * self.num_experts
        self._up_device_ptrs = [0] * self.num_experts
        self._down_device_ptrs = [0] * self.num_experts
        self._physical_to_logical_map = list(range(self.num_experts))

        self._device_tables_device: Optional[torch.device] = None
        self._gate_ptrs_gpu: Optional[torch.Tensor] = None
        self._up_ptrs_gpu: Optional[torch.Tensor] = None
        self._down_ptrs_gpu: Optional[torch.Tensor] = None
        self._act_tmp: Optional[torch.Tensor] = None
        self._act_tmp_shape: Optional[tuple[int, int, int]] = None
        self._pending_output: Optional[torch.Tensor] = None
        self._retired_device_tables: list[tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]] = []

    def __del__(self):
        try:
            self._unregister_all()
        except Exception:
            pass

    def _unregister_all(self):
        if _gh200_ext is None:
            return
        for host_ptr in self._host_registrations:
            try:
                _gh200_ext.unregister_mapped_host(int(host_ptr))
            except Exception:
                pass
        self._host_registrations.clear()
        self._gate_base_device_ptr = 0
        self._up_base_device_ptr = 0
        self._down_base_device_ptr = 0

    def _register_tensor(self, tensor: torch.Tensor) -> int:
        if tensor.device.type != "cpu" or tensor.dtype != torch.bfloat16 or not tensor.is_contiguous():
            raise ValueError("GH200 mapped expert tensors must be contiguous CPU BF16 tensors.")
        host_ptr = int(tensor.data_ptr())
        nbytes = int(tensor.numel() * tensor.element_size())
        device_ptr = int(_gh200_ext.register_mapped_host(host_ptr, nbytes, True))
        self._host_registrations.append(host_ptr)
        return device_ptr

    def _gate_up_expert_offset(self, logical_id: int) -> int:
        return int(logical_id) * self.moe_intermediate_size * self.hidden_size * 2

    def _down_expert_offset(self, logical_id: int) -> int:
        return int(logical_id) * self.hidden_size * self.moe_intermediate_size * 2

    def _install_device_ptr_for_physical_expert(self, physical_id: int):
        logical_id = self._physical_to_logical_map[physical_id]
        self._gate_device_ptrs[physical_id] = (
            self._gate_base_device_ptr + self._gate_up_expert_offset(logical_id)
        )
        self._up_device_ptrs[physical_id] = (
            self._up_base_device_ptr + self._gate_up_expert_offset(logical_id)
        )
        self._down_device_ptrs[physical_id] = (
            self._down_base_device_ptr + self._down_expert_offset(logical_id)
        )

    def _physical_to_logical(self, physical_to_logical_map_cpu: torch.Tensor) -> list[int]:
        if physical_to_logical_map_cpu is None:
            return list(range(self.num_experts))
        mapping = physical_to_logical_map_cpu.to("cpu", non_blocking=False).view(-1).tolist()
        if len(mapping) < self.num_experts:
            raise ValueError(
                f"physical_to_logical_map has {len(mapping)} entries, expected at least {self.num_experts}."
            )
        return [int(v) for v in mapping[: self.num_experts]]

    def _install_weight_lists(
        self,
        gate_weights: list[torch.Tensor],
        up_weights: list[torch.Tensor],
        down_weights: list[torch.Tensor],
        physical_to_logical_map_cpu: torch.Tensor,
    ):
        if (
            len(gate_weights) < self.num_experts
            or len(up_weights) < self.num_experts
            or len(down_weights) < self.num_experts
        ):
            raise ValueError(
                "GH200 BF16 weight loader returned fewer experts than the model config: "
                f"gate={len(gate_weights)}, up={len(up_weights)}, down={len(down_weights)}, "
                f"expected={self.num_experts}."
            )

        self._unregister_all()
        self._gate_device_ptrs = [0] * self.num_experts
        self._up_device_ptrs = [0] * self.num_experts
        self._down_device_ptrs = [0] * self.num_experts

        self._physical_to_logical_map = self._physical_to_logical(physical_to_logical_map_cpu)

        self.gate_weights = torch.stack(
            [gate_weights[int(i)].to(device="cpu", dtype=torch.bfloat16) for i in range(self.num_experts)],
            dim=0,
        ).contiguous()
        self.up_weights = torch.stack(
            [up_weights[int(i)].to(device="cpu", dtype=torch.bfloat16) for i in range(self.num_experts)],
            dim=0,
        ).contiguous()
        self.down_weights = torch.stack(
            [down_weights[int(i)].to(device="cpu", dtype=torch.bfloat16) for i in range(self.num_experts)],
            dim=0,
        ).contiguous()

        self._gate_base_device_ptr = self._register_tensor(self.gate_weights)
        self._up_base_device_ptr = self._register_tensor(self.up_weights)
        self._down_base_device_ptr = self._register_tensor(self.down_weights)

        register_all = _truthy_env("KT_GH200_REGISTER_ALL_EXPERTS", False)
        for physical_id in range(self.num_experts):
            if self.gpu_experts_mask[physical_id].item() and not register_all:
                continue
            self._install_device_ptr_for_physical_expert(physical_id)

        self._device_tables_device = None
        print(
            f"[GH200ZeroCopyMoEWrapper Layer {self.layer_idx}] registered "
            f"gate/up/down BF16 tensors as mapped host memory; exposed "
            f"{sum(1 for ptr in self._gate_device_ptrs if ptr)} / {self.num_experts} expert pointers "
            f"({'all' if register_all else 'cold-only'})"
        )

    def load_weights_from_tensors(
        self,
        gate_proj: torch.Tensor,
        up_proj: torch.Tensor,
        down_proj: torch.Tensor,
        physical_to_logical_map_cpu: torch.Tensor,
    ):
        gate_proj = gate_proj.to("cpu").contiguous()
        up_proj = up_proj.to("cpu").contiguous()
        down_proj = down_proj.to("cpu").contiguous()
        gate_weights = [gate_proj[i] for i in range(gate_proj.shape[0])]
        up_weights = [up_proj[i] for i in range(up_proj.shape[0])]
        down_weights = [down_proj[i] for i in range(down_proj.shape[0])]
        self._install_weight_lists(gate_weights, up_weights, down_weights, physical_to_logical_map_cpu)

    def load_weights(self, physical_to_logical_map_cpu: torch.Tensor):
        if GH200ZeroCopyMoEWrapper._loader_instance is None:
            GH200ZeroCopyMoEWrapper._loader_instance = BF16SafeTensorLoader(self.weight_path)
        loader = GH200ZeroCopyMoEWrapper._loader_instance

        base_key = f"model.layers.{self.layer_idx}"
        try:
            weights = loader.load_experts(base_key, device="cpu")
        except (ValueError, KeyError):
            base_key = f"model.language_model.layers.{self.layer_idx}"
            weights = loader.load_experts(base_key, device="cpu")

        self._install_weight_lists(weights["gate"], weights["up"], weights["down"], physical_to_logical_map_cpu)

    def _ensure_registered_for_current_cold_mask(self) -> bool:
        """Expose pointers for experts that became cold after a dynamic mask update."""
        changed = False
        for physical_id in range(self.num_experts):
            if self.gpu_experts_mask[physical_id].item() or self._gate_device_ptrs[physical_id] != 0:
                continue
            self._install_device_ptr_for_physical_expert(physical_id)
            changed = True
        return changed

    def _ensure_device_tables(self, device: torch.device):
        changed = self._ensure_registered_for_current_cold_mask()
        if self._device_tables_device == device and not changed:
            assert self._gate_ptrs_gpu is not None
            self._gpu_experts_mask_gpu.copy_(self.gpu_experts_mask, non_blocking=True)
            return

        if (
            self._gate_ptrs_gpu is not None
            and self._up_ptrs_gpu is not None
            and self._down_ptrs_gpu is not None
            and self._gpu_experts_mask_gpu is not None
        ):
            self._retired_device_tables.append(
                (self._gate_ptrs_gpu, self._up_ptrs_gpu, self._down_ptrs_gpu, self._gpu_experts_mask_gpu)
            )

        self._gate_ptrs_gpu = torch.tensor(self._gate_device_ptrs, dtype=torch.long, device=device)
        self._up_ptrs_gpu = torch.tensor(self._up_device_ptrs, dtype=torch.long, device=device)
        self._down_ptrs_gpu = torch.tensor(self._down_device_ptrs, dtype=torch.long, device=device)
        self._gpu_experts_mask_gpu = self.gpu_experts_mask.to(device=device, non_blocking=True)
        self._device_tables_device = device

    def _ensure_act_tmp(self, batch_size: int, topk: int, device: torch.device) -> torch.Tensor:
        shape = (batch_size, topk, self.moe_intermediate_size)
        if self._act_tmp is None or self._act_tmp_shape != shape or self._act_tmp.device != device:
            self._act_tmp = torch.empty(shape, dtype=torch.float32, device=device)
            self._act_tmp_shape = shape
        return self._act_tmp

    def _stream_ptr(self, cuda_stream, device: torch.device) -> int:
        if cuda_stream is not None:
            return int(cuda_stream)
        return int(torch.cuda.current_stream(device).cuda_stream)

    def _run_forward(
        self,
        hidden_states: torch.Tensor,
        topk_ids: torch.Tensor,
        topk_weights: torch.Tensor,
        cuda_stream,
    ) -> torch.Tensor:
        if self.gate_weights is None:
            raise RuntimeError("GH200ZeroCopyMoEWrapper weights are not loaded. Call load_weights() first.")
        if hidden_states.device.type != "cuda":
            raise ValueError("GH200ZeroCopyMoEWrapper expects hidden_states on CUDA.")
        if hidden_states.dtype != torch.bfloat16:
            raise ValueError(f"GH200ZeroCopyMoEWrapper expects BF16 hidden_states, got {hidden_states.dtype}.")
        if hidden_states.shape[-1] != self.hidden_size:
            raise ValueError(f"Expected hidden size {self.hidden_size}, got {hidden_states.shape[-1]}.")

        flat_hidden = hidden_states.reshape(-1, hidden_states.shape[-1]).contiguous()
        batch_size = flat_hidden.shape[0]
        topk = topk_ids.shape[-1]
        if topk != self.num_experts_per_tok:
            raise ValueError(f"Expected topk={self.num_experts_per_tok}, got {topk}.")

        device = flat_hidden.device
        self._ensure_device_tables(device)
        act_tmp = self._ensure_act_tmp(batch_size, topk, device)

        topk_ids_i64 = topk_ids.reshape(batch_size, topk).to(device=device, dtype=torch.long).contiguous()
        topk_weights_f32 = topk_weights.reshape(batch_size, topk).to(device=device, dtype=torch.float32).contiguous()
        output = torch.empty((batch_size, self.hidden_size), dtype=torch.bfloat16, device=device)

        _gh200_ext.bf16_moe_forward(
            int(flat_hidden.data_ptr()),
            int(topk_ids_i64.data_ptr()),
            int(topk_weights_f32.data_ptr()),
            int(output.data_ptr()),
            int(act_tmp.data_ptr()),
            int(self._gate_ptrs_gpu.data_ptr()),
            int(self._up_ptrs_gpu.data_ptr()),
            int(self._down_ptrs_gpu.data_ptr()),
            int(self._gpu_experts_mask_gpu.data_ptr()),
            int(batch_size),
            int(self.num_experts),
            int(self.hidden_size),
            int(self.moe_intermediate_size),
            int(topk),
            self._stream_ptr(cuda_stream, device),
        )
        return output

    def submit_forward(
        self,
        hidden_states: torch.Tensor,
        topk_ids: torch.Tensor,
        topk_weights: torch.Tensor,
        cuda_stream,
    ):
        self._pending_output = self._run_forward(hidden_states, topk_ids, topk_weights, cuda_stream)

    def sync_forward(self, hidden_states: torch.Tensor, cuda_stream) -> torch.Tensor:
        if self._pending_output is None:
            raise RuntimeError("No pending GH200 forward. Call submit_forward() first.")
        output = self._pending_output
        self._pending_output = None
        return output

    def forward(
        self,
        hidden_states: torch.Tensor,
        topk_ids: torch.Tensor,
        topk_weights: torch.Tensor,
        cuda_stream,
    ) -> torch.Tensor:
        return self._run_forward(hidden_states, topk_ids, topk_weights, cuda_stream)
