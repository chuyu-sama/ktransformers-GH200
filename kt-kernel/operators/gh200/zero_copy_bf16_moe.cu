#include "zero_copy_bf16_moe.h"

#ifdef KTRANSFORMERS_USE_CUDA

#include <cuda_bf16.h>
#include <cuda_runtime.h>

#include <algorithm>
#include <cstdint>
#include <sstream>
#include <stdexcept>
#include <string>
#include <unistd.h>

namespace {

inline void check_cuda(cudaError_t err, const char* what) {
  if (err != cudaSuccess) {
    std::ostringstream oss;
    oss << what << " failed: " << cudaGetErrorString(err);
    throw std::runtime_error(oss.str());
  }
}

__device__ __forceinline__ float bf16_to_float(const __nv_bfloat16 x) {
  return __bfloat162float(x);
}

__device__ __forceinline__ float silu(const float x) {
  return x / (1.0f + __expf(-x));
}

__global__ void bf16_gate_up_kernel(const __nv_bfloat16* __restrict__ hidden,
                                    const int64_t* __restrict__ topk_ids,
                                    float* __restrict__ act_tmp,
                                    const uintptr_t* __restrict__ gate_ptrs,
                                    const uintptr_t* __restrict__ up_ptrs,
                                    const uint8_t* __restrict__ gpu_experts_mask,
                                    int batch_size,
                                    int expert_num,
                                    int hidden_size,
                                    int intermediate_size,
                                    int topk) {
  const int64_t total = static_cast<int64_t>(batch_size) * topk * intermediate_size;
  for (int64_t idx = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x; idx < total;
       idx += static_cast<int64_t>(blockDim.x) * gridDim.x) {
    const int inter_idx = idx % intermediate_size;
    const int topk_idx = (idx / intermediate_size) % topk;
    const int token_idx = idx / (static_cast<int64_t>(topk) * intermediate_size);
    const int64_t expert_id = topk_ids[static_cast<int64_t>(token_idx) * topk + topk_idx];

    if (expert_id < 0 || expert_id >= expert_num || gpu_experts_mask[expert_id] || gate_ptrs[expert_id] == 0 ||
        up_ptrs[expert_id] == 0) {
      act_tmp[idx] = 0.0f;
      continue;
    }

    const auto* gate = reinterpret_cast<const __nv_bfloat16*>(gate_ptrs[expert_id]);
    const auto* up = reinterpret_cast<const __nv_bfloat16*>(up_ptrs[expert_id]);
    const auto* x = hidden + static_cast<int64_t>(token_idx) * hidden_size;
    const int64_t row_offset = static_cast<int64_t>(inter_idx) * hidden_size;

    float gate_acc = 0.0f;
    float up_acc = 0.0f;
    for (int h = 0; h < hidden_size; ++h) {
      const float xv = bf16_to_float(x[h]);
      gate_acc = fmaf(xv, bf16_to_float(gate[row_offset + h]), gate_acc);
      up_acc = fmaf(xv, bf16_to_float(up[row_offset + h]), up_acc);
    }

    act_tmp[idx] = silu(gate_acc) * up_acc;
  }
}

__global__ void bf16_down_kernel(const int64_t* __restrict__ topk_ids,
                                 const float* __restrict__ topk_weights,
                                 __nv_bfloat16* __restrict__ output,
                                 const float* __restrict__ act_tmp,
                                 const uintptr_t* __restrict__ down_ptrs,
                                 const uint8_t* __restrict__ gpu_experts_mask,
                                 int batch_size,
                                 int expert_num,
                                 int hidden_size,
                                 int intermediate_size,
                                 int topk) {
  const int64_t total = static_cast<int64_t>(batch_size) * hidden_size;
  for (int64_t idx = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x; idx < total;
       idx += static_cast<int64_t>(blockDim.x) * gridDim.x) {
    const int hidden_idx = idx % hidden_size;
    const int token_idx = idx / hidden_size;

    float sum = 0.0f;
    for (int topk_idx = 0; topk_idx < topk; ++topk_idx) {
      const int64_t route_idx = static_cast<int64_t>(token_idx) * topk + topk_idx;
      const int64_t expert_id = topk_ids[route_idx];
      if (expert_id < 0 || expert_id >= expert_num || gpu_experts_mask[expert_id] || down_ptrs[expert_id] == 0) {
        continue;
      }

      const auto* down = reinterpret_cast<const __nv_bfloat16*>(down_ptrs[expert_id]);
      const float* act = act_tmp + route_idx * intermediate_size;
      const int64_t row_offset = static_cast<int64_t>(hidden_idx) * intermediate_size;

      float expert_sum = 0.0f;
      for (int i = 0; i < intermediate_size; ++i) {
        expert_sum = fmaf(act[i], bf16_to_float(down[row_offset + i]), expert_sum);
      }
      sum = fmaf(expert_sum, topk_weights[route_idx], sum);
    }

    output[idx] = __float2bfloat16(sum);
  }
}

__device__ __forceinline__ bool is_valid_cold_route(const int64_t expert_id,
                                                    int expert_num,
                                                    const uintptr_t* __restrict__ gate_ptrs,
                                                    const uintptr_t* __restrict__ up_ptrs,
                                                    const uintptr_t* __restrict__ down_ptrs,
                                                    const uint8_t* __restrict__ gpu_experts_mask) {
  return expert_id >= 0 && expert_id < expert_num && !gpu_experts_mask[expert_id] && gate_ptrs[expert_id] != 0 &&
         up_ptrs[expert_id] != 0 && down_ptrs[expert_id] != 0;
}

__global__ void count_cold_routes_kernel(const int64_t* __restrict__ topk_ids,
                                         int* __restrict__ route_counts,
                                         const uintptr_t* __restrict__ gate_ptrs,
                                         const uintptr_t* __restrict__ up_ptrs,
                                         const uintptr_t* __restrict__ down_ptrs,
                                         const uint8_t* __restrict__ gpu_experts_mask,
                                         int batch_size,
                                         int expert_num,
                                         int topk) {
  const int64_t total_routes = static_cast<int64_t>(batch_size) * topk;
  for (int64_t route_idx = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x; route_idx < total_routes;
       route_idx += static_cast<int64_t>(blockDim.x) * gridDim.x) {
    const int64_t expert_id = topk_ids[route_idx];
    if (is_valid_cold_route(expert_id, expert_num, gate_ptrs, up_ptrs, down_ptrs, gpu_experts_mask)) {
      atomicAdd(route_counts + expert_id, 1);
    }
  }
}

__global__ void prefix_route_offsets_kernel(const int* __restrict__ route_counts,
                                            int* __restrict__ route_offsets,
                                            int expert_num) {
  if (blockIdx.x != 0 || threadIdx.x != 0) {
    return;
  }

  int running = 0;
  route_offsets[0] = 0;
  for (int expert = 0; expert < expert_num; ++expert) {
    running += route_counts[expert];
    route_offsets[expert + 1] = running;
  }
}

__global__ void scatter_cold_routes_kernel(const int64_t* __restrict__ topk_ids,
                                           int* __restrict__ route_cursors,
                                           const int* __restrict__ route_offsets,
                                           int* __restrict__ route_tokens,
                                           int* __restrict__ route_topks,
                                           int* __restrict__ route_experts,
                                           const uintptr_t* __restrict__ gate_ptrs,
                                           const uintptr_t* __restrict__ up_ptrs,
                                           const uintptr_t* __restrict__ down_ptrs,
                                           const uint8_t* __restrict__ gpu_experts_mask,
                                           int batch_size,
                                           int expert_num,
                                           int topk) {
  const int64_t total_routes = static_cast<int64_t>(batch_size) * topk;
  for (int64_t route_idx = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x; route_idx < total_routes;
       route_idx += static_cast<int64_t>(blockDim.x) * gridDim.x) {
    const int64_t expert_id = topk_ids[route_idx];
    if (!is_valid_cold_route(expert_id, expert_num, gate_ptrs, up_ptrs, down_ptrs, gpu_experts_mask)) {
      continue;
    }

    const int token_idx = static_cast<int>(route_idx / topk);
    const int topk_idx = static_cast<int>(route_idx % topk);
    const int slot = route_offsets[expert_id] + atomicAdd(route_cursors + expert_id, 1);
    route_tokens[slot] = token_idx;
    route_topks[slot] = topk_idx;
    route_experts[slot] = static_cast<int>(expert_id);
  }
}

__global__ void grouped_bf16_gate_up_kernel(const __nv_bfloat16* __restrict__ hidden,
                                            const int* __restrict__ route_tokens,
                                            const int* __restrict__ route_topks,
                                            const int* __restrict__ route_experts,
                                            float* __restrict__ act_tmp,
                                            const uintptr_t* __restrict__ gate_ptrs,
                                            const uintptr_t* __restrict__ up_ptrs,
                                            int total_routes,
                                            int hidden_size,
                                            int intermediate_size,
                                            int topk) {
  const int64_t total = static_cast<int64_t>(total_routes) * intermediate_size;
  for (int64_t idx = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x; idx < total;
       idx += static_cast<int64_t>(blockDim.x) * gridDim.x) {
    const int inter_idx = idx % intermediate_size;
    const int route_slot = idx / intermediate_size;
    const int token_idx = route_tokens[route_slot];
    if (token_idx < 0) {
      continue;
    }

    const int topk_idx = route_topks[route_slot];
    const int expert_id = route_experts[route_slot];
    const auto* gate = reinterpret_cast<const __nv_bfloat16*>(gate_ptrs[expert_id]);
    const auto* up = reinterpret_cast<const __nv_bfloat16*>(up_ptrs[expert_id]);
    const auto* x = hidden + static_cast<int64_t>(token_idx) * hidden_size;
    const int64_t row_offset = static_cast<int64_t>(inter_idx) * hidden_size;

    float gate_acc = 0.0f;
    float up_acc = 0.0f;
    for (int h = 0; h < hidden_size; ++h) {
      const float xv = bf16_to_float(x[h]);
      gate_acc = fmaf(xv, bf16_to_float(gate[row_offset + h]), gate_acc);
      up_acc = fmaf(xv, bf16_to_float(up[row_offset + h]), up_acc);
    }

    const int64_t original_route_idx = static_cast<int64_t>(token_idx) * topk + topk_idx;
    act_tmp[original_route_idx * intermediate_size + inter_idx] = silu(gate_acc) * up_acc;
  }
}

__global__ void grouped_bf16_down_kernel(const float* __restrict__ topk_weights,
                                         float* __restrict__ output_accum,
                                         const float* __restrict__ act_tmp,
                                         const int* __restrict__ route_tokens,
                                         const int* __restrict__ route_topks,
                                         const int* __restrict__ route_experts,
                                         const uintptr_t* __restrict__ down_ptrs,
                                         int total_routes,
                                         int hidden_size,
                                         int intermediate_size,
                                         int topk) {
  const int64_t total = static_cast<int64_t>(total_routes) * hidden_size;
  for (int64_t idx = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x; idx < total;
       idx += static_cast<int64_t>(blockDim.x) * gridDim.x) {
    const int hidden_idx = idx % hidden_size;
    const int route_slot = idx / hidden_size;
    const int token_idx = route_tokens[route_slot];
    if (token_idx < 0) {
      continue;
    }

    const int topk_idx = route_topks[route_slot];
    const int expert_id = route_experts[route_slot];
    const auto* down = reinterpret_cast<const __nv_bfloat16*>(down_ptrs[expert_id]);
    const int64_t original_route_idx = static_cast<int64_t>(token_idx) * topk + topk_idx;
    const float* act = act_tmp + original_route_idx * intermediate_size;
    const int64_t row_offset = static_cast<int64_t>(hidden_idx) * intermediate_size;

    float expert_sum = 0.0f;
    for (int i = 0; i < intermediate_size; ++i) {
      expert_sum = fmaf(act[i], bf16_to_float(down[row_offset + i]), expert_sum);
    }
    atomicAdd(output_accum + static_cast<int64_t>(token_idx) * hidden_size + hidden_idx,
              expert_sum * topk_weights[original_route_idx]);
  }
}

__global__ void cast_float_to_bf16_kernel(const float* __restrict__ input,
                                          __nv_bfloat16* __restrict__ output,
                                          int64_t total) {
  for (int64_t idx = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x; idx < total;
       idx += static_cast<int64_t>(blockDim.x) * gridDim.x) {
    output[idx] = __float2bfloat16(input[idx]);
  }
}

int launch_grid(int64_t work_items) {
  constexpr int block = 128;
  int grid = static_cast<int>((work_items + block - 1) / block);
  return std::max(1, std::min(grid, 65535));
}

struct HostRegistrationRange {
  uintptr_t base = 0;
  size_t bytes = 0;
  size_t offset = 0;
};

size_t host_page_size() {
  long page_size = sysconf(_SC_PAGESIZE);
  return page_size > 0 ? static_cast<size_t>(page_size) : 4096;
}

HostRegistrationRange get_host_registration_range(uintptr_t host_ptr, size_t bytes) {
  const size_t page_size = host_page_size();
  const uintptr_t page_mask = static_cast<uintptr_t>(page_size - 1);
  const uintptr_t base = host_ptr & ~page_mask;
  const uintptr_t end = host_ptr + bytes;
  if (end < host_ptr) {
    throw std::runtime_error("cudaHostRegister range overflow.");
  }
  const uintptr_t aligned_end = (end + page_mask) & ~page_mask;
  return {base, static_cast<size_t>(aligned_end - base), static_cast<size_t>(host_ptr - base)};
}

}  // namespace

uintptr_t gh200_register_mapped_host(uintptr_t host_ptr, size_t bytes, bool read_only) {
  if (host_ptr == 0 || bytes == 0) {
    return 0;
  }

  const HostRegistrationRange range = get_host_registration_range(host_ptr, bytes);

  unsigned int flags = cudaHostRegisterMapped | cudaHostRegisterPortable;
#if defined(CUDART_VERSION) && CUDART_VERSION >= 11020
  if (read_only) {
    flags |= cudaHostRegisterReadOnly;
  }
#else
  (void)read_only;
#endif

  cudaError_t err = cudaHostRegister(reinterpret_cast<void*>(range.base), range.bytes, flags);
#if defined(CUDART_VERSION) && CUDART_VERSION >= 11020
  if (err == cudaErrorNotSupported && (flags & cudaHostRegisterReadOnly)) {
    (void)cudaGetLastError();
    flags &= ~cudaHostRegisterReadOnly;
    err = cudaHostRegister(reinterpret_cast<void*>(range.base), range.bytes, flags);
  }
#endif
  if (err == cudaErrorHostMemoryAlreadyRegistered) {
    (void)cudaGetLastError();
  } else {
    check_cuda(err, "cudaHostRegister");
  }

  void* device_base_ptr = nullptr;
  check_cuda(cudaHostGetDevicePointer(&device_base_ptr, reinterpret_cast<void*>(range.base), 0),
             "cudaHostGetDevicePointer");
  return reinterpret_cast<uintptr_t>(device_base_ptr) + range.offset;
}

void gh200_unregister_mapped_host(uintptr_t host_ptr) {
  if (host_ptr == 0) {
    return;
  }
  const uintptr_t registered_base = get_host_registration_range(host_ptr, 1).base;
  cudaError_t err = cudaHostUnregister(reinterpret_cast<void*>(registered_base));
  if (err == cudaErrorHostMemoryNotRegistered) {
    (void)cudaGetLastError();
    return;
  }
  check_cuda(err, "cudaHostUnregister");
}

void gh200_bf16_moe_forward(uintptr_t hidden_ptr, uintptr_t topk_ids_ptr, uintptr_t topk_weights_ptr,
                            uintptr_t output_ptr, uintptr_t act_tmp_ptr, uintptr_t gate_ptrs_ptr,
                            uintptr_t up_ptrs_ptr, uintptr_t down_ptrs_ptr, uintptr_t gpu_experts_mask_ptr,
                            int batch_size, int expert_num, int hidden_size, int intermediate_size, int topk,
                            uintptr_t stream_ptr) {
  if (batch_size <= 0 || expert_num <= 0 || hidden_size <= 0 || intermediate_size <= 0 || topk <= 0) {
    throw std::runtime_error("gh200_bf16_moe_forward received invalid dimensions.");
  }
  if (hidden_ptr == 0 || topk_ids_ptr == 0 || topk_weights_ptr == 0 || output_ptr == 0 || act_tmp_ptr == 0 ||
      gate_ptrs_ptr == 0 || up_ptrs_ptr == 0 || down_ptrs_ptr == 0 || gpu_experts_mask_ptr == 0) {
    throw std::runtime_error("gh200_bf16_moe_forward received a null pointer.");
  }

  cudaStream_t stream = reinterpret_cast<cudaStream_t>(stream_ptr);
  constexpr int block = 128;
  const int64_t gate_work = static_cast<int64_t>(batch_size) * topk * intermediate_size;
  const int64_t down_work = static_cast<int64_t>(batch_size) * hidden_size;

  bf16_gate_up_kernel<<<launch_grid(gate_work), block, 0, stream>>>(
      reinterpret_cast<const __nv_bfloat16*>(hidden_ptr), reinterpret_cast<const int64_t*>(topk_ids_ptr),
      reinterpret_cast<float*>(act_tmp_ptr), reinterpret_cast<const uintptr_t*>(gate_ptrs_ptr),
      reinterpret_cast<const uintptr_t*>(up_ptrs_ptr), reinterpret_cast<const uint8_t*>(gpu_experts_mask_ptr),
      batch_size, expert_num, hidden_size, intermediate_size, topk);
  check_cuda(cudaGetLastError(), "bf16_gate_up_kernel launch");

  bf16_down_kernel<<<launch_grid(down_work), block, 0, stream>>>(
      reinterpret_cast<const int64_t*>(topk_ids_ptr), reinterpret_cast<const float*>(topk_weights_ptr),
      reinterpret_cast<__nv_bfloat16*>(output_ptr), reinterpret_cast<const float*>(act_tmp_ptr),
      reinterpret_cast<const uintptr_t*>(down_ptrs_ptr), reinterpret_cast<const uint8_t*>(gpu_experts_mask_ptr),
      batch_size, expert_num, hidden_size, intermediate_size, topk);
  check_cuda(cudaGetLastError(), "bf16_down_kernel launch");
}

void gh200_bf16_moe_forward_grouped(uintptr_t hidden_ptr, uintptr_t topk_ids_ptr, uintptr_t topk_weights_ptr,
                                    uintptr_t output_ptr, uintptr_t act_tmp_ptr, uintptr_t output_accum_ptr,
                                    uintptr_t route_counts_ptr, uintptr_t route_offsets_ptr,
                                    uintptr_t route_cursors_ptr, uintptr_t route_tokens_ptr,
                                    uintptr_t route_topks_ptr, uintptr_t route_experts_ptr,
                                    uintptr_t gate_ptrs_ptr, uintptr_t up_ptrs_ptr, uintptr_t down_ptrs_ptr,
                                    uintptr_t gpu_experts_mask_ptr, int batch_size, int expert_num, int hidden_size,
                                    int intermediate_size, int topk, uintptr_t stream_ptr) {
  if (batch_size <= 0 || expert_num <= 0 || hidden_size <= 0 || intermediate_size <= 0 || topk <= 0) {
    throw std::runtime_error("gh200_bf16_moe_forward_grouped received invalid dimensions.");
  }
  if (hidden_ptr == 0 || topk_ids_ptr == 0 || topk_weights_ptr == 0 || output_ptr == 0 || act_tmp_ptr == 0 ||
      output_accum_ptr == 0 || route_counts_ptr == 0 || route_offsets_ptr == 0 || route_cursors_ptr == 0 ||
      route_tokens_ptr == 0 || route_topks_ptr == 0 || route_experts_ptr == 0 || gate_ptrs_ptr == 0 ||
      up_ptrs_ptr == 0 || down_ptrs_ptr == 0 || gpu_experts_mask_ptr == 0) {
    throw std::runtime_error("gh200_bf16_moe_forward_grouped received a null pointer.");
  }

  cudaStream_t stream = reinterpret_cast<cudaStream_t>(stream_ptr);
  constexpr int block = 128;
  const int64_t route_work = static_cast<int64_t>(batch_size) * topk;
  const int64_t gate_work = route_work * intermediate_size;
  const int64_t down_work = route_work * hidden_size;
  const int64_t output_work = static_cast<int64_t>(batch_size) * hidden_size;

  auto* route_counts = reinterpret_cast<int*>(route_counts_ptr);
  auto* route_offsets = reinterpret_cast<int*>(route_offsets_ptr);
  auto* route_cursors = reinterpret_cast<int*>(route_cursors_ptr);
  auto* route_tokens = reinterpret_cast<int*>(route_tokens_ptr);
  auto* route_topks = reinterpret_cast<int*>(route_topks_ptr);
  auto* route_experts = reinterpret_cast<int*>(route_experts_ptr);

  check_cuda(cudaMemsetAsync(route_counts, 0, static_cast<size_t>(expert_num) * sizeof(int), stream),
             "cudaMemsetAsync route_counts");
  check_cuda(cudaMemsetAsync(route_cursors, 0, static_cast<size_t>(expert_num) * sizeof(int), stream),
             "cudaMemsetAsync route_cursors");
  check_cuda(cudaMemsetAsync(route_tokens, 0xff, static_cast<size_t>(route_work) * sizeof(int), stream),
             "cudaMemsetAsync route_tokens");
  check_cuda(cudaMemsetAsync(route_topks, 0xff, static_cast<size_t>(route_work) * sizeof(int), stream),
             "cudaMemsetAsync route_topks");
  check_cuda(cudaMemsetAsync(route_experts, 0xff, static_cast<size_t>(route_work) * sizeof(int), stream),
             "cudaMemsetAsync route_experts");
  check_cuda(cudaMemsetAsync(reinterpret_cast<void*>(output_accum_ptr), 0,
                             static_cast<size_t>(output_work) * sizeof(float), stream),
             "cudaMemsetAsync output_accum");

  count_cold_routes_kernel<<<launch_grid(route_work), block, 0, stream>>>(
      reinterpret_cast<const int64_t*>(topk_ids_ptr), route_counts,
      reinterpret_cast<const uintptr_t*>(gate_ptrs_ptr), reinterpret_cast<const uintptr_t*>(up_ptrs_ptr),
      reinterpret_cast<const uintptr_t*>(down_ptrs_ptr), reinterpret_cast<const uint8_t*>(gpu_experts_mask_ptr),
      batch_size, expert_num, topk);
  check_cuda(cudaGetLastError(), "count_cold_routes_kernel launch");

  prefix_route_offsets_kernel<<<1, 1, 0, stream>>>(route_counts, route_offsets, expert_num);
  check_cuda(cudaGetLastError(), "prefix_route_offsets_kernel launch");

  scatter_cold_routes_kernel<<<launch_grid(route_work), block, 0, stream>>>(
      reinterpret_cast<const int64_t*>(topk_ids_ptr), route_cursors, route_offsets, route_tokens, route_topks,
      route_experts, reinterpret_cast<const uintptr_t*>(gate_ptrs_ptr),
      reinterpret_cast<const uintptr_t*>(up_ptrs_ptr), reinterpret_cast<const uintptr_t*>(down_ptrs_ptr),
      reinterpret_cast<const uint8_t*>(gpu_experts_mask_ptr), batch_size, expert_num, topk);
  check_cuda(cudaGetLastError(), "scatter_cold_routes_kernel launch");

  grouped_bf16_gate_up_kernel<<<launch_grid(gate_work), block, 0, stream>>>(
      reinterpret_cast<const __nv_bfloat16*>(hidden_ptr), route_tokens, route_topks, route_experts,
      reinterpret_cast<float*>(act_tmp_ptr), reinterpret_cast<const uintptr_t*>(gate_ptrs_ptr),
      reinterpret_cast<const uintptr_t*>(up_ptrs_ptr), static_cast<int>(route_work), hidden_size, intermediate_size,
      topk);
  check_cuda(cudaGetLastError(), "grouped_bf16_gate_up_kernel launch");

  grouped_bf16_down_kernel<<<launch_grid(down_work), block, 0, stream>>>(
      reinterpret_cast<const float*>(topk_weights_ptr), reinterpret_cast<float*>(output_accum_ptr),
      reinterpret_cast<const float*>(act_tmp_ptr), route_tokens, route_topks, route_experts,
      reinterpret_cast<const uintptr_t*>(down_ptrs_ptr), static_cast<int>(route_work), hidden_size, intermediate_size,
      topk);
  check_cuda(cudaGetLastError(), "grouped_bf16_down_kernel launch");

  cast_float_to_bf16_kernel<<<launch_grid(output_work), block, 0, stream>>>(
      reinterpret_cast<const float*>(output_accum_ptr), reinterpret_cast<__nv_bfloat16*>(output_ptr), output_work);
  check_cuda(cudaGetLastError(), "cast_float_to_bf16_kernel launch");
}

#endif  // KTRANSFORMERS_USE_CUDA
