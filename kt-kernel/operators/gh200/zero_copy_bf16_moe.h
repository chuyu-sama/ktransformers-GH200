#ifndef CPUINFER_OPERATOR_GH200_ZERO_COPY_BF16_MOE_H
#define CPUINFER_OPERATOR_GH200_ZERO_COPY_BF16_MOE_H

#include <cstddef>
#include <cstdint>

#ifdef KTRANSFORMERS_USE_CUDA

uintptr_t gh200_register_mapped_host(uintptr_t host_ptr, size_t bytes, bool read_only);
void gh200_unregister_mapped_host(uintptr_t host_ptr);

void gh200_bf16_moe_forward(uintptr_t hidden_ptr, uintptr_t topk_ids_ptr, uintptr_t topk_weights_ptr,
                            uintptr_t output_ptr, uintptr_t act_tmp_ptr, uintptr_t gate_ptrs_ptr,
                            uintptr_t up_ptrs_ptr, uintptr_t down_ptrs_ptr, uintptr_t gpu_experts_mask_ptr,
                            int batch_size, int expert_num, int hidden_size, int intermediate_size, int topk,
                            uintptr_t stream_ptr);

void gh200_bf16_moe_forward_grouped(uintptr_t hidden_ptr, uintptr_t topk_ids_ptr, uintptr_t topk_weights_ptr,
                                    uintptr_t output_ptr, uintptr_t act_tmp_ptr, uintptr_t output_accum_ptr,
                                    uintptr_t route_counts_ptr, uintptr_t route_offsets_ptr,
                                    uintptr_t route_cursors_ptr, uintptr_t route_tokens_ptr,
                                    uintptr_t route_topks_ptr, uintptr_t route_experts_ptr,
                                    uintptr_t gate_ptrs_ptr, uintptr_t up_ptrs_ptr, uintptr_t down_ptrs_ptr,
                                    uintptr_t gpu_experts_mask_ptr, int batch_size, int expert_num, int hidden_size,
                                    int intermediate_size, int topk, uintptr_t stream_ptr);

#endif  // KTRANSFORMERS_USE_CUDA

#endif  // CPUINFER_OPERATOR_GH200_ZERO_COPY_BF16_MOE_H
