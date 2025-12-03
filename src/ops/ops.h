#pragma once

#include <ATen/ATen.h>
#include <c10/core/DeviceGuard.h>
#include <torch/extension.h>

using namespace at;

using fptr_t = int64_t;
static_assert(sizeof(void *) == sizeof(fptr_t));

fptr_t init_ar_fusion(int64_t device_id, int64_t rank, int64_t world_size, int64_t max_size_in_bytes, int64_t comm_ptrs_buf_len);
void destroy_ar_fusion(fptr_t fptr);
Tensor get_ar_fusion_barrier_handle(fptr_t fptr);
Tensor get_ar_fusion_data_handle(fptr_t fptr);
void open_ar_fusion_barrier_handles(fptr_t fptr, std::vector<Tensor> handles);
void open_ar_fusion_data_handles(fptr_t fptr, std::vector<Tensor> handles);

void ar_fusion_capture_clear(fptr_t fptr);
std::vector<Tensor> get_ar_fusion_captured_handles(fptr_t fptr);
Tensor get_ar_fusion_captured_offsets(fptr_t fptr);
void open_ar_fusion_captured_handles(fptr_t fptr, std::vector<Tensor> handles, std::vector<int64_t> offsets, int64_t ptr_idx);

void allreduce_rms(fptr_t fptr, Tensor &allreduce_in, Tensor &residual_in,
                   Tensor &rms_gamma, Tensor &residual_out, Tensor &norm_out, Tensor &scale_out,
                   double eps, int64_t quant_type);
void fused_rope_rms(Tensor &qkv, Tensor &qw, Tensor &kw, Tensor &cos_sin, Tensor &positions,
                    int64_t num_tokens, int64_t num_heads_q, int64_t num_heads_k, int64_t num_heads_v, int64_t head_size,
                    bool is_neox_style, double eps);
void fused_mrope_3d_rms(Tensor &qkv, Tensor &qw, Tensor &kw, Tensor &cos_sin, Tensor &positions,
                        int64_t num_tokens, int64_t num_heads_q, int64_t num_heads_k, int64_t num_heads_v, int64_t head_size,
                        bool is_neox_style, std::vector<int64_t> mrope_section_, bool is_interleaved, double eps);
void fused_mrope_3d_rms_set_kv(Tensor &qkv, Tensor &qw, Tensor &kw, Tensor &cos_sin, Tensor &positions,
                               int64_t num_tokens, int64_t num_heads_q, int64_t num_heads_k, int64_t num_heads_v, int64_t head_size,
                               bool is_neox_style, std::vector<int64_t> mrope_section_, bool is_interleaved, double eps,
                               Tensor &q, Tensor &k_cache, Tensor &v_cache, Tensor &kv_loc, double k_scale, double v_scale);
