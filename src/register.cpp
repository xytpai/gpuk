#include <torch/extension.h>
#include "ops.h"

TORCH_LIBRARY(gpuk, m) {
    m.def("init_ar_fusion(SymInt rank, SymInt world_size, SymInt max_size_in_bytes) -> int");
    m.impl("init_ar_fusion", &init_ar_fusion);
    m.def("destroy_ar_fusion(SymInt fptr) -> ()");
    m.impl("destroy_ar_fusion", &destroy_ar_fusion);

    m.def("get_ar_fusion_barrier_handle(SymInt fptr) -> Tensor");
    m.impl("get_ar_fusion_barrier_handle", &get_ar_fusion_barrier_handle);
    m.def("get_ar_fusion_data_handle(SymInt fptr) -> Tensor");
    m.impl("get_ar_fusion_data_handle", &get_ar_fusion_data_handle);

    m.def("open_ar_fusion_barrier_handles(SymInt fptr, Tensor[] handles) -> ()");
    m.impl("open_ar_fusion_barrier_handles", &open_ar_fusion_barrier_handles);
    m.def("open_ar_fusion_data_handles(SymInt fptr, Tensor[] handles) -> ()");
    m.impl("open_ar_fusion_data_handles", &open_ar_fusion_data_handles);

    m.def("ar_fusion_capture(SymInt fptr, Tensor input) -> ()");
    m.impl("ar_fusion_capture", &ar_fusion_capture);
    m.def("ar_fusion_capture_clear(SymInt fptr) -> ()");
    m.impl("ar_fusion_capture_clear", &ar_fusion_capture_clear);
    m.def("get_ar_fusion_captured_handles(SymInt fptr) -> (Tensor[], int[])");
    m.impl("get_ar_fusion_captured_handles", &get_ar_fusion_captured_handles);
    m.def("open_ar_fusion_captured_handles(SymInt fptr, Tensor[] handles, int[] offsets, SymInt ptr_idx) -> ()");
    m.impl("open_ar_fusion_captured_handles", &open_ar_fusion_captured_handles);

    m.def("allreduce_rms(SymInt fptr, Tensor allreduce_in, "
          "Tensor residual_in, Tensor rms_gamma, Tensor residual_out, Tensor "
          "norm_out, Tensor scale_out, float eps, SymInt quant_type) -> ()");
    m.def("fused_rope_rms(Tensor qkv, Tensor qw, Tensor kw, Tensor cos_sin, Tensor positions, "
          "SymInt num_tokens, SymInt num_heads_q, SymInt num_heads_k, SymInt num_heads_v, SymInt head_size, "
          "bool is_neox_style, float eps) -> ()");
    m.def("fused_mrope_3d_rms(Tensor qkv, Tensor qw, Tensor kw, Tensor cos_sin, Tensor positions, "
          "SymInt num_tokens, SymInt num_heads_q, SymInt num_heads_k, SymInt num_heads_v, SymInt head_size, "
          "bool is_neox_style, int[] mrope_section_, bool is_interleaved, float eps) -> ()");
    m.def("fused_mrope_3d_rms_set_kv(Tensor qkv, Tensor qw, Tensor kw, Tensor cos_sin, Tensor positions, "
          "SymInt num_tokens, SymInt num_heads_q, SymInt num_heads_k, SymInt num_heads_v, SymInt head_size, "
          "bool is_neox_style, int[] mrope_section_, bool is_interleaved, float eps, "
          "Tensor q, Tensor k_cache, Tensor v_cache, Tensor kv_loc, float k_scale, float v_scale) -> ()");
}

TORCH_LIBRARY_IMPL(gpuk, CUDA, m) {
    m.impl("allreduce_rms", &allreduce_rms);
    m.impl("fused_rope_rms", &fused_rope_rms);
    m.impl("fused_mrope_3d_rms", &fused_mrope_3d_rms);
    m.impl("fused_mrope_3d_rms_set_kv", &fused_mrope_3d_rms_set_kv);
}
