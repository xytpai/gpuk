#include <torch/extension.h>
#include "ops.h"

TORCH_LIBRARY(gpuk, m) {
    m.def("init_ar_fusion(SymInt rank, SymInt world_size, SymInt max_size_in_bytes) -> int");
    m.impl("init_ar_fusion", &init_ar_fusion);
    m.def("destroy_ar_fusion(SymInt fptr) -> ()");
    m.impl("destroy_ar_fusion", &destroy_ar_fusion);
    m.def("get_ar_fusion_handle(SymInt fptr) -> Tensor");
    m.impl("get_ar_fusion_handle", &get_ar_fusion_handle);
    m.def("open_ar_fusion_handles(SymInt fptr, Tensor[] handles) -> ()");
    m.impl("open_ar_fusion_handles", &open_ar_fusion_handles);
    m.def("get_ar_fusion_workspace(SymInt fptr, Tensor ref) -> Tensor");
    m.def("allreduce_rms(SymInt rank, SymInt nranks, Tensor allreduce_in, "
          "Tensor residual_in, Tensor rms_gamma, Tensor residual_out, Tensor "
          "norm_out, Tensor scale_out, float eps, SymInt quant_type, Tensor workspace) -> ()");
}

TORCH_LIBRARY_IMPL(gpuk, CUDA, m) {
    m.impl("get_ar_fusion_workspace", &get_ar_fusion_workspace);
    m.impl("allreduce_rms", &allreduce_rms);
}
