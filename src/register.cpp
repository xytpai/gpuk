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

    m.def("ar_fusion_capture(SymInt fptr, Tensor input, Tensor[] handles) -> ()");
    m.impl("ar_fusion_capture", &ar_fusion_capture);

    m.def("get_tensor_ipc_handle(Tensor input) -> Tensor");
    m.impl("get_tensor_ipc_handle", &get_tensor_ipc_handle);

    m.def("allreduce_rms(SymInt fptr, Tensor allreduce_in, "
          "Tensor residual_in, Tensor rms_gamma, Tensor residual_out, Tensor "
          "norm_out, Tensor scale_out, float eps, SymInt quant_type) -> ()");
}

TORCH_LIBRARY_IMPL(gpuk, CUDA, m) {
    m.impl("allreduce_rms", &allreduce_rms);
}
