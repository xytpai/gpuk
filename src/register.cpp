#include "all_reduce_fusion.h"
#include <torch/extension.h>

TORCH_LIBRARY(torch_dist_ext, m) {
    m.class_<CommWorkspace>("CommWorkspace")
        .def(torch::init<int64_t, int64_t, int64_t, int64_t>())
        .def("get_handle", &CommWorkspace::get_handle)
        .def("open_handles", &CommWorkspace::open_handles)
        .def("get_workspace", &CommWorkspace::get_workspace);
    m.def("allreduce_rms_fusion(SymInt rank, SymInt nranks, Tensor allreduce_in, "
          "Tensor residual_in, Tensor rms_gamma, Tensor residual_out, Tensor "
          "norm_out, float eps, Tensor workspace) -> ()");
}

TORCH_LIBRARY_IMPL(torch_dist_ext, CUDA, m) {
    m.impl("allreduce_rms_fusion", &allreduce_rms_fusion);
}
