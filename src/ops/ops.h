#pragma once

#include <ATen/ATen.h>
#include <c10/core/DeviceGuard.h>
#include <torch/extension.h>

using namespace at;

using fptr_t = int64_t;
static_assert(sizeof(void *) == sizeof(fptr_t));

fptr_t init_ar_fusion(int64_t rank, int64_t world_size, int64_t max_size_in_bytes);
void destroy_ar_fusion(fptr_t fptr);
Tensor get_ar_fusion_handle(fptr_t fptr);
void open_ar_fusion_handles(fptr_t fptr, std::vector<Tensor> handles);
Tensor get_ar_fusion_workspace(fptr_t fptr, const Tensor &ref);

void allreduce_rms(int64_t rank, int64_t nranks, Tensor &allreduce_in, Tensor &residual_in,
                   Tensor &rms_gamma, Tensor &residual_out, Tensor &norm_out, double eps, Tensor &workspace);
