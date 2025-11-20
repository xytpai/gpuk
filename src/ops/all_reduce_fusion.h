#pragma once

#include <ATen/ATen.h>
#include <torch/extension.h>

using namespace at;

#define MAX_RANKS 16

class CommWorkspace : public torch::CustomClassHolder {
public:
    CommWorkspace(int64_t rank, int64_t world_size, int64_t nblocks,
                  int64_t size_in_bytes);
    ~CommWorkspace();
    Tensor get_handle();
    void open_handles(std::vector<Tensor> handles);
    Tensor get_workspace(const Tensor &ref);

private:
    // meta
    int rank_;
    int world_size_;
    int nblocks_;
    int size_in_bytes_;

    // data
    void *data_;
    void *ipc_data_[MAX_RANKS];

    int *counter_;
    // twoshot
    void *twoshot_comm_bufs_[MAX_RANKS];    // 2 * size * sizeof(T)
    int *twoshot_barrier_flags_[MAX_RANKS]; // nblocks * world_size
    int *twoshot_sync_clock_;
    // oneshot
    void *oneshot_comm_bufs_[MAX_RANKS];
    int *oneshot_sync_clock_;
    int *oneshot_comm_size_;
    int *oneshot_clear_;
    ScalarType dtype_;
};

void allreduce_rms_fusion(int64_t rank, int64_t nranks,
                          at::Tensor &allreduce_in, at::Tensor &residual_in,
                          at::Tensor &rms_gamma, at::Tensor &residual_out,
                          at::Tensor &norm_out, double eps, Tensor &workspace);
