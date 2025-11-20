#include "all_reduce_fusion.h"
#include "all_reduce_fusion_impl.h"

#include <ATen/hip/impl/HIPGuardImplMasqueradingAsCUDA.h>
#include <ATen/hip/impl/HIPStreamMasqueradingAsCUDA.h>

using namespace allreduce_fusion;

template <typename T>
struct KernelElementType { using type = T; };

template <>
struct KernelElementType<c10::Half> { using type = __half; };

template <>
struct KernelElementType<c10::BFloat16> {
    using type = __bfloat16;
};

void allreduce_rms_fusion(int64_t rank, int64_t nranks, Tensor &allreduce_in,
                          Tensor &residual_in, Tensor &rms_gamma,
                          Tensor &residual_out, Tensor &norm_out, double eps,
                          Tensor &workspace) {
    const at::hip::OptionalHIPGuardMasqueradingAsCUDA device_guard(
        device_of(allreduce_in));
    auto stream = c10::hip::getCurrentHIPStreamMasqueradingAsCUDA().stream();
    int size = allreduce_in.numel();
    int hidden_dim = allreduce_in.size(-1);
    AT_DISPATCH_FLOATING_TYPES_AND2(
        kHalf, kBFloat16, allreduce_in.scalar_type(), "allreduce_rms_fusion",
        [&] {
            using k_scalar_t = KernelElementType<scalar_t>::type;
            allreduce_fusion::allreduce_rms_fusion_impl<k_scalar_t>(
                (void **)workspace.data_ptr(), rank, nranks, size, hidden_dim,
                (void *)allreduce_in.data_ptr<scalar_t>(),
                (void *)residual_in.data_ptr<scalar_t>(),
                (void *)residual_out.data_ptr<scalar_t>(),
                (void *)norm_out.data_ptr<scalar_t>(),
                (void *)rms_gamma.data_ptr<scalar_t>(), eps, stream);
        });
}

template <typename T>
void flush_data(void *data, int one_shot_comm_size) {
    using element_t = typename neg_zero<T>::bits_type;
    std::vector<element_t> arr;
    arr.resize(one_shot_comm_size / sizeof(T));
    for (int i = 0; i < one_shot_comm_size / sizeof(element_t); ++i) {
        volatile element_t v = neg_zero<T>::neg_zero_bits;
        arr[i] = v;
    }
    gpuMemcpy(data, arr.data(), one_shot_comm_size, gpuMemcpyHostToDevice);
}

CommWorkspace::CommWorkspace(int64_t rank, int64_t world_size, int64_t nblocks,
                             int64_t size_in_bytes) {
    gpuSetDevice(rank);
    rank_ = rank;
    world_size_ = world_size;
    nblocks_ = nblocks;
    size_in_bytes_ = size_in_bytes;

    int data_size = size_in_bytes * 2 + nblocks * world_size * sizeof(int);
    int one_shot_comm_size = details::kOneShotMaxSize * world_size_ * 3;
    data_size += one_shot_comm_size;

    gpuMalloc(&data_, data_size);
    gpuMalloc(&counter_, sizeof(int));
    gpuMemset(counter_, 0, sizeof(int));

    gpuMalloc(&twoshot_sync_clock_, sizeof(int));
    gpuMemset(twoshot_sync_clock_, 0, sizeof(int));

    // oneshot

    gpuMalloc(&oneshot_sync_clock_, sizeof(int));
    gpuMemset(oneshot_sync_clock_, 0, sizeof(int));

    int size = details::kOneShotMaxSize * world_size;
    gpuMalloc(&oneshot_comm_size_, sizeof(int));
    gpuMemcpy(oneshot_comm_size_, &size, sizeof(int), gpuMemcpyHostToDevice);

    gpuMalloc(&oneshot_clear_, sizeof(int));
    gpuMemset(oneshot_clear_, 0, sizeof(int));

    flush_data<float>((void *)((char *)data_ + size_in_bytes * 2 + nblocks * world_size * sizeof(int)),
                      one_shot_comm_size);
    dtype_ = ScalarType::Float;
    gpuDeviceSynchronize();
}

CommWorkspace::~CommWorkspace() {
    gpuFree(counter_);
    gpuFree(twoshot_sync_clock_);
    gpuFree(data_);
    gpuFree(oneshot_sync_clock_);
    gpuFree(oneshot_clear_);
    gpuFree(oneshot_comm_size_);
}

Tensor CommWorkspace::get_handle() {
    gpuIpcMemHandle_t handle;
    TORCH_CHECK(gpuIpcGetMemHandle(&handle, data_) == gpuSuccess);
    auto options =
        torch::TensorOptions().dtype(torch::kUInt8).device(torch::kCPU);
    auto data_handle =
        torch::empty({static_cast<int64_t>(sizeof(gpuIpcMemHandle_t))}, options);
    std::memcpy(data_handle.data_ptr(), &handle, sizeof(gpuIpcMemHandle_t));
    return data_handle;
}

void CommWorkspace::open_handles(std::vector<Tensor> handles) {
    std::vector<gpuIpcMemHandle_t> ipc_handles;
    ipc_handles.reserve(world_size_);
    for (auto &handle : handles) {
        // Ensure the tensor is on the same device as the current device.
        gpuIpcMemHandle_t ipc_handle;
        std::memcpy(&ipc_handle, handle.data_ptr(), sizeof(gpuIpcMemHandle_t));
        ipc_handles.push_back(ipc_handle);
    }

    for (int i = 0; i < world_size_; ++i) {
        if (i != rank_) {
            TORCH_CHECK(gpuIpcOpenMemHandle((void **)&ipc_data_[i], ipc_handles[i],
                                            gpuIpcMemLazyEnablePeerAccess)
                        == gpuSuccess);
        } else {
            ipc_data_[i] = data_;
        }
    }

    for (int i = 0; i < world_size_; ++i) {
        twoshot_comm_bufs_[i] = ipc_data_[i];
        twoshot_barrier_flags_[i] =
            (int *)((char *)ipc_data_[i] + 2 * size_in_bytes_);
        // oneshot
        oneshot_comm_bufs_[i] = (void *)((char *)ipc_data_[i] + 2 * size_in_bytes_ + nblocks_ * world_size_ * sizeof(int));
    }
}

Tensor CommWorkspace::get_workspace(const Tensor &ref) {
    std::vector<void *> workspace(world_size_ * 3 + 5);
    auto dtype = ref.scalar_type();
    int one_shot_comm_size = details::kOneShotMaxSize * world_size_ * 3;
    if (dtype != dtype_) {
        if (dtype == ScalarType::Float) {
            flush_data<float>(oneshot_comm_bufs_[rank_], one_shot_comm_size);
        } else if (dtype == ScalarType::Half) {
            flush_data<__half>(oneshot_comm_bufs_[rank_], one_shot_comm_size);
        } else if (dtype == ScalarType::BFloat16) {
            flush_data<__bfloat16>(oneshot_comm_bufs_[rank_], one_shot_comm_size);
        } else {
            TORCH_CHECK("datatype not support!");
        }
        dtype_ = dtype;
    }
    for (int peer = 0; peer < world_size_; ++peer) {
        workspace[peer] = (void *)twoshot_comm_bufs_[peer];
        workspace[world_size_ + peer] = (void *)twoshot_barrier_flags_[peer];
        workspace[2 * world_size_ + peer] = (void *)oneshot_comm_bufs_[peer];
    }
    workspace[world_size_ * 3 + 0] = (void *)counter_;
    workspace[world_size_ * 3 + 1] = (void *)twoshot_sync_clock_;
    // oneshot
    workspace[world_size_ * 3 + 2] = (void *)oneshot_sync_clock_;
    workspace[world_size_ * 3 + 3] = (void *)oneshot_comm_size_;
    workspace[world_size_ * 3 + 4] = (void *)oneshot_clear_;
    auto options =
        torch::TensorOptions().dtype(torch::kUInt8).device(torch::kCPU);
    auto workspace_tensor = torch::empty(
        {static_cast<int64_t>(workspace.size() * sizeof(void *))}, options);
    std::memcpy(workspace_tensor.data_ptr(), workspace.data(),
                workspace.size() * sizeof(void *));
    return workspace_tensor.to(ref.device());
}
