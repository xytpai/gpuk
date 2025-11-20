#include "ops.h"
#include "all_reduce_fusion_impl.h"

class CommWorkspace {
    static constexpr int MAX_RANKS = 16;

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

public:
    CommWorkspace(int64_t rank, int64_t world_size, int64_t size_in_bytes) {
        TORCH_CHECK(world_size < MAX_RANKS && rank < world_size);
        gpuSetDevice(rank);
        rank_ = rank;
        world_size_ = world_size;
        size_in_bytes_ = size_in_bytes;
        int data_size = size_in_bytes * 2 + NBLOCKS_PER_GPU * world_size * sizeof(int);
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
        flush_data<float>((void *)((char *)data_ + size_in_bytes * 2 + NBLOCKS_PER_GPU * world_size * sizeof(int)), one_shot_comm_size);
        dtype_ = ScalarType::Float;
        gpuDeviceSynchronize();
    }

    ~CommWorkspace() {
        gpuFree(counter_);
        gpuFree(twoshot_sync_clock_);
        gpuFree(data_);
        gpuFree(oneshot_sync_clock_);
        gpuFree(oneshot_clear_);
        gpuFree(oneshot_comm_size_);
    }

    Tensor get_handle() {
        gpuIpcMemHandle_t handle;
        TORCH_CHECK(gpuIpcGetMemHandle(&handle, data_) == gpuSuccess);
        auto options = torch::TensorOptions().dtype(torch::kUInt8).device(torch::kCPU);
        auto data_handle = torch::empty({static_cast<int64_t>(sizeof(gpuIpcMemHandle_t))}, options);
        std::memcpy(data_handle.data_ptr(), &handle, sizeof(gpuIpcMemHandle_t));
        return data_handle;
    }

    void open_handles(std::vector<Tensor> handles) {
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
                TORCH_CHECK(
                    gpuIpcOpenMemHandle((void **)&ipc_data_[i], ipc_handles[i], gpuIpcMemLazyEnablePeerAccess) == gpuSuccess);
            } else {
                ipc_data_[i] = data_;
            }
        }
        for (int i = 0; i < world_size_; ++i) {
            twoshot_comm_bufs_[i] = ipc_data_[i];
            twoshot_barrier_flags_[i] = (int *)((char *)ipc_data_[i] + 2 * size_in_bytes_);
            // oneshot
            oneshot_comm_bufs_[i] = (void *)((char *)ipc_data_[i] + 2 * size_in_bytes_ + NBLOCKS_PER_GPU * world_size_ * sizeof(int));
        }
    }

    Tensor get_workspace(const Tensor &ref) {
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
        auto options = torch::TensorOptions().dtype(torch::kUInt8).device(torch::kCPU);
        auto workspace_tensor = torch::empty({static_cast<int64_t>(workspace.size() * sizeof(void *))}, options);
        std::memcpy(workspace_tensor.data_ptr(), workspace.data(), workspace.size() * sizeof(void *));
        return workspace_tensor.to(ref.device());
    }

private:
    // meta
    int rank_;
    int world_size_;
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

fptr_t init_ar_fusion(int64_t rank, int64_t world_size, int64_t max_size_in_bytes) {
    switch (world_size) {
    case 8:
    case 4:
    case 2:
        break;
    default:
        throw std::invalid_argument("world size is not supported");
    }
    if (rank < 0 || rank >= world_size)
        throw std::invalid_argument("invalid rank passed in");
    return (fptr_t) new CommWorkspace(rank, world_size, max_size_in_bytes);
}

void destroy_ar_fusion(fptr_t fptr) {
    auto ptr = reinterpret_cast<CommWorkspace *>(fptr);
    delete ptr;
}

Tensor get_ar_fusion_handle(fptr_t fptr) {
    auto ptr = reinterpret_cast<CommWorkspace *>(fptr);
    return ptr->get_handle();
}

void open_ar_fusion_handles(fptr_t fptr, std::vector<Tensor> handles) {
    auto ptr = reinterpret_cast<CommWorkspace *>(fptr);
    ptr->open_handles(handles);
}

Tensor get_ar_fusion_workspace(fptr_t fptr, const Tensor &ref) {
    auto ptr = reinterpret_cast<CommWorkspace *>(fptr);
    return ptr->get_workspace(ref);
}

template <typename T>
struct KernelElementType { using type = T; };

template <>
struct KernelElementType<c10::Half> { using type = __half; };

template <>
struct KernelElementType<c10::BFloat16> {
    using type = __bfloat16;
};

#ifdef __CUDACC__
#else
#include <ATen/hip/impl/HIPGuardImplMasqueradingAsCUDA.h>
#include <ATen/hip/impl/HIPStreamMasqueradingAsCUDA.h>
#endif

void allreduce_rms(int64_t rank, int64_t nranks, Tensor &allreduce_in, Tensor &residual_in,
                   Tensor &rms_gamma, Tensor &residual_out, Tensor &norm_out, double eps, Tensor &workspace) {
    auto dev = allreduce_in.device();
    c10::DeviceGuard dev_guard(dev);
#ifdef __CUDACC__
#else
    auto stream = c10::hip::getCurrentHIPStreamMasqueradingAsCUDA().stream();
#endif
    int size = allreduce_in.numel();
    int hidden_dim = allreduce_in.size(-1);
    AT_DISPATCH_FLOATING_TYPES_AND2(
        kHalf,
        kBFloat16,
        allreduce_in.scalar_type(),
        "allreduce_rms", [&] {
            using k_scalar_t = KernelElementType<scalar_t>::type;
            allreduce_rms_fusion_impl<k_scalar_t>(
                (void **)workspace.data_ptr(),
                rank,
                nranks,
                size,
                hidden_dim,
                (void *)allreduce_in.data_ptr<scalar_t>(),
                (void *)residual_in.data_ptr<scalar_t>(),
                (void *)residual_out.data_ptr<scalar_t>(),
                (void *)norm_out.data_ptr(),
                (void *)rms_gamma.data_ptr<scalar_t>(),
                eps,
                stream);
        });
}
