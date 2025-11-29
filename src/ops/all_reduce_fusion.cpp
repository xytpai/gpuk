#include "ops.h"
#include "all_reduce_fusion_impl.h"

class CommWorkspace {
public:
    CommWorkspace(int64_t rank, int64_t world_size, int64_t size_in_bytes, int64_t max_thread_blocks = NBLOCKS_PER_GPU) {
        TORCH_CHECK(rank < world_size);
        gpuSetDevice(rank);
        rank_ = rank;
        world_size_ = world_size;
        size_in_bytes_ = size_in_bytes;
        max_thread_blocks_ = max_thread_blocks;
        gpuMalloc(&sync_clock_, max_thread_blocks_ * sizeof(int));
        gpuMalloc(&barrier_flags_, max_thread_blocks_ * world_size_ * sizeof(int));
        gpuMalloc(&data_, size_in_bytes_ * 2);
        gpuMemset(sync_clock_, 0, max_thread_blocks_ * sizeof(int));
        gpuMemset(barrier_flags_, 0, max_thread_blocks_ * world_size_ * sizeof(int));
    }

    ~CommWorkspace() {
        gpuFree(sync_clock_);
        gpuFree(barrier_flags_);
        gpuFree(data_);
    }

    Tensor get_handle(void *ptr) {
        gpuIpcMemHandle_t handle;
        TORCH_CHECK(gpuIpcGetMemHandle(&handle, ptr) == gpuSuccess);
        auto options = torch::TensorOptions().dtype(torch::kUInt8).device(torch::kCPU);
        auto data_handle = torch::empty({static_cast<int64_t>(sizeof(gpuIpcMemHandle_t))}, options);
        std::memcpy(data_handle.data_ptr(), &handle, sizeof(gpuIpcMemHandle_t));
        return data_handle;
    }

    Tensor get_barrier_handle() {
        return get_handle(barrier_flags_);
    }

    Tensor get_data_handle() {
        return get_handle(data_);
    }

    void open_handles(std::vector<Tensor> handles, void *ptr, std::vector<void *> &ipc_ptrs) {
        std::vector<gpuIpcMemHandle_t> ipc_handles;
        ipc_handles.reserve(world_size_);
        ipc_ptrs.resize(world_size_);
        for (auto &handle : handles) {
            // Ensure the tensor is on the same device as the current device.
            gpuIpcMemHandle_t ipc_handle;
            std::memcpy(&ipc_handle, handle.data_ptr(), sizeof(gpuIpcMemHandle_t));
            ipc_handles.push_back(ipc_handle);
        }
        for (int i = 0; i < world_size_; ++i) {
            if (i != rank_) {
                TORCH_CHECK(
                    gpuIpcOpenMemHandle((void **)&ipc_ptrs[i], ipc_handles[i], gpuIpcMemLazyEnablePeerAccess) == gpuSuccess);
            } else {
                ipc_ptrs[i] = ptr;
            }
        }
    }

    void open_barrier_handles(std::vector<Tensor> handles) {
        open_handles(handles, barrier_flags_, ipc_barrier_flags_);
    }

    void open_data_handles(std::vector<Tensor> handles) {
        open_handles(handles, data_, ipc_data_);
    }

    std::tuple<Tensor, fptr_t> get_workspace(const Tensor &ref) {
        std::vector<void *> workspace(world_size_ * 2 + 1);
        auto dtype = ref.scalar_type();
        for (int r = 0; r < world_size_; ++r) {
            workspace[r] = ipc_data_[r];
            workspace[world_size_ + r] = ipc_barrier_flags_[r];
        }
        workspace[world_size_ * 2 + 0] = sync_clock_;
        auto options = torch::TensorOptions().dtype(torch::kUInt8).device(torch::kCPU);
        auto workspace_tensor = torch::empty({static_cast<int64_t>(workspace.size() * sizeof(void *))}, options);
        std::memcpy(workspace_tensor.data_ptr(), workspace.data(), workspace.size() * sizeof(void *));
        return {workspace_tensor.to(ref.device()), (fptr_t)data_};
    }

private:
    int rank_;
    int world_size_;
    int size_in_bytes_;
    int max_thread_blocks_;
    void *sync_clock_;
    void *barrier_flags_;
    void *data_;
    std::vector<void *> ipc_barrier_flags_;
    std::vector<void *> ipc_data_;
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

Tensor get_ar_fusion_barrier_handle(fptr_t fptr) {
    auto ptr = reinterpret_cast<CommWorkspace *>(fptr);
    return ptr->get_barrier_handle();
}

Tensor get_ar_fusion_data_handle(fptr_t fptr) {
    auto ptr = reinterpret_cast<CommWorkspace *>(fptr);
    return ptr->get_data_handle();
}

void open_ar_fusion_barrier_handles(fptr_t fptr, std::vector<Tensor> handles) {
    auto ptr = reinterpret_cast<CommWorkspace *>(fptr);
    ptr->open_barrier_handles(handles);
}

void open_ar_fusion_data_handles(fptr_t fptr, std::vector<Tensor> handles) {
    auto ptr = reinterpret_cast<CommWorkspace *>(fptr);
    ptr->open_data_handles(handles);
}

std::tuple<Tensor, fptr_t> get_ar_fusion_workspace(fptr_t fptr, const Tensor &ref) {
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
                   Tensor &rms_gamma, Tensor &residual_out, Tensor &norm_out, Tensor &scale_out,
                   double eps, int64_t quant_type, Tensor &workspace, fptr_t comm_buf) {
    TORCH_CHECK(allreduce_in.is_contiguous() && residual_in.is_contiguous() && rms_gamma.is_contiguous());
    TORCH_CHECK(residual_out.is_contiguous() && norm_out.is_contiguous() && scale_out.is_contiguous());
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
                comm_buf,
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
                quant_type,
                (void *)scale_out.data_ptr<float>(),
                stream);
        });
}
