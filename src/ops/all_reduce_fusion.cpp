#include "ops.h"
#include "all_reduce_fusion_impl.h"

namespace ipc_details {

Tensor get_handle(void *ptr) {
    gpuIpcMemHandle_t handle;
    TORCH_CHECK(gpuIpcGetMemHandle(&handle, ptr) == gpuSuccess);
    auto options = torch::TensorOptions().dtype(torch::kUInt8).device(torch::kCPU);
    auto data_handle = torch::empty({static_cast<int64_t>(sizeof(gpuIpcMemHandle_t))}, options);
    std::memcpy(data_handle.data_ptr(), &handle, sizeof(gpuIpcMemHandle_t));
    return data_handle;
}

void open_handles(int rank, std::vector<Tensor> &handles, void *ptr, std::vector<void *> &ipc_ptrs) {
    std::vector<gpuIpcMemHandle_t> ipc_handles;
    int world_size = handles.size();
    ipc_handles.reserve(world_size);
    ipc_ptrs.resize(world_size);
    for (auto &handle : handles) {
        // Ensure the tensor is on the same device as the current device.
        gpuIpcMemHandle_t ipc_handle;
        std::memcpy(&ipc_handle, handle.data_ptr(), sizeof(gpuIpcMemHandle_t));
        ipc_handles.push_back(ipc_handle);
    }
    for (int i = 0; i < world_size; ++i) {
        if (i != rank) {
            TORCH_CHECK(
                gpuIpcOpenMemHandle((void **)&ipc_ptrs[i], ipc_handles[i], gpuIpcMemLazyEnablePeerAccess) == gpuSuccess);
        } else {
            ipc_ptrs[i] = ptr;
        }
    }
}

void create_base_ptr(void **base_ptr, void *ptr) {
    if (gpuPointerGetAttribute(base_ptr, GPU_POINTER_ATTRIBUTE_RANGE_START_ADDR, (gpuDeviceptr_t)ptr) != gpuSuccess) {
        throw std::runtime_error("failed to get pointer attr");
    }
}

} // namespace ipc_details

class CommWorkspace {
public:
    CommWorkspace(
        int64_t device_id,         // assign device
        int64_t rank,              // rank in group
        int64_t world_size,        // group size
        int64_t size_in_bytes,     // private data size
        int64_t comm_ptrs_buf_len, // cached ptrs size
        int64_t max_thread_blocks = NBLOCKS_PER_GPU,
        bool round_robin = true) {
        TORCH_CHECK(rank < world_size);
        gpuSetDevice(device_id);
        device_id_ = device_id;
        rank_ = rank;
        world_size_ = world_size;
        size_in_bytes_ = size_in_bytes;
        comm_ptrs_buf_len_ = comm_ptrs_buf_len;
        max_thread_blocks_ = max_thread_blocks;
        gpuMalloc(&sync_clock_, max_thread_blocks_ * sizeof(int));
        gpuMalloc(&barrier_flags_, max_thread_blocks_ * world_size_ * sizeof(int));
        gpuMalloc(&data_, size_in_bytes_ * 2);
        gpuMalloc(&comm_ptrs_, comm_ptrs_buf_len_ * sizeof(CommPtrs));
        gpuMemset(sync_clock_, 0, max_thread_blocks_ * sizeof(int));
        gpuMemset(barrier_flags_, 0, max_thread_blocks_ * world_size_ * sizeof(int));
        used_comm_ptrs_ = 0;
        round_robin_ = round_robin;
    }

    ~CommWorkspace() {
        gpuFree(sync_clock_);
        gpuFree(barrier_flags_);
        gpuFree(data_);
        gpuFree(comm_ptrs_);
    }

    Tensor get_barrier_handle() {
        return ipc_details::get_handle(barrier_flags_);
    }

    Tensor get_data_handle() {
        return ipc_details::get_handle(data_);
    }

    void open_barrier_handles(std::vector<Tensor> handles) {
        ipc_details::open_handles(rank_, handles, barrier_flags_, ipc_barrier_flags_);
    }

    void open_data_handles(std::vector<Tensor> handles) {
        ipc_details::open_handles(rank_, handles, data_, ipc_data_);
        CommPtrs *cptrs = new CommPtrs[comm_ptrs_buf_len_];
        for (int i = 0; i < comm_ptrs_buf_len_; ++i) {
            for (int j = 0; j < world_size_; ++j) {
                int r = round_robin_ ? ((rank_ + j) % world_size_) : j;
                cptrs[i].data_ptrs[j] = ipc_data_[r];
            }
        }
        gpuMemcpy(comm_ptrs_, cptrs, comm_ptrs_buf_len_ * sizeof(CommPtrs), gpuMemcpyHostToDevice);
        used_comm_ptrs_ = 2;
        delete[] cptrs;
    }

    std::tuple<CommMeta, CommPtrs *> get_comm_data(const Tensor &input, gpuStream_t stream, bool pre_copy = true) {
        int64_t size = input.numel() * input.element_size();
        void *ptr = (void *)input.data_ptr();

        CommMeta meta;
        for (int r = 0; r < world_size_; ++r) {
            meta.barrier_flag_ptrs[r] = ipc_barrier_flags_[r];
        }
        meta.sync_clock = sync_clock_;
        meta.rank = rank_;
        meta.nranks = world_size_;

        CommPtrs *cptrs;
        auto it = ptr_to_comm_ptrs_.find(ptr);
        if (it != ptr_to_comm_ptrs_.end()) {
            cptrs = it->second;
        } else {
            gpuStreamCaptureStatus status;
            gpuStreamIsCapturing(stream, &status);
            int remaining = comm_ptrs_buf_len_ - used_comm_ptrs_ - unregistered_ptrs_.size();
            if (status == gpuStreamCaptureStatusActive && size < 1024 * 4096 * 16 && remaining > 0) {
                unregistered_ptrs_.push_back(ptr);
                cptrs = comm_ptrs_ + used_comm_ptrs_ + unregistered_ptrs_.size() - 1;
            } else {
                cptrs = comm_ptrs_ + 0;
                if (pre_copy) {
                    gpuMemcpyAsync(data_, ptr, size, gpuMemcpyDeviceToDevice, stream);
                }
            }
        }

        return {meta, cptrs};
    }

    std::tuple<CommMeta, CommPtrs *> get_comm_data(const Tensor &input, std::vector<Tensor> &handles, std::vector<int64_t> &offsets, gpuStream_t stream) {
        int64_t size = input.numel() * input.element_size();
        void *ptr = (void *)input.data_ptr();
        void *base_ptr;
        ipc_details::create_base_ptr(&base_ptr, ptr);

        CommMeta meta;
        for (int r = 0; r < world_size_; ++r) {
            meta.barrier_flag_ptrs[r] = ipc_barrier_flags_[r];
        }
        meta.sync_clock = sync_clock_;
        meta.rank = rank_;
        meta.nranks = world_size_;

        std::vector<void *> ipc_data;
        ipc_details::open_handles(rank_, handles, base_ptr, ipc_data);
        CommPtrs cptrs;
        for (int i = 0; i < offsets.size(); ++i) {
            ipc_data[i] = (void *)((char *)ipc_data[i] + offsets[i]);
        }
        for (int i = 0; i < offsets.size(); ++i) {
            int r = round_robin_ ? ((rank_ + i) % world_size_) : i;
            cptrs.data_ptrs[i] = ipc_data[r];
        }
        CommPtrs *_cptrs = comm_ptrs_ + 1;
        gpuMemcpyAsync(_cptrs, &cptrs, sizeof(CommPtrs), gpuMemcpyHostToDevice, stream);
        return {meta, _cptrs};
    }

    void capture_clear() {
        unregistered_ptrs_.clear();
        unregistered_base_ptrs_.clear();
    }

    Tensor get_tensor_handle(Tensor &input) {
        void *ptr = (void *)input.data_ptr();
        void *base_ptr;
        ipc_details::create_base_ptr(&base_ptr, ptr);
        return ipc_details::get_handle(base_ptr);
    }

    int64_t get_tensor_offset(Tensor &input) {
        void *ptr = (void *)input.data_ptr();
        void *base_ptr;
        ipc_details::create_base_ptr(&base_ptr, ptr);
        int64_t offset = ((char *)ptr) - ((char *)base_ptr);
        return offset;
    }

    std::vector<Tensor> get_captured_handles() {
        int num_datas = unregistered_ptrs_.size();
        std::vector<Tensor> ipc_handles;
        ipc_handles.reserve(num_datas);
        for (int i = 0; i < num_datas; ++i) {
            void *ptr = unregistered_ptrs_[i];
            void *base_ptr;
            ipc_details::create_base_ptr(&base_ptr, ptr);
            ipc_handles.push_back(ipc_details::get_handle(base_ptr));
            unregistered_base_ptrs_.push_back(base_ptr);
        }
        return ipc_handles;
    }

    Tensor get_captured_offsets() {
        int num_datas = unregistered_ptrs_.size();
        std::vector<int64_t> offsets;
        offsets.reserve(num_datas);
        for (int i = 0; i < num_datas; ++i) {
            void *ptr = unregistered_ptrs_[i];
            void *base_ptr = unregistered_base_ptrs_[i];
            int64_t offset = ((char *)ptr) - ((char *)base_ptr);
            offsets.push_back(offset);
        }
        auto options = torch::TensorOptions().dtype(torch::kInt64).device(torch::kCPU);
        auto t = torch::tensor(offsets, options);
        return t;
    }

    void open_captured_handles(std::vector<Tensor> &handles, std::vector<int64_t> &offsets, int64_t ptr_idx) {
        void *ptr = unregistered_ptrs_[ptr_idx];
        void *base_ptr = unregistered_base_ptrs_[ptr_idx];
        std::vector<void *> ipc_data;
        ipc_details::open_handles(rank_, handles, base_ptr, ipc_data);
        CommPtrs cptrs;
        for (int i = 0; i < offsets.size(); ++i) {
            ipc_data[i] = (void *)((char *)ipc_data[i] + offsets[i]);
        }
        for (int i = 0; i < offsets.size(); ++i) {
            int r = round_robin_ ? ((rank_ + i) % world_size_) : i;
            cptrs.data_ptrs[i] = ipc_data[r];
        }
        gpuMemcpy(comm_ptrs_ + used_comm_ptrs_, &cptrs, sizeof(CommPtrs), gpuMemcpyHostToDevice);
        ptr_to_comm_ptrs_[ptr] = comm_ptrs_ + used_comm_ptrs_;
        used_comm_ptrs_++;
    }

private:
    int device_id_;
    int rank_;
    int world_size_;
    int size_in_bytes_;
    int comm_ptrs_buf_len_;
    int max_thread_blocks_;
    bool round_robin_;
    void *sync_clock_;
    void *barrier_flags_;
    void *data_;
    std::vector<void *> ipc_barrier_flags_;
    std::vector<void *> ipc_data_;
    // graph
    std::vector<void *> unregistered_ptrs_;
    std::vector<void *> unregistered_base_ptrs_;
    CommPtrs *comm_ptrs_;
    int used_comm_ptrs_;
    std::unordered_map<void *, CommPtrs *> ptr_to_comm_ptrs_;
};

fptr_t init_ar_fusion(int64_t device_id, int64_t rank, int64_t world_size, int64_t max_size_in_bytes, int64_t comm_ptrs_buf_len) {
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
    return (fptr_t) new CommWorkspace(device_id, rank, world_size, max_size_in_bytes, comm_ptrs_buf_len);
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

void ar_fusion_capture_clear(fptr_t fptr) {
    auto ptr = reinterpret_cast<CommWorkspace *>(fptr);
    ptr->capture_clear();
}

Tensor get_ar_fusion_tensor_handle(fptr_t fptr, Tensor &input) {
    auto ptr = reinterpret_cast<CommWorkspace *>(fptr);
    return ptr->get_tensor_handle(input);
}

int64_t get_ar_fusion_tensor_offset(fptr_t fptr, Tensor &input) {
    auto ptr = reinterpret_cast<CommWorkspace *>(fptr);
    return ptr->get_tensor_offset(input);
}

std::vector<Tensor> get_ar_fusion_captured_handles(fptr_t fptr) {
    auto ptr = reinterpret_cast<CommWorkspace *>(fptr);
    return ptr->get_captured_handles();
}

Tensor get_ar_fusion_captured_offsets(fptr_t fptr) {
    auto ptr = reinterpret_cast<CommWorkspace *>(fptr);
    return ptr->get_captured_offsets();
}

void open_ar_fusion_captured_handles(fptr_t fptr, std::vector<Tensor> handles, std::vector<int64_t> offsets, int64_t ptr_idx) {
    auto ptr = reinterpret_cast<CommWorkspace *>(fptr);
    ptr->open_captured_handles(handles, offsets, ptr_idx);
}

template <typename T>
struct KernelElementType {
    using type = T;
};

template <>
struct KernelElementType<c10::Half> {
    using type = __half;
};

template <>
struct KernelElementType<c10::BFloat16> {
    using type = __bfloat16;
};

#ifdef __CUDACC__
#else
#include <ATen/hip/impl/HIPGuardImplMasqueradingAsCUDA.h>
#include <ATen/hip/impl/HIPStreamMasqueradingAsCUDA.h>
#endif

void allreduce_inplace(fptr_t fptr, Tensor &input) {
    TORCH_CHECK(input.is_contiguous());
    auto dev = input.device();
    c10::DeviceGuard dev_guard(dev);
#ifdef __CUDACC__
#else
    auto stream = c10::hip::getCurrentHIPStreamMasqueradingAsCUDA().stream();
#endif
    int size = input.numel();
    auto ptr = reinterpret_cast<CommWorkspace *>(fptr);
    auto comm_data = ptr->get_comm_data(input, stream, false);
    AT_DISPATCH_FLOATING_TYPES_AND2(
        kHalf,
        kBFloat16,
        input.scalar_type(),
        "allreduce_inplace", [&] {
            using k_scalar_t = KernelElementType<scalar_t>::type;
            allreduce_inplace_impl<k_scalar_t>(
                std::get<0>(comm_data),
                std::get<1>(comm_data),
                (void *)input.data_ptr<scalar_t>(),
                size,
                stream);
        });
}

void allreduce_rms(fptr_t fptr, Tensor &allreduce_in, Tensor &residual_in,
                   Tensor &rms_gamma, Tensor &residual_out, Tensor &norm_out, Tensor &scale_out,
                   double eps, int64_t quant_type) {
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
    auto ptr = reinterpret_cast<CommWorkspace *>(fptr);
    auto comm_data = ptr->get_comm_data(allreduce_in, stream);
    AT_DISPATCH_FLOATING_TYPES_AND2(
        kHalf,
        kBFloat16,
        allreduce_in.scalar_type(),
        "allreduce_rms", [&] {
            using k_scalar_t = KernelElementType<scalar_t>::type;
            allreduce_rms_fusion_impl<k_scalar_t>(
                std::get<0>(comm_data),
                std::get<1>(comm_data),
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
