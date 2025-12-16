#pragma once

#include "device_common.h"
#include "kernel_utils.h"

namespace comm_utils {

namespace details {

static constexpr int kBytesPerAccess = 16;

template <bool RELAXED = true>
__device__ __forceinline__ void st_flag(int *addr, int flag) {
#ifdef __CUDACC__
    asm volatile("st.global.release.sys.b32 [%1], %0;" ::"r"(flag), "l"(addr));
#else
    __scoped_atomic_store_n(addr, flag,
                            RELAXED ? __ATOMIC_RELAXED : __ATOMIC_RELEASE,
                            __MEMORY_SCOPE_SYSTEM);
#endif
}

template <bool RELAXED = true>
__device__ __forceinline__ int ld_flag(int *addr) {
    int flag;
#ifdef __CUDACC__
    asm volatile("ld.global.acquire.sys.b32 %0, [%1];"
                 : "=r"(flag)
                 : "l"(addr));
#else
    flag = __scoped_atomic_load_n(addr,
                                  RELAXED ? __ATOMIC_RELAXED : __ATOMIC_ACQUIRE,
                                  __MEMORY_SCOPE_SYSTEM);
#endif
    return flag;
}

} // namespace details

inline int enable_p2p() {
    int ngpus = 0;
    gpuGetDeviceCount(&ngpus);
    for (int local = 0; local < ngpus; ++local) {
        gpuSetDevice(local);
        for (int peer = 0; peer < ngpus; ++peer) {
            if (local == peer)
                continue;
            int can = 0;
            gpuDeviceCanAccessPeer(&can, local, peer);
            assert(can);
            gpuDeviceEnablePeerAccess(peer, 0);
        }
    }
    return ngpus;
}

static constexpr int MAX_RANKS = 8;
static constexpr int NBLOCKS_PER_GPU = 80;

template <int NRanks>
struct CommDeviceMeta {
    void *barrier_flag_ptrs[NRanks];
    void *sync_clock;
    int rank;
    int nranks;
};

struct CommMeta {
    void *barrier_flag_ptrs[MAX_RANKS];
    void *sync_clock;
    int rank;
    int nranks;
};

struct CommPtrs {
    void *data_ptrs[MAX_RANKS];
};

template <int NRanks>
struct SyncComm {
    __device__ __forceinline__ SyncComm(CommDeviceMeta<NRanks> &meta) {
        flag_ptr = ((int *)meta.sync_clock) + blockIdx.x;
        int rank = meta.rank;
        if (threadIdx.x < NRanks) {
            int target_rank = threadIdx.x;
            target_flag = reinterpret_cast<int *>(meta.barrier_flag_ptrs[target_rank]) + blockIdx.x * NRanks + rank;
            current_flag = reinterpret_cast<int *>(meta.barrier_flag_ptrs[rank]) + blockIdx.x * NRanks + target_rank;
        }
        flag = *flag_ptr;
    }

    template <bool RELAXED = true, bool FINAL = true>
    __device__ __forceinline__ void sync() {
        __syncthreads();
        flag += 1;
        if (threadIdx.x < NRanks) {
            details::st_flag<RELAXED>(target_flag, flag);
            while (details::ld_flag<RELAXED>(current_flag) < flag) {
            }
        }
        __syncthreads();
        if constexpr (FINAL) {
            if (threadIdx.x == 0) {
                *flag_ptr = flag;
            }
        }
    }

    int *flag_ptr;
    int *target_flag;
    int *current_flag;
    int flag;
};

class Communicator {
public:
    Communicator() :
        initialized_(false) {
    }

    void local_init(
        int64_t device_id,     // assign device
        int64_t rank,          // rank in group
        int64_t world_size,    // group size
        int64_t size_in_bytes, // private data size
        int64_t max_thread_blocks = NBLOCKS_PER_GPU) {
        assert(rank < world_size);
        device_id_ = device_id;
        rank_ = rank;
        world_size_ = world_size;
        size_in_bytes_ = size_in_bytes;
        max_thread_blocks_ = max_thread_blocks;
        gpuSetDevice(device_id_);
        gpuMalloc(&sync_clock_, max_thread_blocks_ * sizeof(int));
        gpuMalloc(&barrier_flags_, max_thread_blocks_ * world_size_ * sizeof(int));
        gpuMalloc(&data_, size_in_bytes_ * 2);
        gpuMemset(sync_clock_, 0, max_thread_blocks_ * sizeof(int));
        gpuMemset(barrier_flags_, 0, max_thread_blocks_ * world_size_ * sizeof(int));
        gpuDeviceSynchronize();
        initialized_ = true;
    }

    friend void init_communicators(std::vector<Communicator> &communicators);

    int get_dst_id(int i) const {
        return (rank_ + i) % world_size_;
    }

    std::tuple<CommMeta, CommPtrs> get_comm_data() const {
        assert(initialized_);
        CommMeta meta;
        for (int r = 0; r < world_size_; ++r) {
            meta.barrier_flag_ptrs[r] = comm_barrier_flags_[r];
        }
        meta.sync_clock = sync_clock_;
        meta.rank = rank_;
        meta.nranks = world_size_;
        CommPtrs cptrs;
        for (int r = 0; r < world_size_; ++r) {
            cptrs.data_ptrs[r] = comm_data_[r];
        }
        return {meta, cptrs};
    }

    ~Communicator() {
        gpuFree(sync_clock_);
        gpuFree(barrier_flags_);
        gpuFree(data_);
    }

private:
    int device_id_;
    int rank_;
    int world_size_;
    int size_in_bytes_;
    int max_thread_blocks_;
    void *sync_clock_;
    void *barrier_flags_;
    void *data_;
    std::vector<void *> comm_barrier_flags_;
    std::vector<void *> comm_data_;
    bool initialized_;
};

void init_communicators(std::vector<Communicator> &communicators) {
    int world_size = communicators[0].world_size_;
    std::vector<void *> comm_barrier_flags, comm_data;
    comm_barrier_flags.resize(world_size);
    comm_data.resize(world_size);
    for (int i = 0; i < communicators.size(); ++i) {
        assert(communicators[i].initialized_);
        assert(communicators[i].world_size_ == world_size);
        communicators[i].comm_barrier_flags_.resize(world_size);
        communicators[i].comm_data_.resize(world_size);
        comm_barrier_flags[i] = communicators[i].barrier_flags_;
        comm_data[i] = communicators[i].data_;
    }
    for (int i = 0; i < communicators.size(); ++i) {
        for (int r = 0; r < world_size; ++r) {
            communicators[i].comm_barrier_flags_[r] = comm_barrier_flags[communicators[i].get_dst_id(r)];
            communicators[i].comm_data_[r] = comm_data[communicators[i].get_dst_id(r)];
        }
    }
}

template <typename T, int NRanks, int BLOCK_SIZE>
__global__ void __launch_bounds__(BLOCK_SIZE, 1) allreduce_direct_kernel(int size, CommDeviceMeta<NRanks> meta, CommPtrs cptrs) {
    static constexpr int VEC_SIZE = details::kBytesPerAccess / sizeof(T);
    SyncComm<NRanks> comm(meta);
    using vec_t_ = kernel_utils::vec_t<T, VEC_SIZE>;
    using acc_vec_t_ = kernel_utils::vec_t<float, VEC_SIZE>;
    comm.template sync<true, false>();
    vec_t_ vec;
    acc_vec_t_ acc_vec;
    for (
        int idx = ((meta.rank + blockIdx.x * NRanks) * BLOCK_SIZE + threadIdx.x) * VEC_SIZE;
        idx < size;
        idx += gridDim.x * NRanks * BLOCK_SIZE * VEC_SIZE) {
        vec.load(reinterpret_cast<T *>(cptrs.data_ptrs[0]) + idx);
#pragma unroll
        for (int v = 0; v < VEC_SIZE; ++v) {
            acc_vec[v] = (float)vec[v];
        }
#pragma unroll
        for (int r = 1; r < NRanks; ++r) {
            vec.load(reinterpret_cast<T *>(cptrs.data_ptrs[r]) + idx);
#pragma unroll
            for (int v = 0; v < VEC_SIZE; ++v) {
                acc_vec[v] += (float)vec[v];
            }
        }
#pragma unroll
        for (int v = 0; v < VEC_SIZE; ++v) {
            vec[v] = (T)acc_vec[v];
        }
#pragma unroll
        for (int r = 0; r < NRanks; ++r) {
            vec.store(reinterpret_cast<T *>(cptrs.data_ptrs[r]) + idx);
        }
    }
    comm.template sync<true, true>();
}

template <typename T, int NRanks>
void allreduce_kernel_launcher(T *allreduce_in, T *allreduce_out, int size,
                               CommDeviceMeta<NRanks> &meta, CommPtrs &cptrs,
                               gpuStream_t stream) {
    constexpr int VEC_SIZE = details::kBytesPerAccess / sizeof(T);
    constexpr int BLOCK_SIZE = 256;
    constexpr int BLOCK_WORK_SIZE = NRanks * BLOCK_SIZE * VEC_SIZE;
    int nblocks = (size + BLOCK_WORK_SIZE - 1) / BLOCK_WORK_SIZE;
    nblocks = std::min(nblocks, NBLOCKS_PER_GPU);
    dim3 threadsPerBlock(BLOCK_SIZE);
    dim3 numBlocks(nblocks);
    gpuMemcpyAsync(cptrs.data_ptrs[0], allreduce_in, size * sizeof(T), gpuMemcpyDeviceToDevice, stream);
    allreduce_direct_kernel<T, NRanks, BLOCK_SIZE><<<numBlocks, threadsPerBlock, 0, stream>>>(size, meta, cptrs);
    gpuMemcpyAsync(allreduce_out, cptrs.data_ptrs[0], size * sizeof(T), gpuMemcpyDeviceToDevice, stream);
}

template <typename T>
void allreduce_kernel(T *allreduce_in, T *allreduce_out, int size,
                      CommMeta &meta, CommPtrs &cptrs,
                      gpuStream_t stream = 0) {
#define DISPATCH_NRANKS(NRANKS)                                                                        \
    {                                                                                                  \
        CommDeviceMeta<NRANKS> dmeta;                                                                  \
        for (int i = 0; i < NRANKS; ++i) {                                                             \
            dmeta.barrier_flag_ptrs[i] = meta.barrier_flag_ptrs[i];                                    \
        }                                                                                              \
        dmeta.sync_clock = meta.sync_clock;                                                            \
        dmeta.rank = meta.rank;                                                                        \
        dmeta.nranks = meta.nranks;                                                                    \
        allreduce_kernel_launcher<T, NRANKS>(allreduce_in, allreduce_out, size, dmeta, cptrs, stream); \
    }
    int nranks = meta.nranks;
    if (nranks == 8) {
        DISPATCH_NRANKS(8)
    } else if (nranks == 4) {
        DISPATCH_NRANKS(4)
    } else if (nranks == 2) {
        DISPATCH_NRANKS(2)
    } else {
        assert(false);
    }
#undef DISPATCH_NRANKS
}

} // namespace comm_utils
