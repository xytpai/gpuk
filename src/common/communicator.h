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
#elif defined(__HIPCC__)
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
#elif defined(__HIPCC__)
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

} // namespace comm_utils
