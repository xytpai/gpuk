import os
import time
import torch
import torch.multiprocessing as mp
import torch.distributed as dist
import gpuk
import aiter
from aiter.test_common import checkAllclose


def init_world(device_id, num_devices, parts, port=24514):
    _GROUP = None
    if _GROUP is not None:
        return
    torch.cuda.set_device(device_id)
    dist.init_process_group(
        backend="nccl",
        init_method=f"tcp://127.0.0.1:{port}",
        rank=device_id,
        world_size=num_devices,
        device_id=device_id,
    )
    group_size = num_devices // parts
    group_id = device_id // group_size
    group_ranks = list(range(group_id * group_size, (group_id + 1) * group_size))
    _GROUP = dist.new_group(ranks=group_ranks)
    print(f"[init_world] device_id:{device_id}, group_ranks:{group_ranks}", flush=True)
    return _GROUP


def worker(device_id, world_size, parts, allreduce_in_):
    group = init_world(device_id, world_size, parts)
    dist_env = gpuk.GPUKDistEnv(device_id=device_id, group=group)
    rank = dist.get_rank(group=group)
    world_size = dist.get_world_size(group=group)
    for i in range(len(allreduce_in_)):
        local_allreduce_in = allreduce_in_[i][device_id].cuda(device_id)
        num_tokens, hidden_dim = local_allreduce_in.shape

        torch.cuda.synchronize()
        dist.barrier(group=group)
        start_native = time.time()
        output_native = dist_env.allreduce_native(local_allreduce_in)
        torch.cuda.synchronize()
        dist.barrier(group=group)
        end_native = time.time()
        dur_native = end_native - start_native

        torch.cuda.synchronize()
        dist.barrier(group=group)
        start_custom = time.time()
        output_custom = dist_env.allreduce(local_allreduce_in)
        torch.cuda.synchronize()
        dist.barrier(group=group)
        end_custom = time.time()
        dur_custom = end_custom - start_custom

        checkAllclose(
            output_custom.float(),
            output_native.float(),
            rtol=1e-2,
            atol=1e-2,
        )

        if rank == 0:
            print(f"dur_native:{dur_native}, dur_custom:{dur_custom}, speedup:{dur_native/dur_custom}")
    dist.destroy_process_group()


def testcase(
    world_size,
    parts,
    num_tokens,
    hidden_dim,
    dtype,
    nsamples=5,
):
    print(
        f"\n============ world_size:{world_size}, parts:{parts}, num_tokens:{num_tokens}, hidden_dim:{hidden_dim}, dtype:{dtype}, nsamples:{nsamples} ============\n"
    )
    allreduce_in_ = []
    for i in range(nsamples):
        data = torch.empty(world_size, num_tokens, hidden_dim, dtype=dtype, device='cuda').uniform_(-1, 1).cpu()
        allreduce_in_.append(data)
    mp.spawn(
        worker,
        args=(
            world_size,
            parts,
            allreduce_in_,
        ),
        nprocs=world_size,
        join=True,
    )


def main(world_size=8, parts=1):
    num_tokens = 8192
    testcase(
        world_size=world_size,
        parts=parts,
        num_tokens=num_tokens,
        hidden_dim=4096,
        dtype=torch.bfloat16,
    )


if __name__ == "__main__":
    main()
