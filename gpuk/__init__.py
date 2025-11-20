import os
import ctypes
import torch
import torch.distributed as dist
from torch.distributed import ProcessGroup
from typing import Tuple


this_dir = os.path.dirname(__file__)
package_name = os.path.basename(this_dir)
filename = os.path.join(os.path.dirname(this_dir), f"lib{package_name}.so")
print("Loading extension from:", filename)
torch.ops.load_library(filename)
prefix = f"torch.ops.{package_name}"


init_ar_fusion = eval(f"{prefix}.init_ar_fusion")
destroy_ar_fusion = eval(f"{prefix}.destroy_ar_fusion")
get_ar_fusion_handle = eval(f"{prefix}.get_ar_fusion_handle")
open_ar_fusion_handles = eval(f"{prefix}.open_ar_fusion_handles")
get_ar_fusion_workspace = eval(f"{prefix}.get_ar_fusion_workspace")
allreduce_rms = eval(f"{prefix}.allreduce_rms")


class ARFusion:
    _SUPPORTED_WORLD_SIZES = [2, 4, 8]

    def __init__(
        self,
        group: ProcessGroup = None,
        max_size_in_bytes=16384 * 16384,
    ) -> None:
        self.group = group
        rank = dist.get_rank(group=self.group)
        torch.cuda.set_device(rank)
        self.rank = rank
        self.fptr = None
        world_size = dist.get_world_size(group=self.group)
        if world_size == 1:
            return

        if world_size not in ARFusion._SUPPORTED_WORLD_SIZES:
            return

        torch.cuda.set_device(rank)
        self.fptr = init_ar_fusion(rank, world_size, max_size_in_bytes)
        handle = get_ar_fusion_handle(self.fptr)
        handle_list = [None] * world_size
        dist.all_gather_object(handle_list, handle, group=self.group)
        open_ar_fusion_handles(self.fptr, handle_list)
        torch.cuda.synchronize(rank)
        dist.barrier(group=group)

    def get_workspace(self, ref: torch.Tensor):
        return get_ar_fusion_workspace(self.fptr, ref)

    def __del__(self):
        if self.fptr:
            destroy_ar_fusion(self.fptr)


class DistributedEnv:
    def __init__(self, rank, world_size):
        torch.cuda.set_device(rank)
        dist.init_process_group(
            backend="nccl",
            init_method="tcp://127.0.0.1:22229",
            rank=rank,
            world_size=world_size,
        )
        self.rank = rank
        self.world_size = world_size
        self.group = dist.group.WORLD
        self.ar_fusion = ARFusion(group=self.group)
        self.barrier()

    def __del__(self):
        dist.destroy_process_group(self.group)

    def barrier(self):
        torch.cuda.set_device(self.rank)
        dist.barrier(self.group)
        torch.cuda.synchronize()

    def allreduce_add_rms_native(
        self, allreduce_in, residual_in, rms_weight, eps
    ):
        def rms_norm_forward(x: torch.Tensor, weight: torch.Tensor, eps: float):
            input_dtype = x.dtype
            variance = x.float().pow(2).mean(-1, keepdim=True)
            x = x * torch.rsqrt(variance + eps)
            x = x.to(input_dtype)
            return weight * x
        dist.all_reduce(allreduce_in)
        residual_out = allreduce_in + residual_in
        norm_out = rms_norm_forward(residual_out, rms_weight, eps)
        return residual_out, norm_out

    def allreduce_add_rms_fused(
        self, allreduce_in, residual_in, rms_weight, eps
    ):
        residual_out = torch.empty_like(residual_in)
        norm_out = torch.empty_like(allreduce_in)
        allreduce_rms(
            self.rank,
            self.world_size,
            allreduce_in,
            residual_in,
            rms_weight,
            residual_out,
            norm_out,
            eps,
            self.ar_fusion.get_workspace(allreduce_in),
        )
        return residual_out, norm_out
