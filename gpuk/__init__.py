import os
import ctypes
import torch
import torch.distributed as dist
from typing import Tuple


this_dir = os.path.dirname(__file__)
package_name = os.path.basename(this_dir)
filename = os.path.join(os.path.dirname(this_dir), f"lib{package_name}.so")
print("Loading extension from:", filename)
torch.ops.load_library(filename)


CommWorkspace = eval(f"torch.classes.{package_name}.CommWorkspace")
allreduce_rms_fusion_impl = eval(f"torch.ops.{package_name}.allreduce_rms_fusion")


class CommProcess:
    def __init__(self, rank, world_size, size_in_bytes, group):
        torch.cuda.set_device(rank)
        nblocks = 256
        self.comm = CommWorkspace(rank, world_size, nblocks, size_in_bytes)
        handle = self.comm.get_handle()
        handle_list = [None] * world_size
        dist.all_gather_object(handle_list, handle, group=group)
        torch.cuda.synchronize(rank)
        dist.barrier(group=group)
        self.comm.open_handles(handle_list)
        torch.cuda.synchronize(rank)
        dist.barrier(group=group)

    def get_workspace(self, ref):
        return self.comm.get_workspace(ref)


comm = None
world_size_ = None


def setup_env(rank, world_size, max_size_in_bytes=8192*16384, group=None):
    global comm
    global workspace
    global world_size_
    comm = CommProcess(rank, world_size, max_size_in_bytes, group)
    world_size_ = world_size


@torch.library.custom_op("torch_dist_ext::get_workspace", mutates_args=[])
def get_workspace(ref: torch.Tensor) -> torch.Tensor:
    global comm
    assert comm is not None
    return comm.get_workspace(ref)


@torch.library.register_fake("torch_dist_ext::get_workspace")
def get_workspace_fake(ref: torch.Tensor) -> torch.Tensor:
    global world_size_
    assert world_size_ is not None
    nbytes = (world_size_ * 3 + 5) * ctypes.sizeof(ctypes.c_void_p)
    return torch.empty(nbytes, dtype=torch.uint8)


@torch.library.custom_op("torch_dist_ext::allreduce_rms", mutates_args=[])
def allreduce_rms(rank: int, nranks: int, allreduce_in: torch.Tensor, 
    residual_in: torch.Tensor, rms_gamma: torch.Tensor, eps: float, workspace: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
    size = allreduce_in.numel()
    hidden_dim = allreduce_in.shape[-1]
    assert hidden_dim <= 8192
    norm_out = torch.empty_like(allreduce_in)
    residual_out = torch.empty_like(residual_in)
    allreduce_rms_fusion_impl(rank, nranks, allreduce_in, residual_in, rms_gamma, residual_out, norm_out, eps, workspace)
    return norm_out, residual_out


@torch.library.register_fake("torch_dist_ext::allreduce_rms")
def allreduce_rms_fake(rank: int, nranks: int, allreduce_in: torch.Tensor, 
    residual_in: torch.Tensor, rms_gamma: torch.Tensor, eps: float, workspace: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
    norm_out = torch.empty_like(allreduce_in)
    residual_out = torch.empty_like(residual_in)
    return norm_out, residual_out


allreduce_rms = torch.ops.torch_dist_ext.allreduce_rms
