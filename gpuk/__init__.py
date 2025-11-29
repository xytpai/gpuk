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
get_ar_fusion_barrier_handle = eval(f"{prefix}.get_ar_fusion_barrier_handle")
get_ar_fusion_data_handle = eval(f"{prefix}.get_ar_fusion_data_handle")
open_ar_fusion_barrier_handles = eval(f"{prefix}.open_ar_fusion_barrier_handles")
open_ar_fusion_data_handles = eval(f"{prefix}.open_ar_fusion_data_handles")
ar_fusion_capture = eval(f"{prefix}.ar_fusion_capture")
get_tensor_ipc_handle = eval(f"{prefix}.get_tensor_ipc_handle")
allreduce_rms = eval(f"{prefix}.allreduce_rms")


fp8 = torch.float8_e4m3fnuz
# fp8 = torch.float8_e4m3fn


fp8_max_val_ = {
    torch.float8_e4m3fn: 240,
    torch.float8_e4m3fnuz: 120,
}
fp8_max_val = fp8_max_val_[fp8]
fp8_policy_id_ = {
    torch.float8_e4m3fn: 1,
    torch.float8_e4m3fnuz: 2,
}
fp8_policy_id = fp8_policy_id_[fp8]


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
        self.world_size = world_size
        if world_size == 1:
            return

        if world_size not in ARFusion._SUPPORTED_WORLD_SIZES:
            return

        torch.cuda.set_device(rank)
        self.fptr = init_ar_fusion(rank, world_size, max_size_in_bytes)
        barrier_handle = get_ar_fusion_barrier_handle(self.fptr)
        data_handle = get_ar_fusion_data_handle(self.fptr)
        barrier_handle_list = [None] * world_size
        data_handle_list = [None] * world_size
        dist.all_gather_object(barrier_handle_list, barrier_handle, group=self.group)
        dist.all_gather_object(data_handle_list, data_handle, group=self.group)
        open_ar_fusion_barrier_handles(self.fptr, barrier_handle_list)
        open_ar_fusion_data_handles(self.fptr, data_handle_list)
        self.barrier()
        self.captured_inputs = []
        self.is_capture = False
        
    def barrier(self):
        torch.cuda.set_device(self.rank)
        torch.cuda.synchronize(self.rank)
        dist.barrier(group=self.group)
    
    def capture(self, input: torch.Tensor):
        if torch.cuda.is_current_stream_capturing():
            self.captured_inputs.append(input)
            self.is_capture = True
            return
        if self.is_capture:
            for x in self.captured_inputs:
                handle = get_tensor_ipc_handle(x)
                handle_list = [None] * self.world_size
                dist.all_gather_object(handle_list, handle, group=self.group)
                ar_fusion_capture(self.fptr, x, handle_list)
            self.barrier()

    def __del__(self):
        if self.fptr:
            destroy_ar_fusion(self.fptr)


class DistributedEnv:
    def __init__(self, rank, world_size, dtype=torch.bfloat16, init_process_group=False, port=23339):
        torch.cuda.set_device(rank)
        if init_process_group:
            dist.init_process_group(
                backend="nccl",
                init_method=f"tcp://127.0.0.1:{port}",
                rank=rank,
                world_size=world_size,
            )
        self.rank = rank
        self.world_size = world_size
        self.group = dist.group.WORLD
        self.ar_fusion = ARFusion(group=self.group)
        self.barrier()

    def __del__(self):
        if getattr(self, 'group', None):
            dist.destroy_process_group(self.group)
        else:
            dist.destroy_process_group(None)

    def barrier(self):
        self.ar_fusion.barrier()

    def allreduce_add_rms_native(
        self, allreduce_in, residual_in, rms_weight, eps, fp8_out=False
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
        if fp8_out:
            norm_out_scale, _ = norm_out.float().abs().max(dim=-1, keepdim=True)
            norm_out_scale = norm_out_scale / fp8_max_val
            norm_out = norm_out / norm_out_scale
            norm_out.clamp_(min=-fp8_max_val, max=fp8_max_val)
            norm_out = norm_out.to(fp8)
            return residual_out, norm_out, norm_out_scale
        else:
            scale_out = torch.empty(
                allreduce_in.shape[0],
                1,
                dtype=torch.float32,
                device=allreduce_in.device,
            )
            return residual_out, norm_out, scale_out

    def allreduce_add_rms_fused(
        self, allreduce_in, residual_in, rms_weight, eps, fp8_out=False
    ):
        self.ar_fusion.capture(allreduce_in)
        residual_out = torch.empty_like(residual_in)
        norm_out = torch.empty_like(allreduce_in)
        if fp8_out:
            norm_out = norm_out.to(fp8)
            scale_out = torch.empty(
                allreduce_in.shape[0],
                1,
                dtype=torch.float32,
                device=allreduce_in.device,
            )
        else:
            scale_out = torch.empty(1, dtype=torch.float32, device=allreduce_in.device)
        allreduce_rms(
            self.ar_fusion.fptr,
            allreduce_in,
            residual_in,
            rms_weight,
            residual_out,
            norm_out,
            scale_out,
            eps,
            fp8_policy_id if fp8_out else 0,
        )
        if fp8_out:
            return residual_out, norm_out, scale_out
        else:
            return residual_out, norm_out, scale_out
