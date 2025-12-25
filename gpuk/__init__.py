import os
import ctypes
import torch
import torch.distributed as dist
from torch.distributed import ProcessGroup
from typing import Tuple
from contextlib import contextmanager


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
ar_fusion_capture_clear = eval(f"{prefix}.ar_fusion_capture_clear")
get_ar_fusion_tensor_handle = eval(f"{prefix}.get_ar_fusion_tensor_handle")
get_ar_fusion_tensor_offset = eval(f"{prefix}.get_ar_fusion_tensor_offset")
get_ar_fusion_captured_handles = eval(f"{prefix}.get_ar_fusion_captured_handles")
get_ar_fusion_captured_offsets = eval(f"{prefix}.get_ar_fusion_captured_offsets")
open_ar_fusion_captured_handles = eval(f"{prefix}.open_ar_fusion_captured_handles")
allreduce_inplace = eval(f"{prefix}.allreduce_inplace")
allreduce_rms = eval(f"{prefix}.allreduce_rms")
fused_rope_rms = eval(f"{prefix}.fused_rope_rms")
fused_mrope_3d_rms = eval(f"{prefix}.fused_mrope_3d_rms")
fused_mrope_3d_rms_set_kv = eval(f"{prefix}.fused_mrope_3d_rms_set_kv")


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


class GPUKDistEnv:
    _SUPPORTED_WORLD_SIZES = [2, 4, 8]

    def __init__(
        self,
        group: ProcessGroup = None,
        device_id: int = None,
        max_size_in_bytes=16384 * 16384,
        comm_ptrs_buf_len=1024 * 256,
        dtype: torch.dtype=torch.bfloat16,
    ) -> None:
        self.group = group
        self.device_id = device_id
        self.rank = dist.get_rank(group=self.group)
        self.world_size = dist.get_world_size(group=self.group)
        self.fptr = None
        torch.cuda.set_device(self.device_id)
        
        if self.world_size == 1:
            return

        if self.world_size not in GPUKDistEnv._SUPPORTED_WORLD_SIZES:
            return

        self.fptr = init_ar_fusion(self.device_id, self.rank, self.world_size, max_size_in_bytes, comm_ptrs_buf_len)
        barrier_handle = get_ar_fusion_barrier_handle(self.fptr)
        data_handle = get_ar_fusion_data_handle(self.fptr)
        self.barrier()
        barrier_handle_list = [None] * self.world_size
        data_handle_list = [None] * self.world_size
        dist.all_gather_object(barrier_handle_list, barrier_handle, group=self.group)
        dist.all_gather_object(data_handle_list, data_handle, group=self.group)
        open_ar_fusion_barrier_handles(self.fptr, barrier_handle_list)
        open_ar_fusion_data_handles(self.fptr, data_handle_list)
        self.barrier()
        self._IS_CAPTURING = False
        self._IS_CAPTURED = False
        self.disabled = False

    def barrier(self):
        torch.cuda.set_device(self.device_id)
        torch.cuda.synchronize(self.device_id)
        dist.barrier(group=self.group)

    def consume_capture(self):
        self.barrier()
        handles = get_ar_fusion_captured_handles(self.fptr)
        offsets = get_ar_fusion_captured_offsets(self.fptr)
        for idx in range(len(handles)):
            handle_list = [None] * self.world_size
            offset_list = [None] * self.world_size
            dist.all_gather_object(handle_list, handles[idx], group=self.group)
            dist.all_gather_object(offset_list, int(offsets[idx].item()), group=self.group)
            self.barrier()
            open_ar_fusion_captured_handles(self.fptr, handle_list, offset_list, idx)
        ar_fusion_capture_clear(self.fptr)
        self.barrier()
    
    @contextmanager
    def capture(self):
        try:
            self._IS_CAPTURING = True
            yield
        finally:
            self._IS_CAPTURING = False
            if not self.disabled:
                self.consume_capture()
    
    def capture_(self, input: torch.Tensor):
        if torch.cuda.is_current_stream_capturing():
            pass
            self._IS_CAPTURED = True
        else:
            if self._IS_CAPTURED:
                self.consume_capture()
                self._IS_CAPTURED = False

    def __del__(self):
        if self.fptr:
            destroy_ar_fusion(self.fptr)
    
    def allreduce_native(self, allreduce_in):
        allreduce_out = allreduce_in.clone()
        dist.all_reduce(allreduce_out, group=self.group)
        return allreduce_out
        
    def allreduce(self, allreduce_in):
        allreduce_out = allreduce_in.clone()
        allreduce_inplace(
            self.fptr,
            allreduce_out)
        return allreduce_out

    def allreduce_add_rms_native(
        self, allreduce_in, residual_in, rms_weight, eps, fp8_out=False
    ):
        def rms_norm_forward(x: torch.Tensor, weight: torch.Tensor, eps: float):
            input_dtype = x.dtype
            variance = x.float().pow(2).mean(-1, keepdim=True)
            x = x * torch.rsqrt(variance + eps)
            x = x.to(input_dtype)
            return weight * x
        dist.all_reduce(allreduce_in, group=self.group)
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
        self.capture_(allreduce_in)
        residual_out = torch.empty_like(residual_in)
        if fp8_out:
            norm_out = torch.empty_like(allreduce_in, dtype=fp8)
            scale_out = torch.empty(
                allreduce_in.shape[0],
                1,
                dtype=torch.float32,
                device=allreduce_in.device,
            )
        else:
            norm_out = torch.empty_like(allreduce_in)
            scale_out = torch.empty(1, dtype=torch.float32, device=allreduce_in.device)
        allreduce_rms(
            self.fptr,
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
