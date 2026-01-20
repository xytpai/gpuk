import torch
from torch import Tensor
from aiter.test_common import checkAllclose, perftest, benchmark
from typing import List
import gpuk


def rms_norm_forward(x: Tensor, weight: Tensor, eps: float):
    input_dtype = x.dtype
    variance = x.float().pow(2).mean(-1, keepdim=True)
    x = x * torch.rsqrt(variance + eps)
    x = x.to(input_dtype)
    return weight * x


def apply_rotary_emb_torch(
    x: Tensor,
    cos: Tensor,
    sin: Tensor,
    is_neox_style: bool,
) -> Tensor:
    cos = cos.unsqueeze(-2).to(x.dtype)
    sin = sin.unsqueeze(-2).to(x.dtype)
    if is_neox_style:
        x1, x2 = torch.chunk(x, 2, dim=-1)
    else:
        x1 = x[..., ::2]
        x2 = x[..., 1::2]
    o1 = x1 * cos - x2 * sin
    o2 = x2 * cos + x1 * sin
    if is_neox_style:
        return torch.cat((o1, o2), dim=-1)
    else:
        return torch.stack((o1, o2), dim=-1).flatten(-2)


@perftest()
def run_torch_rope_rms_2way(
    q0: Tensor,  # contiguous (batch_size * num_tokens0 * num_heads_q * head_size)
    k0: Tensor,  # contiguous (batch_size * num_tokens0 * num_heads_k * head_size)
    q1: Tensor,  # contiguous (batch_size * num_tokens1 * num_heads_q * head_size)
    k1: Tensor,  # contiguous (batch_size * num_tokens1 * num_heads_k * head_size)
    w_q0: Tensor, # contiguous (head_size)
    w_k0: Tensor, # contiguous (head_size)
    w_q1: Tensor, # contiguous (head_size)
    w_k1: Tensor, # contiguous (head_size)
    cos_sin0: Tensor, # contiguous (num_tokens0 * head_size)
    cos_sin1: Tensor, # contiguous (num_tokens1 * head_size)
    batch_size: int,
    num_tokens0: int,
    num_tokens1: int,
    num_heads_q: int,
    num_heads_k: int,
    head_size: int,
    is_interleaved: bool,
    eps: float,
):
    is_neox_style = not is_interleaved
    q0_shape = q0.shape
    k0_shape = k0.shape
    q1_shape = q1.shape
    k1_shape = k1.shape
    q0_by_head = rms_norm_forward(q0.view(batch_size, num_tokens0, num_heads_q, head_size), w_q0, eps)
    k0_by_head = rms_norm_forward(k0.view(batch_size, num_tokens0, num_heads_k, head_size), w_k0, eps)
    q1_by_head = rms_norm_forward(q1.view(batch_size, num_tokens1, num_heads_q, head_size), w_q1, eps)
    k1_by_head = rms_norm_forward(k1.view(batch_size, num_tokens1, num_heads_k, head_size), w_k1, eps)
    cos_sin0 = cos_sin0.view(num_tokens0, head_size)
    cos_sin1 = cos_sin1.view(num_tokens1, head_size)
    cos0, sin0 = cos_sin0.chunk(2, dim=-1)
    cos1, sin1 = cos_sin1.chunk(2, dim=-1)
    q0 = apply_rotary_emb_torch(q0_by_head, cos0, sin0, is_neox_style)
    k0 = apply_rotary_emb_torch(k0_by_head, cos0, sin0, is_neox_style)
    q1 = apply_rotary_emb_torch(q1_by_head, cos1, sin1, is_neox_style)
    k1 = apply_rotary_emb_torch(k1_by_head, cos1, sin1, is_neox_style)
    q0 = q0.reshape(q0_shape)
    k0 = k0.reshape(k0_shape)
    q1 = q1.reshape(q1_shape)
    k1 = k1.reshape(k1_shape)
    q01 = torch.cat([q0, q1], dim=1)
    k01 = torch.cat([k0, k1], dim=1)
    return q01, k01


@perftest()
def run_fused_rope_rms_2way(
    q0: Tensor,  # contiguous (batch_size * num_tokens0 * num_heads_q * head_size)
    k0: Tensor,  # contiguous (batch_size * num_tokens0 * num_heads_k * head_size)
    q1: Tensor,  # contiguous (batch_size * num_tokens1 * num_heads_q * head_size)
    k1: Tensor,  # contiguous (batch_size * num_tokens1 * num_heads_k * head_size)
    w_q0: Tensor, # contiguous (head_size)
    w_k0: Tensor, # contiguous (head_size)
    w_q1: Tensor, # contiguous (head_size)
    w_k1: Tensor, # contiguous (head_size)
    cos_sin0: Tensor, # contiguous (num_tokens0 * head_size)
    cos_sin1: Tensor, # contiguous (num_tokens1 * head_size)
    batch_size: int,
    num_tokens0: int,
    num_tokens1: int,
    num_heads_q: int,
    num_heads_k: int,
    head_size: int,
    is_interleaved: bool,
    eps: float,
):
    q01 = torch.empty((batch_size, num_tokens0 + num_tokens1, num_heads_q, head_size), dtype=q0.dtype, device=q0.device)
    k01 = torch.empty((batch_size, num_tokens0 + num_tokens1, num_heads_k, head_size), dtype=k0.dtype, device=k0.device)
    gpuk.fused_rope_rms_2way(
        q0,
        k0,
        q1,
        k1,
        w_q0,
        w_k0,
        w_q1,
        w_k1,
        cos_sin0,
        cos_sin1,
        batch_size,
        num_tokens0,
        num_tokens1,
        num_heads_q,
        num_heads_k,
        head_size,
        is_interleaved,
        eps,
        q01,
        k01,
    )
    return q01, k01


@benchmark()
def test_rope_rms_2way(
    dtype,
    batch_size,
    num_tokens0,
    num_tokens1,
    num_heads_q,
    num_heads_k,
    head_size,
    is_interleaved,
    eps=1e-6,
):
    q0 = torch.randn(
        (batch_size, num_tokens0, num_heads_q, head_size),
        dtype=dtype,
        device="cuda",
    )
    k0 = torch.randn(
        (batch_size, num_tokens0, num_heads_k, head_size),
        dtype=dtype,
        device="cuda",
    )
    q1 = torch.randn(
        (batch_size, num_tokens1, num_heads_q, head_size),
        dtype=dtype,
        device="cuda",
    )
    k1 = torch.randn(
        (batch_size, num_tokens1, num_heads_k, head_size),
        dtype=dtype,
        device="cuda",
    )
    w_q0 = torch.randn(head_size, dtype=dtype, device="cuda")
    w_k0 = torch.randn(head_size, dtype=dtype, device="cuda")
    w_q1 = torch.randn(head_size, dtype=dtype, device="cuda")
    w_k1 = torch.randn(head_size, dtype=dtype, device="cuda")
    cos_sin0 = torch.randn(
        (num_tokens0, head_size),
        dtype=dtype,
        device="cuda",
    )
    cos_sin1 = torch.randn(
        (num_tokens1, head_size),
        dtype=dtype,
        device="cuda",
    )
    (q01_ref, k01_ref), avg_torch = run_torch_rope_rms_2way(
        q0,
        k0,
        q1,
        k1,
        w_q0,
        w_k0,
        w_q1,
        w_k1,
        cos_sin0,
        cos_sin1,
        batch_size,
        num_tokens0,
        num_tokens1,
        num_heads_q,
        num_heads_k,
        head_size,
        is_interleaved,
        eps,
    )
    (q01, k01), avg_cu = run_fused_rope_rms_2way(
        q0,
        k0,
        q1,
        k1,
        w_q0,
        w_k0,
        w_q1,
        w_k1,
        cos_sin0,
        cos_sin1,
        batch_size,
        num_tokens0,
        num_tokens1,
        num_heads_q,
        num_heads_k,
        head_size,
        is_interleaved,
        eps,
    )

    info = f"dtype:{dtype}, batch_size:{batch_size}, num_tokens0:{num_tokens0}, num_tokens1:{num_tokens1}, num_heads_q:{num_heads_q}, num_heads_k:{num_heads_k}"
    info += f", head_size:{head_size}, is_interleaved:{is_interleaved}, eps:{eps}"
    msg = f"[perf] === {info} === torch avg: {avg_torch:<8.2f} us, cu avg: {avg_cu:<8.2f} us, uplift: {avg_torch/avg_cu-1:<5.1%}"
    checkAllclose(q01_ref, q01, msg="q01", rtol=1e-2, atol=0.05)
    checkAllclose(k01_ref, k01, msg="k01", rtol=1e-2, atol=0.05)
    print(msg, flush=True)


if __name__ == "__main__":
    is_interleaveds = [True, False]
    batch_sizes = [1, 2]
    num_tokens0 = 678
    num_tokens1 = 3608
    num_heads_q = 24
    num_heads_k = 25
    head_sizes = [64, 128]
    dtype = torch.bfloat16
    for is_interleaved in is_interleaveds:
        for batch_size in batch_sizes:
            for head_size in head_sizes:
                test_rope_rms_2way(
                    dtype,
                    batch_size,
                    num_tokens0,
                    num_tokens1,
                    num_heads_q,
                    num_heads_k,
                    head_size,
                    is_interleaved,
                    eps=1e-6,
                )
    print("done")
