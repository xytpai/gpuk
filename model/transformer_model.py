import os
import time
import json
import numpy as np
from dacite import from_dict
from typing import Optional
from dataclasses import dataclass

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.multiprocessing as mp
from torch import Tensor
from safetensors.torch import load_file

import map_state_names
from tensor_parallel import ColumnParallelLinear, RowParallelLinear
from distributed_utils import get_tensor_parallel_world_size
from tokenization import AutoTokenizer
from distributed_utils import init_tensor_parallel


@dataclass
class ModelArgs:
    dim: int = 4096
    n_layers: int = 32
    n_heads: int = 32
    n_kv_heads: Optional[int] = None
    vocab_size: int = -1
    multiple_of: int = 256  # make SwiGLU hidden layer size multiple of large power of 2
    ffn_dim_multiplier: Optional[float] = None
    norm_eps: float = 1e-5
    rope_theta: float = 500000
    max_batch_size: int = 32
    max_seq_len: int = 2048
    dropout_prob: float = 0.0
    dtype: str = 'torch.half'
    rope_interval_split: bool = True
    # Optional
    use_qk_norm: bool = False
    explicit_ffn_dim: Optional[int] = None
    head_dim: Optional[int] = None
    # MoE
    num_experts: int = 0
    num_activated_experts: int = 0
    norm_experts_prob: bool = False
    # Training
    ignore_index: int = -1


class KVCache(nn.Module):
    def __init__(self, max_batch_size, max_seq_length, n_heads, head_size, dtype, device):
        super().__init__()
        cache_shape = (max_batch_size, n_heads, max_seq_length, head_size)
        self.register_buffer('k_cache', torch.zeros(cache_shape, dtype=dtype, device=device))
        self.register_buffer('v_cache', torch.zeros(cache_shape, dtype=dtype, device=device))

    def update(self, input_pos: Tensor, k_val, v_val):
        # input_pos: L[t]
        # k_val, v_val: F[b, nh, t, hs]
        k_out = self.k_cache
        v_out = self.v_cache
        k_out[:, :, input_pos] = k_val
        v_out[:, :, input_pos] = v_val
        return k_out, v_out


class RMSNorm(nn.Module):
    def __init__(self, dim, norm_eps, dtype, device):
        super().__init__()
        self.eps = norm_eps
        self.weight = torch.nn.Parameter(torch.ones(dim, dtype=dtype, device=device))

    def forward(self, x):
        input_dtype = x.dtype
        variance = x.float().pow(2).mean(-1, keepdim=True)
        x = x * torch.rsqrt(variance + self.eps)
        x = x.to(input_dtype)
        return self.weight * x


class Attention(nn.Module):
    def __init__(self, args: ModelArgs, device: str):
        super().__init__()
        assert args.dim % args.n_heads == 0
        self.dropout_prob = args.dropout_prob
        self.n_kv_heads = args.n_heads if args.n_kv_heads is None else args.n_kv_heads
        parallel_size = get_tensor_parallel_world_size()
        self.n_local_heads = args.n_heads // parallel_size
        self.n_local_kv_heads = self.n_kv_heads // parallel_size
        self.n_rep = self.n_local_heads // self.n_local_kv_heads
        self.head_dim = args.head_dim if args.head_dim else args.dim // args.n_heads
        # weights
        self.wq = ColumnParallelLinear(args.dim, args.n_heads * self.head_dim, bias=False, dtype=args.dtype, device=device, gather_output=False)
        self.wk = ColumnParallelLinear(args.dim, self.n_kv_heads * self.head_dim, bias=False, dtype=args.dtype, device=device, gather_output=False)
        self.wv = ColumnParallelLinear(args.dim, self.n_kv_heads * self.head_dim, bias=False, dtype=args.dtype, device=device, gather_output=False)
        self.wo = RowParallelLinear(args.n_heads * self.head_dim, args.dim, bias=False, dtype=args.dtype, device=device, input_is_parallel=True)
        # qk norm
        self.use_qk_norm = args.use_qk_norm
        if self.use_qk_norm:
            self.q_norm = RMSNorm(self.head_dim, args.norm_eps, args.dtype, device)
            self.k_norm = RMSNorm(self.head_dim, args.norm_eps, args.dtype, device)
        self.kv_cache = None
        self.rope_interval_split = args.rope_interval_split

    @staticmethod
    def precompute_freqs_cis(head_dim: int, max_position_embeddings: int, theta: float = 10000.0):
        inv_freqs = 1.0 / (theta ** (
            torch.arange(0, head_dim, 2, dtype=torch.int64)[: (head_dim // 2)].float() / head_dim))
        t = torch.arange(max_position_embeddings, device=inv_freqs.device, dtype=torch.float32)
        freqs = torch.outer(t, inv_freqs) # F(max_position_embeddings, head_dim/2)
        return freqs.cos(), freqs.sin()
    
    def apply_rotary_emb(self, xq, xk, cos, sin):
        cos = cos.unsqueeze(-2)
        sin = sin.unsqueeze(-2) # F[t, 1, head_dim/2]
        def apply(x):
            if self.rope_interval_split:
                x1, x2 = torch.chunk(x.reshape(*x.shape[:-1], -1, 2), 2, dim=-1)
                x1, x2 = x1.squeeze(-1), x2.squeeze(-1)
            else:
                x1, x2 = torch.chunk(x.float(), 2, dim=-1) # 2 * (t, num_heads, head_dim/2)
            y1 = x1 * cos - x2 * sin
            y2 = x2 * cos + x1 * sin
            return torch.cat((y1, y2), dim=-1)
        xq_out = apply(xq)
        xk_out = apply(xk)
        return xq_out.type_as(xq), xk_out.type_as(xk)

    def forward(self, x, attention_mask, cos, sin, input_pos: Optional[Tensor]=None, xa: Optional[Tensor]=None):
        batch_size, seq_length, _ = x.size()
        # Infer xq, xk and xv
        xq = self.wq(x)
        x_for_kv = x if xa is None else xa
        xk = self.wk(x_for_kv)
        xv = self.wv(x_for_kv)
        xq = xq.view(batch_size, seq_length, -1, self.head_dim)
        xk = xk.view(batch_size, seq_length, -1, self.head_dim)
        xv = xv.view(batch_size, seq_length, -1, self.head_dim)
        if self.use_qk_norm:
            xq = self.q_norm(xq)
            xk = self.k_norm(xk)
        # Apply RoPE
        xq, xk = self.apply_rotary_emb(xq, xk, cos, sin)
        # Refine xq, xk and xv shape
        xq, xk, xv = [item.transpose(1, 2).contiguous() for item in [xq, xk, xv]] # -> (b, nh, t, hs)
        if self.kv_cache is not None:
            xk, xv = self.kv_cache.update(input_pos, xk, xv)
        xk = xk.repeat_interleave(self.n_rep, dim=1)
        xv = xv.repeat_interleave(self.n_rep, dim=1)
        # DSPA
        output = F.scaled_dot_product_attention(
            xq, xk, xv, attn_mask=attention_mask, dropout_p=self.dropout_prob)
        # Infer output
        output = output.transpose(1, 2).contiguous().view(batch_size, seq_length, -1)
        return self.wo(output)


def refine_hidden_dim(explicit_ffn_dim, hidden_dim, multiple_of=256, ffn_dim_multiplier=None):
    if explicit_ffn_dim is not None:
        return explicit_ffn_dim
    hidden_dim = int(2 * hidden_dim / 3)
    if ffn_dim_multiplier is not None:
        hidden_dim = int(ffn_dim_multiplier * hidden_dim)
    hidden_dim = multiple_of * ((hidden_dim + multiple_of - 1) // multiple_of)
    return hidden_dim


class FeedForward(nn.Module):
    def __init__(self, args: ModelArgs, device: str):
        super().__init__()
        # Project to 4h. If using swiglu double the output width, see https://arxiv.org/pdf/2002.05202.pdf
        hidden_dim = refine_hidden_dim(args.explicit_ffn_dim, 4 * args.dim, args.multiple_of, args.ffn_dim_multiplier)
        self.w1 = ColumnParallelLinear(args.dim, hidden_dim, bias=False, dtype=args.dtype, device=device, gather_output=False)
        self.w2 = RowParallelLinear(hidden_dim, args.dim, bias=False, dtype=args.dtype, device=device, input_is_parallel=True)
        self.w3 = ColumnParallelLinear(args.dim, hidden_dim, bias=False, dtype=args.dtype, device=device, gather_output=False)

    def forward(self, x):
        return self.w2(F.silu(self.w1(x)) * self.w3(x))


class MOEFeedForward(nn.Module):
    def __init__(self, args: ModelArgs, device: str):
        super().__init__()
        self.num_experts = args.num_experts
        self.num_activated_experts = args.num_activated_experts
        self.norm_experts_prob = args.norm_experts_prob
        self.gate = nn.Linear(args.dim, args.num_experts, bias=False)
        self.experts = nn.ModuleList([FeedForward(args, device) for _ in range(self.num_experts)])

    def forward(self, x):
        batch_size, seq_length, dim = x.shape
        x = x.view(-1, dim)
        routing_weights = F.softmax(self.gate(x), dim=-1, dtype=torch.float) # -> F(BxS, num_experts)
        routing_weights, selected_experts = torch.topk(routing_weights, self.num_activated_experts, dim=-1)
        if self.norm_experts_prob:
            routing_weights /= routing_weights.sum(dim=-1, keepdim=True)
        routing_weights = routing_weights.to(x.dtype) # -> F(BxS, num_activated_experts)

        output = torch.zeros((batch_size * seq_length, dim), dtype=x.dtype, device=x.device)
        expert_mask = F.one_hot(selected_experts, num_classes=self.num_experts).permute(2, 1, 0)
        # -> F(num_experts, num_activated_experts, BxS)
        expert_hit = torch.greater(expert_mask.sum(dim=(-1, -2)), 0).nonzero()
        for expert_idx in expert_hit:
            expert_layer = self.experts[expert_idx]
            idx, top_x = torch.where(expert_mask[expert_idx].squeeze(0))
            current_state = x[None, top_x].reshape(-1, dim)
            current_hidden_states = expert_layer(current_state) * routing_weights[top_x, idx, None]
            output.index_add_(0, top_x, current_hidden_states.to(x.dtype))
        output = output.reshape(batch_size, seq_length, dim)
        return output


class TransformerBlock(nn.Module):
    def __init__(self, layer_id: int, args: ModelArgs, device: str):
        super().__init__()
        self.attention = Attention(args, device)
        if getattr(args, 'enable_visual', None):
            self.visual_attention = Attention(args, device)
            self.visual_norm = RMSNorm(args.dim, args.norm_eps, args.dtype, device)
            self.enable_visual = True
        else:
            self.enable_visual = False
        if getattr(args, 'enable_audio', None):
            self.audio_attention = Attention(args, device)
            self.audio_norm = RMSNorm(args.dim, args.norm_eps, args.dtype, device)
            self.enable_audio = True
        else:
            self.enable_audio = False
        self.layer_id = layer_id
        self.feed_forward = FeedForward(args, device)
        self.attention_norm = RMSNorm(args.dim, args.norm_eps, args.dtype, device)
        self.ffn_norm = RMSNorm(args.dim, args.norm_eps, args.dtype, device)

    def forward(self, x, attention_mask, cos, sin, input_pos: Optional[Tensor]=None, xa: Optional[Tensor]=None):
        x = x + self.attention(self.attention_norm(x), attention_mask, cos, sin, input_pos)
        if self.enable_visual:
            x = x + self.visual_attention(self.visual_norm(x), attention_mask, cos, sin, input_pos, xa)
        if self.enable_audio:
            x = x + self.audio_attention(self.audio_norm(x), attention_mask, cos, sin, input_pos, xa)
        out = x + self.feed_forward(self.ffn_norm(x))
        return out


class Transformer(nn.Module):
    def __init__(self, args: ModelArgs, device='cuda:0'):
        super().__init__()
        self.args = args
        self.vocab_size = args.vocab_size
        self.n_layers = args.n_layers
        self.max_seq_len = args.max_seq_len
        self.tok_embeddings = nn.Embedding(args.vocab_size, args.dim, dtype=args.dtype, device=device)
        self.layers = nn.ModuleList([TransformerBlock(layer_id, args, device) for layer_id in range(args.n_layers)])
        self.norm = RMSNorm(args.dim, args.norm_eps, args.dtype, device)
        self.output = ColumnParallelLinear(args.dim, self.vocab_size, bias=False, dtype=args.dtype, device=device, gather_output=True)
        self.head_dim = args.head_dim if args.head_dim else args.dim // args.n_heads
        self.cos, self.sin = Attention.precompute_freqs_cis(
            self.head_dim,
            self.max_seq_len * 2,
            args.rope_theta,
        )
        self.register_buffer('causal_mask', torch.tril(
            torch.ones(self.max_seq_len, self.max_seq_len, dtype=torch.bool, device=device)), persistent=False)

    def device(self):
        return self.output.weight.device
    
    def dtype(self):
        return self.args.dtype
    
    def size(self):
        return sum([np.prod(list(p.size())) for p in self.parameters()])
    
    def assign_kv_cache(self, max_batch_size):
        for layer in self.layers:
            layer.attention.kv_cache = KVCache(
                max_batch_size, self.max_seq_len, self.args.n_kv_heads, 
                self.head_dim,
                self.dtype(), self.device())
    
    def forward(
        self, 
        tokens, 
        input_pos: Optional[Tensor]=None, 
        images: Optional[Tensor]=None):
        # tokens: L[b, t]
        # images: F[b, c, h, w]
        _, seq_length = tokens.size()
        h = self.tok_embeddings(tokens) # [b, t, hidden_size]
        self.cos = self.cos.to(self.device())
        self.sin = self.sin.to(self.device())

        if input_pos is None:
            input_pos = torch.arange(0, seq_length, device=self.device())
            cos, sin = self.cos[input_pos], self.sin[input_pos]
            causal_mask = self.causal_mask[None, None, :seq_length, :seq_length]
        else: # KV Cache enabled
            cos, sin = self.cos[input_pos], self.sin[input_pos]
            causal_mask = self.causal_mask[None, None, input_pos]
        for layer in self.layers:
            h = layer(h, causal_mask, cos, sin, input_pos)
        h = self.norm(h)
        return self.output(h)

    @staticmethod
    def post_loss(h, target_ids):
        # h -> F[b, t, vocab_size]
        loss = F.cross_entropy(h.view(-1, h.size(-1)), 
            target_ids.contiguous().view(-1), 
            ignore_index=self.args.ignore_index)
        return loss

    @staticmethod
    def post_pred(h, temperature):
        h = h[:, -1, :] / temperature # -> F[b, vocab_size]
        ids_next = torch.multinomial(F.softmax(h, dim=-1), num_samples=1) # b
        return ids_next


class WorkerProc:
    def __init__(self, file, world_size=1, tp_size=1):
        with open(file, 'r', encoding='utf-8') as f:
            jf = json.load(f)
        args = from_dict(data_class=ModelArgs, data=jf)
        args.dtype = eval(args.dtype)
        self.args = args
        base_dir = jf['base_dir']
        tfile = os.path.join(base_dir, jf['tfile'])
        self.tokenizer = AutoTokenizer(tfile)
        self.load_state_dict(jf)
        self.world_size = world_size
        self.tp_size = tp_size
        self.temperature = 0.9
    
    def load_state_dict(self, jf):
        base_dir = jf['base_dir']
        state_dict = {}
        for wfile in jf['wfiles']:
            wfile = os.path.join(base_dir, wfile)
            if wfile.endswith('.safetensors'):
                ckpt = load_file(wfile, device='cpu')
            else:
                ckpt = torch.load(wfile, map_location='cpu')
            state_dict.update(ckpt)
        # state_dict = map_state_names.run(state_dict)
        self.state_dict = state_dict
    
    @torch.no_grad()
    def generate(self, rank, model, input_text):
        self.max_seq_len = model.max_seq_len
        model.assign_kv_cache(1)
        input_ids = self.tokenizer.encode(input_text)
        T = len(input_ids)
        prompt = torch.empty([1, self.max_seq_len], device=model.device(), dtype=torch.long)
        input_pos = torch.arange(0, T, device=model.device())
        prompt[:, input_pos] = torch.LongTensor(input_ids).to(model.device()).view(1, -1)
        latency = []
        for i in range(self.max_seq_len - T):
            torch.cuda.synchronize()
            start = time.time()
            # with torch.backends.cuda.sdp_kernel(
            #     enable_flash=self.enable_flash, enable_mem_efficient=self.enable_flash, enable_math=True):
            if True:
                h = model(prompt[:, input_pos], input_pos)
                idx_next = Transformer.post_pred(h, self.temperature)
            torch.cuda.synchronize()
            end = time.time()
            latency.append(end - start)
            idx_next_item = idx_next[:, -1].item()
            word_next = self.tokenizer.decode([idx_next_item])
            if idx_next_item in self.tokenizer.stop_tokens:
                break
            if rank == 0:
                print(word_next, end='', flush=True)
            input_pos = torch.tensor([T], device=model.device(), dtype=torch.int)
            prompt[:, T] = idx_next
            T += 1
        latency = latency[10:]
        token_per_s = 1.0 / (sum(latency) / len(latency))
        if rank == 0:
            print(f'\ntoken_per_s:{token_per_s}')
        return prompt
    
    def __call__(self, rank, input_text):
        init_tensor_parallel(rank, self.world_size, self.tp_size)
        model = Transformer(self.args, f"cuda:{rank}")
        if self.world_size == 1:
            missing_keys, unexpected_keys = model.load_state_dict(self.state_dict, strict=True)
            if len(missing_keys) > 0 or len(unexpected_keys) > 0:
                print('load model: ' + str({'missing_keys':missing_keys, 'unexpected_keys':unexpected_keys}))
        else:
            raise
        if rank == 0:
            type_size = 1
            print('Model {} : params: {:4f}B'.format(model._get_name(), model.size() * type_size / 1000 / 1000 / 1000))
        self.generate(rank, model, input_text)
import os
import time
import json
import numpy as np
from dacite import from_dict
from typing import Optional
from dataclasses import dataclass

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.multiprocessing as mp
from torch import Tensor
from safetensors.torch import load_file

import map_state_names
from tensor_parallel import ColumnParallelLinear, RowParallelLinear
from distributed_utils import get_tensor_parallel_world_size
from tokenization import AutoTokenizer
from distributed_utils import init_tensor_parallel


@dataclass
class ModelArgs:
    dim: int = 4096
    n_layers: int = 32
    n_heads: int = 32
    n_kv_heads: Optional[int] = None
    vocab_size: int = -1
    multiple_of: int = 256  # make SwiGLU hidden layer size multiple of large power of 2
    ffn_dim_multiplier: Optional[float] = None
    norm_eps: float = 1e-5
    rope_theta: float = 500000
    max_batch_size: int = 32
    max_seq_len: int = 2048
    dropout_prob: float = 0.0
    dtype: str = 'torch.half'
    rope_interval_split: bool = True
    # Optional
    use_qk_norm: bool = False
    explicit_ffn_dim: Optional[int] = None
    head_dim: Optional[int] = None
    # MoE
    num_experts: int = 0
    num_activated_experts: int = 0
    norm_experts_prob: bool = False
    # Training
    ignore_index: int = -1


class KVCache(nn.Module):
    def __init__(self, max_batch_size, max_seq_length, n_heads, head_size, dtype, device):
        super().__init__()
        cache_shape = (max_batch_size, n_heads, max_seq_length, head_size)
        self.register_buffer('k_cache', torch.zeros(cache_shape, dtype=dtype, device=device))
        self.register_buffer('v_cache', torch.zeros(cache_shape, dtype=dtype, device=device))

    def update(self, input_pos: Tensor, k_val, v_val):
        # input_pos: L[t]
        # k_val, v_val: F[b, nh, t, hs]
        k_out = self.k_cache
        v_out = self.v_cache
        k_out[:, :, input_pos] = k_val
        v_out[:, :, input_pos] = v_val
        return k_out, v_out


class RMSNorm(nn.Module):
    def __init__(self, dim, norm_eps, dtype, device):
        super().__init__()
        self.eps = norm_eps
        self.weight = torch.nn.Parameter(torch.ones(dim, dtype=dtype, device=device))

    def forward(self, x):
        input_dtype = x.dtype
        variance = x.float().pow(2).mean(-1, keepdim=True)
        x = x * torch.rsqrt(variance + self.eps)
        x = x.to(input_dtype)
        return self.weight * x


class Attention(nn.Module):
    def __init__(self, args: ModelArgs, device: str):
        super().__init__()
        assert args.dim % args.n_heads == 0
        self.dropout_prob = args.dropout_prob
        self.n_kv_heads = args.n_heads if args.n_kv_heads is None else args.n_kv_heads
        parallel_size = get_tensor_parallel_world_size()
        self.n_local_heads = args.n_heads // parallel_size
        self.n_local_kv_heads = self.n_kv_heads // parallel_size
        self.n_rep = self.n_local_heads // self.n_local_kv_heads
        self.head_dim = args.head_dim if args.head_dim else args.dim // args.n_heads
        # weights
        self.wq = ColumnParallelLinear(args.dim, args.n_heads * self.head_dim, bias=False, dtype=args.dtype, device=device, gather_output=False)
        self.wk = ColumnParallelLinear(args.dim, self.n_kv_heads * self.head_dim, bias=False, dtype=args.dtype, device=device, gather_output=False)
        self.wv = ColumnParallelLinear(args.dim, self.n_kv_heads * self.head_dim, bias=False, dtype=args.dtype, device=device, gather_output=False)
        self.wo = RowParallelLinear(args.n_heads * self.head_dim, args.dim, bias=False, dtype=args.dtype, device=device, input_is_parallel=True)
        # qk norm
        self.use_qk_norm = args.use_qk_norm
        if self.use_qk_norm:
            self.q_norm = RMSNorm(self.head_dim, args.norm_eps, args.dtype, device)
            self.k_norm = RMSNorm(self.head_dim, args.norm_eps, args.dtype, device)
        self.kv_cache = None
        self.rope_interval_split = args.rope_interval_split

    @staticmethod
    def precompute_freqs_cis(head_dim: int, max_position_embeddings: int, theta: float = 10000.0):
        inv_freqs = 1.0 / (theta ** (
            torch.arange(0, head_dim, 2, dtype=torch.int64)[: (head_dim // 2)].float() / head_dim))
        t = torch.arange(max_position_embeddings, device=inv_freqs.device, dtype=torch.float32)
        freqs = torch.outer(t, inv_freqs) # F(max_position_embeddings, head_dim/2)
        return freqs.cos(), freqs.sin()
    
    def apply_rotary_emb(self, xq, xk, cos, sin):
        cos = cos.unsqueeze(-2)
        sin = sin.unsqueeze(-2) # F[t, 1, head_dim/2]
        def apply(x):
            if self.rope_interval_split:
                x1, x2 = torch.chunk(x.reshape(*x.shape[:-1], -1, 2), 2, dim=-1)
                x1, x2 = x1.squeeze(-1), x2.squeeze(-1)
            else:
                x1, x2 = torch.chunk(x.float(), 2, dim=-1) # 2 * (t, num_heads, head_dim/2)
            y1 = x1 * cos - x2 * sin
            y2 = x2 * cos + x1 * sin
            return torch.cat((y1, y2), dim=-1)
        xq_out = apply(xq)
        xk_out = apply(xk)
        return xq_out.type_as(xq), xk_out.type_as(xk)

    def forward(self, x, attention_mask, cos, sin, input_pos: Optional[Tensor]=None, xa: Optional[Tensor]=None):
        batch_size, seq_length, _ = x.size()
        # Infer xq, xk and xv
        xq = self.wq(x)
        x_for_kv = x if xa is None else xa
        xk = self.wk(x_for_kv)
        xv = self.wv(x_for_kv)
        xq = xq.view(batch_size, seq_length, -1, self.head_dim)
        xk = xk.view(batch_size, seq_length, -1, self.head_dim)
        xv = xv.view(batch_size, seq_length, -1, self.head_dim)
        if self.use_qk_norm:
            xq = self.q_norm(xq)
            xk = self.k_norm(xk)
        # Apply RoPE
        xq, xk = self.apply_rotary_emb(xq, xk, cos, sin)
        # Refine xq, xk and xv shape
        xq, xk, xv = [item.transpose(1, 2).contiguous() for item in [xq, xk, xv]] # -> (b, nh, t, hs)
        if self.kv_cache is not None:
            xk, xv = self.kv_cache.update(input_pos, xk, xv)
        xk = xk.repeat_interleave(self.n_rep, dim=1)
        xv = xv.repeat_interleave(self.n_rep, dim=1)
        # DSPA
        output = F.scaled_dot_product_attention(
            xq, xk, xv, attn_mask=attention_mask, dropout_p=self.dropout_prob)
        # Infer output
        output = output.transpose(1, 2).contiguous().view(batch_size, seq_length, -1)
        return self.wo(output)


def refine_hidden_dim(explicit_ffn_dim, hidden_dim, multiple_of=256, ffn_dim_multiplier=None):
    if explicit_ffn_dim is not None:
        return explicit_ffn_dim
    hidden_dim = int(2 * hidden_dim / 3)
    if ffn_dim_multiplier is not None:
        hidden_dim = int(ffn_dim_multiplier * hidden_dim)
    hidden_dim = multiple_of * ((hidden_dim + multiple_of - 1) // multiple_of)
    return hidden_dim


class FeedForward(nn.Module):
    def __init__(self, args: ModelArgs, device: str):
        super().__init__()
        # Project to 4h. If using swiglu double the output width, see https://arxiv.org/pdf/2002.05202.pdf
        hidden_dim = refine_hidden_dim(args.explicit_ffn_dim, 4 * args.dim, args.multiple_of, args.ffn_dim_multiplier)
        self.w1 = ColumnParallelLinear(args.dim, hidden_dim, bias=False, dtype=args.dtype, device=device, gather_output=False)
        self.w2 = RowParallelLinear(hidden_dim, args.dim, bias=False, dtype=args.dtype, device=device, input_is_parallel=True)
        self.w3 = ColumnParallelLinear(args.dim, hidden_dim, bias=False, dtype=args.dtype, device=device, gather_output=False)

    def forward(self, x):
        return self.w2(F.silu(self.w1(x)) * self.w3(x))


class MOEFeedForward(nn.Module):
    def __init__(self, args: ModelArgs, device: str):
        super().__init__()
        self.num_experts = args.num_experts
        self.num_activated_experts = args.num_activated_experts
        self.norm_experts_prob = args.norm_experts_prob
        self.gate = nn.Linear(args.dim, args.num_experts, bias=False)
        self.experts = nn.ModuleList([FeedForward(args, device) for _ in range(self.num_experts)])

    def forward(self, x):
        batch_size, seq_length, dim = x.shape
        x = x.view(-1, dim)
        routing_weights = F.softmax(self.gate(x), dim=-1, dtype=torch.float) # -> F(BxS, num_experts)
        routing_weights, selected_experts = torch.topk(routing_weights, self.num_activated_experts, dim=-1)
        if self.norm_experts_prob:
            routing_weights /= routing_weights.sum(dim=-1, keepdim=True)
        routing_weights = routing_weights.to(x.dtype) # -> F(BxS, num_activated_experts)

        output = torch.zeros((batch_size * seq_length, dim), dtype=x.dtype, device=x.device)
        expert_mask = F.one_hot(selected_experts, num_classes=self.num_experts).permute(2, 1, 0)
        # -> F(num_experts, num_activated_experts, BxS)
        expert_hit = torch.greater(expert_mask.sum(dim=(-1, -2)), 0).nonzero()
        for expert_idx in expert_hit:
            expert_layer = self.experts[expert_idx]
            idx, top_x = torch.where(expert_mask[expert_idx].squeeze(0))
            current_state = x[None, top_x].reshape(-1, dim)
            current_hidden_states = expert_layer(current_state) * routing_weights[top_x, idx, None]
            output.index_add_(0, top_x, current_hidden_states.to(x.dtype))
        output = output.reshape(batch_size, seq_length, dim)
        return output


class TransformerBlock(nn.Module):
    def __init__(self, layer_id: int, args: ModelArgs, device: str):
        super().__init__()
        self.attention = Attention(args, device)
        if getattr(args, 'enable_visual', None):
            self.visual_attention = Attention(args, device)
            self.visual_norm = RMSNorm(args.dim, args.norm_eps, args.dtype, device)
            self.enable_visual = True
        else:
            self.enable_visual = False
        if getattr(args, 'enable_audio', None):
            self.audio_attention = Attention(args, device)
            self.audio_norm = RMSNorm(args.dim, args.norm_eps, args.dtype, device)
            self.enable_audio = True
        else:
            self.enable_audio = False
        self.layer_id = layer_id
        self.feed_forward = FeedForward(args, device)
        self.attention_norm = RMSNorm(args.dim, args.norm_eps, args.dtype, device)
        self.ffn_norm = RMSNorm(args.dim, args.norm_eps, args.dtype, device)

    def forward(self, x, attention_mask, cos, sin, input_pos: Optional[Tensor]=None, xa: Optional[Tensor]=None):
        x = x + self.attention(self.attention_norm(x), attention_mask, cos, sin, input_pos)
        if self.enable_visual:
            x = x + self.visual_attention(self.visual_norm(x), attention_mask, cos, sin, input_pos, xa)
        if self.enable_audio:
            x = x + self.audio_attention(self.audio_norm(x), attention_mask, cos, sin, input_pos, xa)
        out = x + self.feed_forward(self.ffn_norm(x))
        return out


@torch.compile
class Transformer(nn.Module):
    def __init__(self, args: ModelArgs, device='cuda:0'):
        super().__init__()
        self.args = args
        self.vocab_size = args.vocab_size
        self.n_layers = args.n_layers
        self.max_seq_len = args.max_seq_len
        self.tok_embeddings = nn.Embedding(args.vocab_size, args.dim, dtype=args.dtype, device=device)
        self.layers = nn.ModuleList([TransformerBlock(layer_id, args, device) for layer_id in range(args.n_layers)])
        self.norm = RMSNorm(args.dim, args.norm_eps, args.dtype, device)
        self.output = ColumnParallelLinear(args.dim, self.vocab_size, bias=False, dtype=args.dtype, device=device, gather_output=True)
        self.head_dim = args.head_dim if args.head_dim else args.dim // args.n_heads
        self.cos, self.sin = Attention.precompute_freqs_cis(
            self.head_dim,
            self.max_seq_len * 2,
            args.rope_theta,
        )
        self.register_buffer('causal_mask', torch.tril(
            torch.ones(self.max_seq_len, self.max_seq_len, dtype=torch.bool, device=device)), persistent=False)

    def device(self):
        return self.output.weight.device
    
    def dtype(self):
        return self.args.dtype
    
    def size(self):
        return sum([np.prod(list(p.size())) for p in self.parameters()])
    
    def assign_kv_cache(self, max_batch_size):
        for layer in self.layers:
            layer.attention.kv_cache = KVCache(
                max_batch_size, self.max_seq_len, self.args.n_kv_heads, 
                self.head_dim,
                self.dtype(), self.device())
    
    def forward(
        self, 
        tokens, 
        input_pos: Optional[Tensor]=None, 
        images: Optional[Tensor]=None):
        # tokens: L[b, t]
        # images: F[b, c, h, w]
        _, seq_length = tokens.size()
        h = self.tok_embeddings(tokens) # [b, t, hidden_size]
        self.cos = self.cos.to(self.device())
        self.sin = self.sin.to(self.device())

        if input_pos is None:
            input_pos = torch.arange(0, seq_length, device=self.device())
            cos, sin = self.cos[input_pos], self.sin[input_pos]
            causal_mask = self.causal_mask[None, None, :seq_length, :seq_length]
        else: # KV Cache enabled
            cos, sin = self.cos[input_pos], self.sin[input_pos]
            causal_mask = self.causal_mask[None, None, input_pos]
        for layer in self.layers:
            h = layer(h, causal_mask, cos, sin, input_pos)
        h = self.norm(h)
        return self.output(h)

    @staticmethod
    def post_loss(h, target_ids):
        # h -> F[b, t, vocab_size]
        loss = F.cross_entropy(h.view(-1, h.size(-1)), 
            target_ids.contiguous().view(-1), 
            ignore_index=self.args.ignore_index)
        return loss

    @staticmethod
    def post_pred(h, temperature):
        h = h[:, -1, :] / temperature # -> F[b, vocab_size]
        ids_next = torch.multinomial(F.softmax(h, dim=-1), num_samples=1) # b
        return ids_next


class WorkerProc:
    def __init__(self, file, world_size=1, tp_size=1):
        with open(file, 'r', encoding='utf-8') as f:
            jf = json.load(f)
        args = from_dict(data_class=ModelArgs, data=jf)
        args.dtype = eval(args.dtype)
        self.args = args
        base_dir = jf['base_dir']
        tfile = os.path.join(base_dir, jf['tfile'])
        self.tokenizer = AutoTokenizer(tfile)
        self.load_state_dict(jf)
        self.world_size = world_size
        self.tp_size = tp_size
        self.temperature = 0.9
    
    def load_state_dict(self, jf):
        base_dir = jf['base_dir']
        state_dict = {}
        for wfile in jf['wfiles']:
            wfile = os.path.join(base_dir, wfile)
            if wfile.endswith('.safetensors'):
                ckpt = load_file(wfile, device='cpu')
            else:
                ckpt = torch.load(wfile, map_location='cpu')
            state_dict.update(ckpt)
        state_dict = map_state_names.run(state_dict)
        self.state_dict = state_dict
    
    @torch.no_grad()
    def generate(self, rank, model, input_text):
        self.max_seq_len = model.max_seq_len
        model.assign_kv_cache(1)
        input_ids = self.tokenizer.encode(input_text)
        T = len(input_ids)
        prompt = torch.empty([1, self.max_seq_len], device=model.device(), dtype=torch.long)
        input_pos = torch.arange(0, T, device=model.device())
        prompt[:, input_pos] = torch.LongTensor(input_ids).to(model.device()).view(1, -1)
        latency = []
        for i in range(self.max_seq_len - T):
            torch.cuda.synchronize()
            start = time.time()
            # with torch.backends.cuda.sdp_kernel(
            #     enable_flash=self.enable_flash, enable_mem_efficient=self.enable_flash, enable_math=True):
            if True:
                h = model(prompt[:, input_pos], input_pos)
                idx_next = Transformer.post_pred(h, self.temperature)
            torch.cuda.synchronize()
            end = time.time()
            latency.append(end - start)
            idx_next_item = idx_next[:, -1].item()
            word_next = self.tokenizer.decode([idx_next_item])
            if idx_next_item in self.tokenizer.stop_tokens:
                break
            if rank == 0:
                print(word_next, end='', flush=True)
            input_pos = torch.tensor([T], device=model.device(), dtype=torch.int)
            prompt[:, T] = idx_next
            T += 1
        latency = latency[10:]
        token_per_s = 1.0 / (sum(latency) / len(latency))
        if rank == 0:
            print(f'\ntoken_per_s:{token_per_s}')
        return prompt
    
    def __call__(self, rank, input_text):
        init_tensor_parallel(rank, self.world_size, self.tp_size)
        model = Transformer(self.args, f"cuda:{rank}")
        if self.world_size == 1:
            missing_keys, unexpected_keys = model.load_state_dict(self.state_dict, strict=True)
            if len(missing_keys) > 0 or len(unexpected_keys) > 0:
                print('load model: ' + str({'missing_keys':missing_keys, 'unexpected_keys':unexpected_keys}))
        else:
            raise
        if rank == 0:
            type_size = 1
            print('Model {} : params: {:4f}B'.format(model._get_name(), model.size() * type_size / 1000 / 1000 / 1000))
        
        if False:
            with torch.profiler.profile(
                activities=[torch.profiler.ProfilerActivity.CUDA],
            ) as prof:
                self.generate(rank, model, input_text)
            print(prof.key_averages().table(sort_by="cuda_time_total", row_limit=-1))
        else:
            self.generate(rank, model, input_text)


if __name__ == '__main__':
    import sys
    cpath = sys.argv[1]
    input_text = sys.argv[2]
    world_size = 1
    tp_size = 1
    worker = WorkerProc(cpath, world_size, tp_size)
    mp.spawn(worker, args=(input_text, ), nprocs=world_size, join=True)
