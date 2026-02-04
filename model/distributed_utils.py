import torch
from typing import Tuple, Any


def ensure_divisibility(numerator: int, denominator: int) -> None:
    """Ensure that numerator is divisible by the denominator."""
    assert numerator % denominator == 0, "{} is not divisible by {}".format(numerator, denominator)


def divide_and_check_no_remainder(numerator: int, denominator: int) -> int:
    """Ensure that numerator is divisible by the denominator and return
    the division value."""
    ensure_divisibility(numerator, denominator)
    return numerator // denominator


def split_tensor_along_last_dim(
    tensor: torch.Tensor, num_partitions: int, contiguous_split_chunks: bool = False
) -> Tuple[torch.Tensor, ...]:
    """Split a tensor along its last dimension.
    Arguments:
        tensor: input tensor.
        num_partitions: number of partitions to split the tensor
        contiguous_split_chunks: If True, make each chunk contiguous
                                 in memory.
    """
    # Get the size and dimension.
    last_dim = tensor.dim() - 1
    last_dim_size = divide_and_check_no_remainder(tensor.size()[last_dim], num_partitions)
    # Split.
    tensor_list = torch.split(tensor, last_dim_size, dim=last_dim)
    # Note: torch.split does not create contiguous tensors by default.
    if contiguous_split_chunks:
        return tuple(chunk.contiguous() for chunk in tensor_list)
    return tensor_list


_RANK = None
_WORLD_SIZE = None
_TENSOR_PARALLEL_GROUP = None


def init_tensor_parallel(rank, world_size, tp_size, backend='nccl', init_url="tcp://127.0.0.1:23456"):
    global _RANK
    global _WORLD_SIZE
    global _TENSOR_PARALLEL_GROUP
    assert _RANK is None, "_RANK is already initialized"
    assert _WORLD_SIZE is None, "_WORLD_SIZE is already initialized"
    assert _TENSOR_PARALLEL_GROUP is None, "_TENSOR_PARALLEL_GROUP is already initialized"
    assert world_size % tp_size == 0

    torch.distributed.init_process_group(
        backend=backend,
        init_method=init_url,
        rank=rank,
        world_size=world_size,
    )

    for i in range(world_size // tp_size):
        ranks = list(range(i * tp_size, (i + 1) * tp_size))
        device_group = torch.distributed.new_group(ranks, backend=backend)
        if rank in ranks:
            _TENSOR_PARALLEL_GROUP = device_group


def get_tensor_parallel_group() -> torch.distributed.ProcessGroup:
    assert _TENSOR_PARALLEL_GROUP is not None, "_TENSOR_PARALLEL_GROUP is not initialized"
    return _TENSOR_PARALLEL_GROUP


def get_tensor_parallel_world_size() -> int:
    return torch.distributed.get_world_size(group=get_tensor_parallel_group())


def get_tensor_parallel_rank() -> int:
    return torch.distributed.get_rank(group=get_tensor_parallel_group())


def get_tensor_parallel_src_rank() -> int:
    global_rank = torch.distributed.get_rank()
    local_world_size = get_tensor_parallel_world_size()
    return (global_rank // local_world_size) * local_world_size


def _reduce(ctx: Any, input_: torch.Tensor) -> torch.Tensor:
    group = get_tensor_parallel_group()
    if ctx:
        ctx.mark_dirty(input_)
    if torch.distributed.get_world_size(group=group) == 1:
        return input_
    torch.distributed.all_reduce(input_, group=group)
    return input_


def _split(input_: torch.Tensor) -> torch.Tensor:
    group = get_tensor_parallel_group()
    if torch.distributed.get_world_size(group=group) == 1:
        return input_
    world_size = torch.distributed.get_world_size(group=group)
    input_list = split_tensor_along_last_dim(input_, world_size)
    rank = torch.distributed.get_rank(group=group)
    output = input_list[rank].contiguous()
    return output


def _gather(input_: torch.Tensor) -> torch.Tensor:
    group = get_tensor_parallel_group()
    if torch.distributed.get_world_size(group=group) == 1:
        return input_
    last_dim = input_.dim() - 1
    rank = torch.distributed.get_rank(group=group)
    world_size = torch.distributed.get_world_size(group=group)
    tensor_list = [torch.empty_like(input_) for _ in range(world_size)]
    tensor_list[rank] = input_
    torch.distributed.all_gather(tensor_list, input_, group=group)
    output = torch.cat(tensor_list, dim=last_dim).contiguous()
    return output


class _CopyToTensorParallelRegion(torch.autograd.Function):
    """Pass the input to the model parallel region."""

    @staticmethod
    def forward(ctx, input_):  # type: ignore
        return input_

    @staticmethod
    def backward(ctx, grad_output):  # type: ignore
        return _reduce(None, grad_output)


class _ReduceFromTensorParallelRegion(torch.autograd.Function):
    """All-redcue the input from the model parallel region."""

    @staticmethod
    def forward(ctx, input_):  # type: ignore
        return _reduce(ctx, input_)

    @staticmethod
    def backward(ctx, grad_output):  # type: ignore
        return grad_output


class _ScatterToTensorParallelRegion(torch.autograd.Function):
    """Split the input and keep only the corresponding chuck to the rank."""

    @staticmethod
    def forward(ctx, input_):  # type: ignore
        return _split(input_)

    @staticmethod
    def backward(ctx, grad_output):  # type: ignore
        return _gather(grad_output)


class _GatherFromTensorParallelRegion(torch.autograd.Function):
    """Gather the input from model parallel region and concatinate."""

    @staticmethod
    def forward(ctx, input_):  # type: ignore
        return _gather(input_)

    @staticmethod
    def backward(ctx, grad_output):  # type: ignore
        return _split(grad_output)


def copy_to_tensor_parallel_region(input_: torch.Tensor) -> torch.Tensor:
    return _CopyToTensorParallelRegion.apply(input_)


def reduce_from_tensor_parallel_region(input_: torch.Tensor) -> torch.Tensor:
    return _ReduceFromTensorParallelRegion.apply(input_)


def scatter_to_tensor_parallel_region(input_: torch.Tensor) -> torch.Tensor:
    return _ScatterToTensorParallelRegion.apply(input_)


def gather_from_tensor_parallel_region(input_: torch.Tensor) -> torch.Tensor:
    return _GatherFromTensorParallelRegion.apply(input_)
