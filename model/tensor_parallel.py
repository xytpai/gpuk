import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.nn.init as init
import distributed_utils as dutils
from typing import Callable


def _initialize_affine_weight(
    weight: torch.Tensor,
    out_features: int,
    in_features: int,
    per_partition_size: int,
    partition_dim: int,
    init_method: Callable[[torch.Tensor], torch.Tensor],
    stride: int = 1):
    world_size = dutils.get_tensor_parallel_world_size()
    if world_size == 1:
        init_method(weight)
        return
    # Initialize master weight
    master_weight = torch.empty(out_features, in_features, dtype=weight.dtype, requires_grad=False)
    init_method(master_weight)
    # Split and copy
    per_partition_per_stride_size = dutils.divide_and_check_no_remainder(per_partition_size, stride)
    weight_list = torch.split(master_weight, per_partition_per_stride_size, dim=partition_dim)
    rank = dutils.get_tensor_parallel_rank()
    my_weight_list = weight_list[rank::world_size]
    with torch.no_grad():
        torch.cat(my_weight_list, dim=partition_dim, out=weight)


class ColumnParallelLinear(nn.Module):
    """Linear layer with column parallelism.
    The linear layer is defined as Y = XA + b. A is parallelized along
    its second dimension as A = [A_1, ..., A_p].
    """
    def __init__(
        self,
        in_features: int,
        out_features: int,
        dtype: torch.dtype,
        bias: bool = True,
        gather_output: bool = True,
        device: str = 'cuda:0'):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.gather_output = gather_output
        world_size = dutils.get_tensor_parallel_world_size()
        self.output_size_per_partition = dutils.divide_and_check_no_remainder(out_features, world_size)
        self.weight = nn.Parameter(torch.empty(self.output_size_per_partition, self.in_features, dtype=dtype, device=device))
        if bias:
            self.bias = nn.Parameter(torch.zeros(self.output_size_per_partition, dtype=dtype, device=device))
        else:
            self.register_parameter("bias", None)
        _initialize_affine_weight(
            self.weight,
            self.out_features,
            self.in_features,
            self.output_size_per_partition,
            0, init.xavier_normal_, 1)

    def forward(self, input_):
        input_parallel = dutils.copy_to_tensor_parallel_region(input_)
        output_parallel = F.linear(input_parallel, self.weight, self.bias)
        if self.gather_output:
            output = dutils.gather_from_tensor_parallel_region(output_parallel)
        else:
            output = output_parallel
        return output


class RowParallelLinear(nn.Module):
    """Linear layer with row parallelism.
    The linear layer is defined as Y = XA + b. A is parallelized along
    its first dimension and X along its second dimension as:
               -   -
              | A_1 |
              | .   |
          A = | .   |        X = [X_1, ..., X_p]
              | .   |
              | A_p |
               -   -
    """
    def __init__(
        self,
        in_features: int,
        out_features: int,
        dtype: torch.dtype,
        bias: bool = True,
        input_is_parallel: bool = False,
        device: str = 'cuda:0'):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.input_is_parallel = input_is_parallel
        world_size = dutils.get_tensor_parallel_world_size()
        self.input_size_per_partition = dutils.divide_and_check_no_remainder(in_features, world_size)
        self.weight = nn.Parameter(torch.empty(self.out_features, self.input_size_per_partition, dtype=dtype, device=device))
        if bias:
            self.bias = nn.Parameter(torch.zeros(self.out_features, dtype=dtype, device=device))
        else:
            self.register_parameter("bias", None)
        _initialize_affine_weight(
            self.weight,
            self.out_features,
            self.in_features,
            self.input_size_per_partition,
            1, init.xavier_normal_, 1)

    def forward(self, input_):
        if self.input_is_parallel:
            input_parallel = input_
        else:
            input_parallel = dutils.scatter_to_tensor_parallel_region(input_)
        output_parallel = F.linear(input_parallel, self.weight)
        output_ = dutils.reduce_from_tensor_parallel_region(output_parallel)
        if self.bias is not None:
            output = output_ + self.bias
        else:
            output = output_
        return output
