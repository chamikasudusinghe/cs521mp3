import jax
import jax.numpy as jnp
from flax import linen as nn
from typing import Sequence

class Int8Conv(nn.Module):
    features: int
    kernel_size: Sequence[int]
    strides: Sequence[int] = (1, 1)
    padding: str = 'SAME'
    use_bias: bool = False

    @nn.compact
    def __call__(self, x, weight, scale):
        weight = weight.astype(jnp.float32) * scale
        return nn.Conv(
            features=self.features,
            kernel_size=self.kernel_size,
            strides=self.strides,
            padding=self.padding,
            use_bias=self.use_bias,
            kernel_init=lambda *_: weight
        )(x)

class BasicBlock(nn.Module):
    channels: int
    stride: int = 1
    quantized: bool = False

    @nn.compact
    def __call__(self, x, params=None):
        residual = x

        def get_wb(name):
            return params[name]["kernel"]["int8_val"], params[name]["kernel"]["scale"]

        if self.quantized:
            w1, s1 = get_wb("conv1")
            x = Int8Conv(self.channels, (3, 3), (self.stride, self.stride), use_bias=False)(x, w1, s1)
        else:
            x = nn.Conv(self.channels, (3, 3), self.stride, padding='SAME', use_bias=False)(x)

        x = nn.BatchNorm(use_running_average=True)(x)
        x = nn.relu(x)

        if self.quantized:
            w2, s2 = get_wb("conv2")
            x = Int8Conv(self.channels, (3, 3), (1, 1), use_bias=False)(x, w2, s2)
        else:
            x = nn.Conv(self.channels, (3, 3), 1, padding='SAME', use_bias=False)(x)

        x = nn.BatchNorm(use_running_average=True)(x)

        if residual.shape != x.shape:
            if self.quantized:
                ws, ss = get_wb("conv_proj")
                residual = Int8Conv(self.channels, (1, 1), (self.stride, self.stride), use_bias=False)(residual, ws, ss)
            else:
                residual = nn.Conv(self.channels, (1, 1), self.stride, use_bias=False)(residual)
            residual = nn.BatchNorm(use_running_average=True)(residual)

        return nn.relu(x + residual)

class ResNet(nn.Module):
    stage_sizes: Sequence[int]
    num_classes: int = 10
    quantized: bool = False

    @nn.compact
    def __call__(self, x, params=None):
        if self.quantized:
            w, s = params["conv1"]["kernel"]["int8_val"], params["conv1"]["kernel"]["scale"]
            x = Int8Conv(64, (3, 3), (1, 1), use_bias=False)(x, w, s)
        else:
            x = nn.Conv(64, (3, 3), (1, 1), padding='SAME', use_bias=False)(x)

        x = nn.BatchNorm(use_running_average=True)(x)
        x = nn.relu(x)

        for i, block_size in enumerate(self.stage_sizes):
            for j in range(block_size):
                stride = 2 if i > 0 and j == 0 else 1
                block_params = params[f"block_{i}_{j}"] if params else None
                x = BasicBlock(64 * 2**i, stride=stride, quantized=self.quantized)(x, block_params)

        x = jnp.mean(x, axis=(1, 2))  # global average pooling
        if self.quantized:
            w, s = params["dense"]["kernel"]["int8_val"], params["dense"]["kernel"]["scale"]
            w = w.astype(jnp.float32) * s
            x = x @ w
            x += params["dense"]["bias"]
        else:
            x = nn.Dense(self.num_classes)(x)
        return x

def ResNet18(quantized=False):
    return ResNet(stage_sizes=[2, 2, 2, 2], quantized=quantized)
