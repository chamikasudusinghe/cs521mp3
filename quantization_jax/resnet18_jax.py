import jax
import jax.numpy as jnp
from flax import linen as nn
from typing import Sequence

class BasicBlock(nn.Module):
    channels: int
    stride: int = 1

    @nn.compact
    def __call__(self, x):
        residual = x
        x = nn.Conv(self.channels, (3, 3), self.stride, padding='SAME', use_bias=False)(x)
        x = nn.BatchNorm(use_running_average=True)(x)
        x = nn.relu(x)
        x = nn.Conv(self.channels, (3, 3), 1, padding='SAME', use_bias=False)(x)
        x = nn.BatchNorm(use_running_average=True)(x)

        if residual.shape != x.shape:
            residual = nn.Conv(self.channels, (1, 1), self.stride, use_bias=False)(residual)
            residual = nn.BatchNorm(use_running_average=True)(residual)

        return nn.relu(x + residual)

class ResNet(nn.Module):
    stage_sizes: Sequence[int]
    num_classes: int = 10

    @nn.compact
    def __call__(self, x):
        x = nn.Conv(64, (3, 3), strides=(1, 1), padding='SAME', use_bias=False)(x)
        x = nn.BatchNorm(use_running_average=True)(x)
        x = nn.relu(x)

        for i, block_size in enumerate(self.stage_sizes):
            for j in range(block_size):
                stride = 2 if i > 0 and j == 0 else 1
                x = BasicBlock(channels=64 * 2**i, stride=stride)(x)

        x = jnp.mean(x, axis=(1, 2))  # global average pooling
        x = nn.Dense(self.num_classes)(x)
        return x

def ResNet18():
    return ResNet(stage_sizes=[2, 2, 2, 2])