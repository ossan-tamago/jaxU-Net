from dataclasses import field
from typing import List, Tuple, Union

import jax
import jax.numpy as jnp
from flax import linen as nn


class ResidualBlock(nn.Module):
    features: int

    @nn.compact
    def __call__(self, x, train: bool):
        input_features = x.shape[3]
        if input_features == self.features:
            residual = x
        else:
            residual = nn.Conv(self.features, kernel_size=(3, 3))(x)

        x = nn.BatchNorm(use_running_average=not train, use_bias=False, use_scale=False)(x)
        x = nn.Conv(self.features, (3, 3), 1, 1)(x)
        x = nn.swish(x)
        x = nn.Conv(self.features, (3, 3), 1, 1)(x)
        x += residual
        return x


class DownBlock(nn.Module):
    features: int
    blocks: int

    @nn.compact
    def __call__(self, x, train: bool) -> Tuple:
        skips = []
        for _ in range(self.blocks):
            x = ResidualBlock(self.features)(x, train=train)
            skips.append(x)
        x = nn.avg_pool(x, window_shape=(2, 2), strides=(2, 2))
        return x, skips


def upsample2d(x, scale: Union[int, Tuple[int, int]], method: str = 'bilinear'):
    b, h, w, c = x.shape

    if isinstance(scale, int):
        h_out, w_out = scale * h, scale * w
    elif len(scale) == 2:
        h_out, w_out = scale[0] * h, scale[1] * w
    else:
        raise ValueError('scale argument should be either int or Tuple[int, int]')

    return jax.image.resize(x, shape=(b, h_out, w_out, c), method=method)


class UpBlock(nn.Module):
    features: int
    blocks: int

    @nn.compact
    def __call__(self, x, skips: List, train: bool):
        x = upsample2d(x, scale=2, method='bilinear')
        for _ in range(self.blocks):
            x = jnp.concatenate([x, skips.pop()], axis=-1)
            x = ResidualBlock(self.features)(x, train=train)
        return x


class UNet(nn.Module):
    feature_stages: List[int] = field(default_factory=lambda: [32, 64, 96, 128])
    blocks: int = 2
    classes: int = 2

    @nn.compact
    def __call__(self, x, train: bool):
        skip_stages = []
        for features in self.feature_stages[:-1]:
            x, skips = DownBlock(features, self.blocks)(x, train=train)
            skip_stages.append(skips)

        for _ in range(self.blocks):
            x = ResidualBlock(self.feature_stages[-1])(x, train=train)

        for features in reversed(self.feature_stages[:-1]):
            skips = skip_stages.pop()
            x = UpBlock(features, self.blocks)(x, skips, train=train)

        x = nn.Conv(self.classes, (1, 1), kernel_init=nn.initializers.zeros)(x)

        return x
