import math
from typing import NamedTuple
from functools import partial

import tensorflow.keras as keras


CONV_INIT = keras.initializers.VarianceScaling(scale=2., mode='fan_out', distribution='untruncated_normal')
DENSE_INIT = keras.initializers.VarianceScaling(scale=1/3, mode='fan_out', distribution='uniform')


class BlockArgs(NamedTuple):
    kernel_size: int
    strides: int
    filters_in: int
    filters_out: int
    expand_ratio: int
    se_ratio: float
    repeat: int


class NetArgs(NamedTuple):
    width_coefficient: float
    depth_coefficient: float
    dropout_rate: float


class EfficientnetParams(NamedTuple):
    bn_momentum: float
    bn_epsilon: float
    survival_prob: float
    depth_divisor: int


def get_effnet_params(config):
    params = EfficientnetParams(
        bn_momentum=.99,
        bn_epsilon=1e-3,
        survival_prob=0. if config.backbone_name.endswith('b0') else .8,
        depth_divisor=8,
    )
    if config.backbone_args:
        params = params._replace(**config.backbone_args)
    return params


def round_filters(filters: int,
                  width_coefficient: float,
                  depth_divisor: int = 8):
    """Rounding 'filters * width_coefficient' to nearest multiple of 'depth_divisor'.

    e.g. 'filters * width_coefficient'(-) : 11 or 13, 'depth_divisor'(*) : 8
    filters * width_coefficient : 11 |--------|---**** |        | : (11 + 8/2) // 8 * 8 = 8
    filters * width_coefficient : 13 |--------|-----***|*       | : (13 + 8/2) // 8 * 8 = 16
    """
    new_filters = filters * width_coefficient
    new_filters = int(new_filters + depth_divisor / 2) // depth_divisor * depth_divisor
    new_filters = max(new_filters, depth_divisor)
    if new_filters < .9 * filters:  # It should be more than the original 90%.
        new_filters += depth_divisor
    return new_filters


def round_repeat(repeat: int,
                 depth_coefficient: float):
    """Ceiling number of repeat of blocks."""
    repeat *= depth_coefficient
    return int(math.ceil(repeat))


class Efficientnet(keras.Model):
    # EfficientNet Default Configuration - (B0)
    DEFAULT_BLOCK_ARGS = [
        BlockArgs(kernel_size=3, strides=1, filters_in=32, filters_out=16,
                  expand_ratio=1, se_ratio=.25, repeat=1),
        BlockArgs(kernel_size=3, strides=2, filters_in=16, filters_out=24,
                  expand_ratio=6, se_ratio=.25, repeat=2),
        BlockArgs(kernel_size=5, strides=2, filters_in=24, filters_out=40,
                  expand_ratio=6, se_ratio=.25, repeat=2),
        BlockArgs(kernel_size=3, strides=2, filters_in=40, filters_out=80,
                  expand_ratio=6, se_ratio=.25, repeat=3),
        BlockArgs(kernel_size=5, strides=1, filters_in=80, filters_out=112,
                  expand_ratio=6, se_ratio=.25, repeat=3),
        BlockArgs(kernel_size=5, strides=2, filters_in=112, filters_out=192,
                  expand_ratio=6, se_ratio=.25, repeat=4),
        BlockArgs(kernel_size=3, strides=1, filters_in=192, filters_out=320,
                  expand_ratio=6, se_ratio=.25, repeat=1)
    ]

    EFFICIENTNET_ARGS = {
        'efficientnet-b0': NetArgs(width_coefficient=1., depth_coefficient=1., dropout_rate=.2),
        'efficientnet-b1': NetArgs(width_coefficient=1., depth_coefficient=1.1, dropout_rate=.2),
        'efficientnet-b2': NetArgs(width_coefficient=1.1, depth_coefficient=1.2, dropout_rate=.3),
        'efficientnet-b3': NetArgs(width_coefficient=1.2, depth_coefficient=1.4, dropout_rate=.3),
        'efficientnet-b4': NetArgs(width_coefficient=1.4, depth_coefficient=1.8, dropout_rate=.4),
        'efficientnet-b5': NetArgs(width_coefficient=1.6, depth_coefficient=2.2, dropout_rate=.4),
        'efficientnet-b6': NetArgs(width_coefficient=1.8, depth_coefficient=2.6, dropout_rate=.5),
        'efficientnet-b7': NetArgs(width_coefficient=2.0, depth_coefficient=3.1, dropout_rate=.5),
    }

    def __init__(self, config):
        super(Efficientnet, self).__init__()
        effnet_params = get_effnet_params(config)
        effnet_args = self.EFFICIENTNET_ARGS[config.backbone_name]

        self.round_filters = partial(round_filters,
                                     width_coefficient=effnet_args.width_coefficient,
                                     depth_divisor=effnet_params.depth_divisor,)
        self.round_repeat = partial(round_repeat,
                                    depth_coefficient=effnet_args.depth_coefficient)

        self.dropout_rate = effnet_args.dropout_rate
