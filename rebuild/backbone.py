import math
from typing import Callable
from dataclasses import dataclass, replace, asdict
from functools import partial

import tensorflow.keras as keras


CONV_INIT = keras.initializers.VarianceScaling(scale=2., mode='fan_out', distribution='untruncated_normal')
DENSE_INIT = keras.initializers.VarianceScaling(scale=1/3, mode='fan_out', distribution='uniform')


@dataclass
class BlockArgs:
    repeat: int
    kernel_size: int
    strides: int
    filters_in: int
    filters_out: int
    expand_ratio: int
    se_ratio: float


@dataclass
class EffnetParams:
    width_coefficient: float
    depth_coefficient: float
    dropout_rate: float
    drop_depth_rate: float


@dataclass
class EffnetHParams:
    bn_momentum: float = .99
    bn_epsilon: float = 1e-3
    depth_divisor: int = 8
    act_fn: Callable = keras.activations.swish


@dataclass
class EffnetAllParams(EffnetHParams, EffnetParams):
    pass


default_blockargs_list = [
    BlockArgs(1, 3, 1, 32, 16, 1, .25),
    BlockArgs(2, 3, 2, 16, 24, 6, .25),
    BlockArgs(2, 5, 2, 24, 40, 6, .25),
    BlockArgs(3, 3, 2, 40, 80, 6, .25),
    BlockArgs(3, 5, 2, 80, 112, 6, .25),
    BlockArgs(4, 5, 2, 112, 192, 6, .25),
    BlockArgs(1, 3, 1, 192, 320, 6, .25),
]

effnet_params_dict = {
    'efficientnet-b0': EffnetParams(1., 1., .2, 1.),
    'efficientnet-b1': EffnetParams(1.,  1.1, .2, .8),
    'efficientnet-b2': EffnetParams(1.1, 1.2, .3, .8),
    'efficientnet-b3': EffnetParams(1.2, 1.4, .3, .8),
    'efficientnet-b4': EffnetParams(1.4, 1.8, .4, .8),
    'efficientnet-b5': EffnetParams(1.6, 2.2, .4, .8),
    'efficientnet-b6': EffnetParams(1.8, 2.6, .5, .8),
    'efficientnet-b7': EffnetParams(2.0, 3.1, .5, .8),
}


def get_effnet_params(config):
    hparams = EffnetHParams()
    if config.backbone_config is not None:
        hparams = replace(hparams, **config.backbone_config)

    params = effnet_params_dict[config.backbone_name]
    all_params = EffnetAllParams(**asdict(hparams), **asdict(params))

    return all_params


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


class Stem(keras.layers.Layer):
    def __init__(self,
                 params: EffnetAllParams,
                 output_ch: int,
                 name: str = 'stem'):
        super().__init__(name=name)

        self.conv = keras.layers.Conv2D(
            filters=output_ch,
            kernel_size=3,
            strides=2,
            padding='same',
            use_bias=False,
            kernel_initializer=CONV_INIT
        )

        self.bn = keras.layers.experimental.SyncBatchNormalization(
            momentum=params.bn_momentum,
            epsilon=params.bn_epsilon
        )

        self.act_fn = params.act_fn

    def call(self, inputs, **kwargs):
        return self.act_fn(self.bn(self.conv(inputs), **kwargs))


class MBConv(keras.layers.Layer):
    def __init__(self,
                 kernel_size: int,
                 strides: int,
                 filters_in: int,
                 filters_out: int,
                 expand_ratio: int,
                 se_ratio: float,
                 params: EffnetAllParams,
                 name: str = None):
        super().__init__(name=name)

        if expand_ratio > 1:
            self.conv_expand = keras.layers.Conv2D(
                filters=filters_in*expand_ratio,
                kernel_size=kernel_size,
                strides=strides,
                padding='same',
                use_bias=False,
                kernel_initializer=CONV_INIT
            )
            self.bn_expand = keras.layers.experimental.SyncBatchNormalization(

            )



class EffNet(keras.Model):
    def __init__(self,
                 params: EffnetAllParams,
                 name: str = None):
        super().__init__(name=name)

        blockargs_list = default_blockargs_list

        round_filters_partial = partial(round_filters,
                                        width_coefficient=params.width_coefficient,
                                        depth_divisor=params.depth_divisor)
        round_repeat_partial = partial(round_repeat, depth_coefficient=params.depth_coefficient)

        self.stem = Stem(params, round_filters_partial(filters=blockargs_list[0].filters_in))
