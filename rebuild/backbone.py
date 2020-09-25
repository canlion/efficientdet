import math
from typing import Callable, Tuple
from dataclasses import dataclass, replace, asdict
from functools import partial
from itertools import count

import tensorflow as tf
import tensorflow.keras as keras


CONV_INIT = keras.initializers.VarianceScaling(scale=2., mode='fan_out', distribution='untruncated_normal')
DENSE_INIT = keras.initializers.VarianceScaling(scale=1/3, mode='fan_out', distribution='uniform')


@dataclass(frozen=True)
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
    survival_prob: float


@dataclass
class EffnetHParams:
    input_size: Tuple[int, int] = (224, 224)
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
    BlockArgs(3, 5, 1, 80, 112, 6, .25),
    BlockArgs(4, 5, 2, 112, 192, 6, .25),
    BlockArgs(1, 3, 1, 192, 320, 6, .25),
]

effnet_params_dict = {
    'efficientnet-b0': EffnetParams(1., 1., .2, 0.),
    'efficientnet-b1': EffnetParams(1.,  1.1, .2, .8),
    'efficientnet-b2': EffnetParams(1.1, 1.2, .3, .8),
    'efficientnet-b3': EffnetParams(1.2, 1.4, .3, .8),
    'efficientnet-b4': EffnetParams(1.4, 1.8, .4, .8),
    'efficientnet-b5': EffnetParams(1.6, 2.2, .4, .8),
    'efficientnet-b6': EffnetParams(1.8, 2.6, .5, .8),
    'efficientnet-b7': EffnetParams(2.0, 3.1, .5, .8),
}


def get_effnet_params(config):
    """Update and merge parameters."""
    hparams = EffnetHParams()
    if config.backbone_config:
        hparams = replace(hparams, **config.backbone_config)
    hparams.act_fn = partial(keras.layers.Activation, activation=hparams.act_fn)

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


def build_se(x,
             filters_reduce: int,
             filters_expand: int,
             name_prefix: str = None):
    inputs = x
    x = tf.reduce_mean(x, [1, 2], keepdims=True)
    x = keras.layers.Conv2D(
        filters=filters_reduce,
        kernel_size=1,
        strides=1,
        padding='same',
        use_bias=True,
        kernel_initializer=CONV_INIT,
        name=name_prefix+'_se_reduce')(x)
    # x = keras.layers.ReLU(name=name_prefix+'_se_relu')(x)
    x = keras.layers.Activation(tf.keras.activations.swish)(x)

    x = keras.layers.Conv2D(
        filters=filters_expand,
        kernel_size=1,
        strides=1,
        padding='same',
        use_bias=True,
        kernel_initializer=CONV_INIT,
        name=name_prefix+'_se_expand')(x)
    x = keras.layers.Activation('sigmoid', name=name_prefix+'_se_sigmoid')(x)

    return x * inputs


def build_stem(x,
               params: EffnetAllParams,
               output_ch: int,
               name_prefix: str = None):
    x = keras.layers.Conv2D(
        filters=output_ch,
        kernel_size=3,
        strides=2,
        padding='same',
        use_bias=False,
        kernel_initializer=CONV_INIT,
        name=name_prefix+'_conv')(x)
    x = keras.layers.experimental.SyncBatchNormalization(
        momentum=params.bn_momentum,
        epsilon=params.bn_epsilon,
        name=name_prefix+'_bn')(x)
    x = params.act_fn(name=name_prefix+'_activation')(x)

    return x


def build_mbconv_block(x,
                       kernel_size: int,
                       strides: int,
                       filters_in: int,
                       filters_out: int,
                       expand_ratio: int,
                       se_ratio: float,
                       survival_prob: float,
                       params: EffnetAllParams,
                       name_prefix: str = None):
    batch_norm = partial(keras.layers.experimental.SyncBatchNormalization,
                         momentum=params.bn_momentum,
                         epsilon=params.bn_epsilon)
    inputs = x

    filters_expand = filters_in * expand_ratio
    if expand_ratio > 1:
        x = keras.layers.Conv2D(
            filters=filters_expand,
            kernel_size=1,
            strides=1,
            padding='same',
            use_bias=False,
            kernel_initializer=CONV_INIT,
            name=name_prefix+'_expand_conv')(x)
        x = batch_norm(name=name_prefix+'_expand_bn')(x)
        x = params.act_fn(name=name_prefix+'_expand_activation')(x)

    x = keras.layers.DepthwiseConv2D(
        kernel_size=kernel_size,
        strides=strides,
        padding='same',
        use_bias=False,
        depthwise_initializer=CONV_INIT,
        name=name_prefix+'_dwconv')(x)
    x = batch_norm(name=name_prefix+'_bn')(x)
    x = params.act_fn(name=name_prefix+'_activation')(x)

    filters_reduce = max(1, int(filters_in * se_ratio))
    x = build_se(x,
                 filters_reduce=filters_reduce,
                 filters_expand=filters_expand,
                 name_prefix=name_prefix)

    x = keras.layers.Conv2D(
        filters=filters_out,
        kernel_size=1,
        strides=1,
        padding='same',
        use_bias=False,
        kernel_initializer=CONV_INIT,
        name=name_prefix+'_project_conv')(x)
    x = batch_norm(name=name_prefix+'_project_bn')(x)
    # x = params.act_fn(name=name_prefix+'_project_activation')(x)

    if (strides == 1) and (filters_in == filters_out):
        if survival_prob:
            x = keras.layers.Dropout(rate=1-survival_prob,
                                     noise_shape=[None, 1, 1, 1],
                                     name=name_prefix+'_depth_drop')(x)
        x = tf.add(x, inputs, name=name_prefix+'_identity_skip')
    return x


def build_head(x,
               params: EffnetAllParams,
               num_classes: int = 1000,
               name_prefix: str = None):
    x = keras.layers.GlobalAveragePooling2D(name=name_prefix+'avg_pool')(x)
    x = keras.layers.Dropout(
        rate=params.dropout_rate,
        name=name_prefix+'top_dropout')(x)
    x = keras.layers.Dense(
        units=num_classes,
        activation='softmax',
        kernel_initializer=DENSE_INIT,
        name=name_prefix+'probs')(x)

    return x


def build_effnet(params: EffnetAllParams,
                 feature_only: bool,
                 name: str = None):
    round_filters_partial = partial(round_filters,
                                    width_coefficient=params.width_coefficient,
                                    depth_divisor=params.depth_divisor)
    round_repeat_partial = partial(round_repeat,
                                   depth_coefficient=params.depth_coefficient)

    blockargs_list = [
        replace(args,
                repeat=round_repeat_partial(args.repeat),
                filters_in=round_filters_partial(args.filters_in),
                filters_out=round_filters_partial(args.filters_out))
        for args in default_blockargs_list
    ]

    x = keras.layers.Input(shape=(*params.input_size, 3))
    inputs = x

    x = build_stem(x, params, blockargs_list[0].filters_in, 'stem')

    feature_list = list()
    total_blocks = sum(args.repeat for args in blockargs_list)
    block_cnt = count(0)
    survival_prob = params.survival_prob
    depth_drop_rate_unit = (1 - survival_prob) / total_blocks
    for idx_args, args in enumerate(blockargs_list):
        for idx in range(args.repeat):
            block_no = next(block_cnt)
            if params.survival_prob:
                survival_prob = 1 - depth_drop_rate_unit * block_no
            x = build_mbconv_block(
                x=x,
                kernel_size=args.kernel_size,
                strides=args.strides,
                filters_in=args.filters_in,
                filters_out=args.filters_out,
                expand_ratio=args.expand_ratio,
                se_ratio=args.se_ratio,
                survival_prob=survival_prob,
                params=params,
                name_prefix='block{}{}'.format(idx_args+1, chr(idx+97))
            )
            if idx == 0:
                args = replace(args,
                               strides=1,
                               filters_in=args.filters_out)
        if (block_no == total_blocks - 1) or (blockargs_list[idx_args+1].strides > 1):
            feature_list.append(x)

    if feature_only:
        model = keras.Model(inputs=inputs, outputs=feature_list, name=name)
    else:
        x = keras.layers.Conv2D(
            filters=round_filters_partial(1280),
            kernel_size=1,
            strides=1,
            padding='same',
            use_bias=False,
            kernel_initializer=CONV_INIT,
            name='top_conv')(x)
        x = keras.layers.experimental.SyncBatchNormalization(
            momentum=params.bn_momentum,
            epsilon=params.bn_epsilon,
            name='top_bn')(x)
        x = params.act_fn(name='top_activation')(x)
        x = build_head(x, params, name_prefix='')
        model = keras.Model(inputs=inputs, outputs=x, name=name)

    return model
