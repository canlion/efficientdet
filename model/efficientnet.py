from collections import namedtuple
import math
from functools import partial

import tensorflow as tf
import tensorflow.keras as keras


BlockArgs = namedtuple('BlockArgs', ['kernel_size', 'strides', 'filters_in', 'filters_out',
                                     'expand_ratio', 'se_ratio', 'repeat'])

CONV_INITIALIZER = keras.initializers.VarianceScaling(scale=2., mode='fan_out', distribution='untruncated_normal')


def conv_3x3(filters, strides=1, activation=None, use_bias=False, name=None):
    return keras.layers.Conv2D(filters=filters, kernel_size=3, strides=strides, padding='same', activation=activation,
                               use_bias=use_bias, kernel_initializer=CONV_INITIALIZER, name=name)


def conv_1x1(filters, activation=None, use_bias=False, name=None):
    return keras.layers.Conv2D(filters=filters, kernel_size=1, strides=1, padding='same', activation=activation,
                               use_bias=use_bias, kernel_initializer=CONV_INITIALIZER, name=name)


def conv_depthwise(kernel_size, strides, activation=None, use_bias=False, name=None):
    return keras.layers.DepthwiseConv2D(kernel_size=kernel_size, strides=strides, padding='same', activation=activation,
                                        use_bias=use_bias, depthwise_initializer=CONV_INITIALIZER, name=name)


def round_filters(filters, width_coefficient, depth_divisor=8):
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


def round_repeat(repeat, depth_coefficient):
    repeat *= depth_coefficient
    return int(math.ceil(repeat))


class MBConvBlock:
    def __init__(self, block_args: BlockArgs, depth_drop_rate, freeze_bn=True, block_name=None):
        self.block_args = block_args
        self.depth_drop_rate = depth_drop_rate
        self.block_name = block_name
        self.activation = keras.activations.swish
        self.freeze_bn = freeze_bn

        self._build()

    def _build(self):
        def name(x):
            return self.block_name + x

        args = self.block_args
        filters = args.filters_in * args.expand_ratio

        # Expansion
        if args.expand_ratio > 1:
            self.expand_conv = conv_1x1(filters, name=name('expand_conv'))
            self.expand_bn = keras.layers.BatchNormalization(name=name('expand_bn'))
            self.expand_activation = keras.layers.Activation(self.activation, name=name('expand_activation'))

        # Filtering
        self.dwconv = conv_depthwise(args.kernel_size, args.strides, name=name('dwconv'))
        self.bn = keras.layers.BatchNormalization(name=name('bn'))
        self.activation = keras.layers.Activation(self.activation, name=name('activation'))

        # SE
        filters_se = max(1, int(args.filters_in * args.se_ratio))
        self.se_squeeze = keras.layers.GlobalAvgPool2D(name=name('se_squeeze'))
        self.se_reshape = keras.layers.Reshape((1, 1, filters), name=name('se_reshape'))
        self.se_reduce = conv_1x1(filters_se, activation=self.activation, use_bias=True, name=name('se_reduce'))
        self.se_expand = conv_1x1(filters, activation='sigmoid', use_bias=True, name=name('se_expand'))
        self.se_excite = keras.layers.Multiply(name=name('se_excite'))

        # Projection
        self.projection_conv = conv_1x1(args.filters_out, name=name('project_conv'))
        self.projection_bn = keras.layers.BatchNormalization(name=name('project_bn'))

        # Stochastic depth
        if args.strides == 1 and args.filters_in == args.filters_out:
            # Randomly set examples of mini-batch to 0. Only skip-connection is maintained.
            self.depth_drop = keras.layers.Dropout(self.depth_drop_rate, noise_shape=[None, 1, 1, 1], name=name('drop'))
            self.add = keras.layers.Add(name=name('add'))

        if self.freeze_bn:
            for bn in [self.expand_bn, self.bn, self.projection_bn]:
                bn.trainable = False

    def _se(self, inputs):
        outputs_se = inputs
        outputs_squeeze = self.se_reshape(self.se_squeeze(outputs_se))
        outputs_reduce = self.se_reduce(outputs_squeeze)
        outputs_expand = self.se_expand(outputs_reduce)
        return self.se_excite([outputs_expand, inputs])

    def __call__(self, inputs):
        args = self.block_args
        outputs = inputs

        # Expansion
        if args.expand_ratio > 1:
            outputs = self.expand_activation(self.expand_bn(self.expand_conv(outputs)))
        # Filtering
        outputs = self.activation(self.bn(self.dwconv(outputs)))
        # SE
        outputs = self._se(outputs)
        # Projection
        outputs = self.projection_bn(self.projection_conv(outputs))
        # stochastic depth & skip connection
        if args.strides == 1 and args.filters_in == args.filters_out:
            outputs = self.depth_drop(outputs)
            outputs = self.add([outputs, inputs])

        return outputs


NetArgs = namedtuple('EfficientnetArgs', ['width_coefficient', 'depth_coefficient'])


class EfficientNetBase:
    # EfficientNet Default Arguments (B0)
    B0_BLOCK_ARGS = [
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
        'B0': NetArgs(width_coefficient=1., depth_coefficient=1.),
        'B1': NetArgs(width_coefficient=1., depth_coefficient=1.1),
        'B2': NetArgs(width_coefficient=1.1, depth_coefficient=1.2),
        'B3': NetArgs(width_coefficient=1.2, depth_coefficient=1.4),
        'B4': NetArgs(width_coefficient=1.4, depth_coefficient=1.8),
        'B5': NetArgs(width_coefficient=1.6, depth_coefficient=2.2),
        'B6': NetArgs(width_coefficient=1.8, depth_coefficient=2.6),
        'B7': NetArgs(width_coefficient=2.0, depth_coefficient=3.1),
    }

    def __init__(self,
                 B,
                 depth_divisor=8,
                 drop_rate=.2,
                 freeze_bn=False,
                 name='efficientnet'):

        net_args = self.EFFICIENTNET_ARGS[B]

        self.round_filters = partial(round_filters,
                                     width_coefficient=net_args.width_coefficient,
                                     depth_divisor=depth_divisor)
        self.round_repeat = partial(round_repeat, depth_coefficient=net_args.depth_coefficient)
        self.name = name

        self.depth_drop_rate = drop_rate
        self.freeze_bn = freeze_bn
        self.activation = keras.activations.swish

        self._build()

    def _build(self):
        # Stem
        self.stem_conv = conv_3x3(filters=self.round_filters(32), strides=2, name='stem_conv')
        self.stem_bn = keras.layers.BatchNormalization(name='stem_bn')
        self.stem_activation = keras.layers.Activation(self.activation)

        # MBConvBlocks
        block_args = self.B0_BLOCK_ARGS
        self.block_list = list()
        self.feature_name = list()
        total_blocks_num = sum([args.repeat for args in block_args])
        blocks_cnt = 0
        drop_rate_unit = self.depth_drop_rate / total_blocks_num

        for outer_order, args in enumerate(block_args):
            args = args._replace(filters_in=self.round_filters(args.filters_in),
                                 filters_out=self.round_filters(args.filters_out),
                                 repeat=self.round_repeat(args.repeat))
            for inner_order in range(args.repeat):
                self.block_list.append(
                    MBConvBlock(
                        block_args=args,
                        depth_drop_rate=blocks_cnt * drop_rate_unit,
                        freeze_bn=self.freeze_bn,
                        block_name='block{}{}_'.format(outer_order+1, chr(inner_order+97))))
                blocks_cnt += 1
                # Modify strides and filters_in after build the first block.
                # Changes in filters and feature resolution are made only on the first block.
                if inner_order == 0:
                    args = args._replace(strides=1, filters_in=args.filters_out)

            if (outer_order == len(block_args) - 1) or (block_args[outer_order+1].strides == 2):
                self.feature_name.append(self.block_list[-1].block_name)

    def outputs(self, inputs):
        feature_list = list()
        # inputs = keras.layers.Input((*self.input_resolution, 3))

        # Stem
        outputs = self.stem_activation(self.stem_bn(self.stem_conv(inputs)))

        # MBConvBlocks
        for block in self.block_list:
            outputs = block(outputs)
            if block.block_name in self.feature_name:
                feature_list.append(outputs)

        return feature_list


if __name__ == '__main__':
    inputs = keras.layers.Input((768, 768, 3))
    effnet_features = EfficientNetBase('B2').outputs(inputs)

    model = keras.Model(inputs=inputs, outputs=effnet_features)
    print(model.layers[-3].weights)
    model.load_weights('efficientnet_weights/efficientnet-b2_weights_tf_dim_ordering_tf_kernels_autoaugment_notop.h5',
                       by_name=True)
    print(model.layers[-3].weights)
    model.summary()
