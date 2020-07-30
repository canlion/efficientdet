from functools import partial

import tensorflow as tf
import tensorflow.keras as keras

from .layers import ConvBlock, FastFusion


initializer = keras.initializers.VarianceScaling(scale=2., mode='fan_out', distribution='untruncated_normal')


class BiFPNLayer(keras.layers.Layer):
    def __init__(self, width, levels, activation=keras.activations.swish, epsilon=1e-4, name=None):
        super(BiFPNLayer, self).__init__(name=name)

        self.levels = levels
        self.epsilon = epsilon
        conv_cfg = {
            'depthwise_initializer': initializer,
            'pointwise_initializer': initializer,
            'depthwise_regularizer': keras.regularizers.l2(1e-4),
            'pointwise_regularizer': keras.regularizers.l2(1e-4),
        }
        bn_cfg = {
            'momentum': .997,
            'epsilon': 1e-4
        }
        sep_conv = partial(ConvBlock,
                           conv_type='separable',
                           conv_cfg=conv_cfg,
                           apply_bn=True,
                           bn_cfg=bn_cfg,
                           activation=activation)

        self.td_convs = [sep_conv(filters=width, kernel_size=3, strides=1) for _ in range(levels-1)]
        self.td_fusion = [FastFusion() for _ in range(levels-1)]
        self.bu_convs = [sep_conv(filters=width, kernel_size=3, strides=1) for _ in range(levels-1)]
        self.bu_fusion = [FastFusion() for _ in range(levels-1)]
        self.upscale = keras.layers.UpSampling2D(size=(2, 2))
        self.downscale = keras.layers.MaxPool2D(pool_size=(2, 2), padding='same')


    def call(self, inputs_list, **kwargs):
        td_list = inputs_list[:1]
        for i in range(self.levels-1):
            feature_up = self.upscale(td_list[i])
            feature_fusion = self.td_fusion[i]([feature_up, inputs_list[i+1]], **kwargs)
            td_list.append(self.td_convs[i](feature_fusion, **kwargs))

        bu_list = td_list[-1:]
        for j in range(self.levels-2):
            idx = self.levels-2-j
            feature_down = self.downscale(bu_list[j])
            feature_fusion = self.bu_fusion[j]([feature_down, td_list[idx], inputs_list[idx]], **kwargs)
            bu_list.append(self.bu_convs[j](feature_fusion, **kwargs))

        feature_down = self.downscale(bu_list[-1])
        feature_fusion = self.bu_fusion[-1]([feature_down, td_list[0]], **kwargs)
        bu_list.append(self.bu_convs[-1](feature_fusion, **kwargs))

        return bu_list[::-1]


class BiFPN(keras.layers.Layer):
    def __init__(self, width, repeat, start_P_level=3, num_levels=5, activation=keras.activations.swish):
        super(BiFPN, self).__init__()
        self.width = width
        self.start_P_level = start_P_level
        self.num_levels = num_levels
        self.bifpn_list = [BiFPNLayer(width, num_levels, activation) for _ in range(repeat)]
        self.activation = activation
        self.conv_list = list()
        self.additional_conv_list = list()

    def build(self, feature_list):
        num_feature = len(feature_list)

        sep_conv_cfg = {
            'depthwise_initializer': initializer,
            'pointwise_initializer': initializer,
            'depthwise_regularizer': keras.regularizers.l2(1e-4),
            'pointwise_regularizer': keras.regularizers.l2(1e-4),
        }
        additional_conv_cfg = {
            'kernel_initializer': initializer,
        }
        bn_cfg = {
            'momentum': .997,
            'epsilon': 1e-4
        }

        self.conv_list = [
            ConvBlock(self.width, 3, 1, 'separable', sep_conv_cfg, bn_cfg=bn_cfg, activation=self.activation)
            for _ in range(self.num_levels)
        ]

        for _ in range(self.num_levels - (num_feature - self.start_P_level + 1)):
            self.additional_conv_list.append(
                ConvBlock(self.width, 3, 2, conv_cfg=additional_conv_cfg, bn_cfg=bn_cfg, activation=self.activation))

        super(BiFPN, self).build(feature_list)

    def call(self, inputs, **kwargs):
        inputs = inputs[self.start_P_level-1:]
        for i in range(self.num_levels - len(inputs)):
            inputs.append(
                self.additional_conv_list[i](inputs[-1], **kwargs)
            )

        for i in range(len(inputs)):
            inputs[i] = self.conv_list[i](inputs[i], **kwargs)

        feature_list = inputs[::-1]
        for bifpn in self.bifpn_list:
            feature_list = bifpn(feature_list, **kwargs)

        return feature_list


if __name__ == '__main__':
    import os
    import numpy as np
    os.environ['CUDA_VISIBLE_DEVICES'] = '-1'

    input_list = [
        np.random.normal(size=(4, 256, 256, 4)),
        np.random.normal(size=(4, 128, 128, 4)),
        np.random.normal(size=(4, 64, 64, 4)),
        np.random.normal(size=(4, 32, 32, 4)),
        np.random.normal(size=(4, 16, 16, 4)),
    ]

    bifpn = BiFPN(width=64, repeat=3, start_P_level=3, num_levels=5)
    result = bifpn(input_list, training=False)
    print([feature.shape for feature in result])
