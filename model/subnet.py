import tensorflow as tf
import tensorflow.keras as keras

from .layers import ConvBlock


initializer = keras.initializers.VarianceScaling(scale=2., mode='fan_out', distribution='untruncated_normal')


class BoxNet(keras.layers.Layer):
    def __init__(self, depth=3, width=64, activation=keras.activations.relu, num_anchor=9, name=None):
        super(BoxNet, self).__init__(name=name)
        self.depth = depth
        self.width = width
        self.activation = activation
        self.num_anchor = num_anchor

    def build(self, shape_list):
        num_feautre = len(shape_list)

        conv_cfg = {
            'depthwise_initializer': initializer,
            'pointwise_initializer': initializer,
            'depthwise_regularizer': keras.regularizers.l2(1e-4),
            'pointwise_regularizer': keras.regularizers.l2(1e-4),
        }
        self.conv_list = [
            ConvBlock(self.width, conv_type='separable', conv_cfg=conv_cfg, apply_bn=False) for _ in range(self.depth)
        ]
        self.head = ConvBlock(self.num_anchor * 4, conv_type='separable', conv_cfg=conv_cfg, apply_bn=False)

        bn_cfg = {
            'momentum': .997,
            'epsilon': 1e-4
        }
        self.bn_list = [
            [keras.layers.BatchNormalization(**bn_cfg) for _ in range(self.depth)] for _ in range(num_feautre)
        ]
        super(BoxNet, self).build(shape_list)

    def call(self, inputs, **kwargs):
        # N = inputs[0].get_shape().as_list()[0]
        N = tf.shape(inputs[0])[0]
        output_list = list()
        for i, feature in enumerate(inputs):
            output = feature
            for j in range(self.depth):
                output = self.conv_list[j](output, **kwargs)
                output = self.bn_list[i][j](output, **kwargs)
                output = self.activation(output)
            reg = self.head(output, **kwargs)
            output_list.append(reg)
        reshape_list = list(map(lambda x: keras.backend.reshape(x, (N, -1, 4)), output_list))
        output = keras.backend.concatenate(reshape_list, axis=1)
        return output


class ClsNet(keras.layers.Layer):
    def __init__(self, depth=3, width=64, activation=keras.activations.relu, num_cls=20, num_anchor=9, name=None):
        super(ClsNet, self).__init__(name=name)
        self.depth = depth
        self.width = width
        self.activation = activation
        self.num_cls = num_cls
        self.num_anchor = num_anchor

    def build(self, shape_list):
        num_feautre = len(shape_list)

        conv_cfg = {
            'depthwise_initializer': initializer,
            'pointwise_initializer': initializer,
            'depthwise_regularizer': keras.regularizers.l2(1e-4),
            'pointwise_regularizer': keras.regularizers.l2(1e-4),
        }
        self.conv_list = [
            ConvBlock(self.width, conv_type='separable', conv_cfg=conv_cfg, apply_bn=False) for _ in range(self.depth)
        ]
        self.head = ConvBlock(self.num_cls * self.num_anchor, conv_type='separable', conv_cfg=conv_cfg, apply_bn=False)

        bn_cfg = {
            'momentum': .997,
            'epsilon': 1e-4
        }
        self.bn_list = [
            [keras.layers.BatchNormalization(**bn_cfg) for _ in range(self.depth)] for _ in range(num_feautre)
        ]
        super(ClsNet, self).build(shape_list)

    def call(self, inputs, **kwargs):
        # N = inputs[0].get_shape().as_list()[0]
        N = tf.shape(inputs[0])[0]
        output_list = list()
        for i, feature in enumerate(inputs):
            output = feature
            for j in range(self.depth):
                output = self.conv_list[j](output, **kwargs)
                output = self.bn_list[i][j](output, **kwargs)
                output = self.activation(output)
            cls = self.head(output, **kwargs)
            output_list.append(cls)
        reshape_list = list(map(lambda x: keras.backend.reshape(x, (N, -1, self.num_cls)), output_list))
        output = keras.backend.concatenate(reshape_list, axis=1)
        return keras.activations.sigmoid(output)


if __name__ == '__main__':
    import os
    import numpy as np

    os.environ['CUDA_VISIBLE_DEVICES'] = '-1'

    inputs = [
        np.random.normal(size=(4, n, n, 64)) for n in [4, 8, 16, 32, 64]
    ]

    boxnet = BoxNet()
    clsnet = ClsNet()
    result_box = boxnet(inputs, training=False)
    result_cls = clsnet(inputs, training=False)
    print(result_box.shape, result_cls.shape)
