import tensorflow as tf
import tensorflow.keras as keras


class FastFusion(keras.layers.Layer):
    def __init__(self, epsilon=1e-4, name=None):
        super(FastFusion, self).__init__(name=name)
        self._epsilon = epsilon

    def build(self, feature_list):
        num_features = len(feature_list)
        self.feature_w = self.add_weight(
            name=self.name+'_feature_w',
            shape=(1, 1, 1, 1, num_features),
            initializer=keras.initializers.constant(1/num_features),
            trainable=True,
            dtype=self.dtype
        )
        super(FastFusion, self).build(feature_list)

    def call(self, inputs, **kwargs):
        w_nonzero = keras.activations.relu(self.feature_w)
        w_normalize = w_nonzero / (tf.reduce_sum(w_nonzero) + self._epsilon)
        feature_stack = tf.stack(inputs, axis=-1)
        feature_weighted = tf.multiply(feature_stack, w_normalize)
        feature_fusion = tf.reduce_sum(feature_weighted, axis=-1)

        return feature_fusion


class ConvBlock(keras.layers.Layer):
    convs = {
        'normal': keras.layers.Conv2D,
        'separable': keras.layers.SeparableConv2D,
        'depthwise': keras.layers.DepthwiseConv2D,
    }

    convs_cfg = {
        'normal': {'padding': 'same',
                   'activation': None,
                   'use_bias': True,
                   'kernel_initializer': 'he_normal',
                   'bias_initializer': 'zeros',
                   'kernel_regularizer': None
                   },
        'separable': {'padding': 'same',
                      'activation': None,
                      'use_bias': True,
                      'depthwise_initializer': 'he_normal',
                      'pointwise_initializer': 'he_normal',
                      'bias_initializer': 'zeros',
                      'depthwise_regularizer': None,
                      'pointwise_regularizer': None
                      },
        'depthwise': {'padding': 'same',
                      'activation': None,
                      'use_bias': True,
                      'depthwise_initializer': 'he_normal',
                      'bias_initializer': 'zeros',
                      'depthwise_regularizer': None
                      },
    }

    def __init__(self,
                 filters=None,
                 kernel_size=3,
                 strides=1,
                 conv_type='normal',
                 conv_cfg=None,
                 apply_bn=True,
                 bn_cfg=None,
                 activation=None,
                 name=None):
        super(ConvBlock, self).__init__(name=name)
        assert conv_type in ['normal', 'separable', 'depthwise'], '{}: Not allowed convolution type.'.format(conv_type)
        conv = self.convs[conv_type]
        cfg = self.convs_cfg[conv_type]
        if conv_cfg is not None:
            cfg.update(conv_cfg)

        self.bn, self.activation = None, None

        if conv_type == 'depthwise':
            self.conv = conv(kernel_size=kernel_size, strides=strides, **cfg)
        else:
            self.conv = conv(filters=filters, kernel_size=kernel_size, strides=strides, **cfg)

        if apply_bn:
            cfg = dict()
            if bn_cfg is not None:
                cfg.update(bn_cfg)
            self.bn = keras.layers.BatchNormalization(**cfg)

        if activation is not None:
            self.activation = keras.layers.Activation(activation)

    def call(self, inputs, **kwargs):
        outputs = self.conv(inputs)
        if self.bn is not None:
            outputs = self.bn(outputs, **kwargs)
        if self.activation is not None:
            outputs = self.activation(outputs)

        return outputs


if __name__ == '__main__':
    import os
    os.environ['CUDA_VISIBLE_DEVICES'] = '-1'

    normal = ConvBlock(8, 3, 1, activation=keras.activations.swish)
    sep = ConvBlock(8, 3, 1, conv_type='separable', activation=None)
    dw = ConvBlock(kernel_size=3, strides=1, conv_type='depthwise', activation=keras.activations.swish)

    print(normal.submodules)
    print(sep.submodules)
    print(dw.submodules)

    import numpy as np

    inputs = np.random.normal(size=(4, 256, 256, 3))
    print(normal(inputs).shape)

    ff = FastFusion()

    inputs_0 = np.ones((4, 4, 4, 3)) * 4
    inputs_1 = inputs_0.copy()
    print(ff([inputs_0, inputs_1]))

