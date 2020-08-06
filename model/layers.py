import tensorflow as tf
import tensorflow.keras as keras





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

