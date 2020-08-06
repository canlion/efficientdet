from functools import partial

import tensorflow as tf
import tensorflow.keras as keras


class BoxNet(keras.layers.Layer):
    def __init__(self, config, name=None):
        super(BoxNet, self).__init__(name=name)
        self.repeat = config.box_repeat
        self.width = config.box_width
        self.act_fn = config.act_fn
        self.num_anchor = len(config.size_scale) * len(config.ratio)

    def build(self, input_shape):
        conv = partial(keras.layers.SeparableConv2D,
                       kernel_size=3, strides=1, padding='same',
                       depthwise_initializer=keras.initializers.VarianceScaling(),
                       pointwise_initializer=keras.initializers.VarianceScaling())

        self.convs = [conv(filters=self.width, name='Box_Conv_{}'.format(i)) for i in range(self.repeat)]
        self.bns = [keras.layers.BatchNormalization(name='Box_BN_{}'.format(i)) for i in range(self.repeat)]
        self.acts = [keras.layers.Activation(self.act_fn, name='Box_Act_{}'.format(i)) for i in range(self.repeat)]

        self.box_head = conv(filters=self.num_anchor*4)

        super(BoxNet, self).build(input_shape)

    def call(self, inputs, **kwargs):
        batch_size = tf.shape(inputs[0])[0]
        box_list = list()
        for feature in inputs:
            for conv, bn, act in zip(self.convs, self.bns, self.acts):
                feature = act(bn(conv(feature), **kwargs))
            box = self.box_head(feature)
            box_list.append(box)

        box_reshape_list = list(map(lambda x: keras.backend.reshape(x, (batch_size, -1, 4)), box_list))
        box_concat = keras.backend.concatenate(box_reshape_list, axis=1)
        return box_concat


class ClsNet(keras.layers.Layer):
    def __init__(self, config, name=None):
        super(ClsNet, self).__init__(name=name)
        self.repeat = config.class_repeat
        self.width = config.class_width
        self.act_fn = config.act_fn
        self.num_classes = config.num_classes
        self.num_anchor = len(config.size_scale) * len(config.ratio)

    def build(self, input_shape):
        conv = partial(keras.layers.SeparableConv2D,
                       kernel_size=3, strides=1, padding='same',
                       depthwise_initializer=keras.initializers.VarianceScaling(),
                       pointwise_initializer=keras.initializers.VarianceScaling())

        self.convs = [conv(filters=self.width, name='Cls_Conv_{}'.format(i)) for i in range(self.repeat)]
        self.bns = [keras.layers.BatchNormalization(name='Cls_BN_{}'.format(i)) for i in range(self.repeat)]
        self.acts = [keras.layers.Activation(self.act_fn, name='Cls_Act_{}'.format(i)) for i in range(self.repeat)]

        self.cls_head = conv(filters=self.num_anchor * self.num_classes)

        super(ClsNet, self).build(input_shape)

    def call(self, inputs, **kwargs):
        batch_size = tf.shape(inputs[0])[0]
        cls_list = list()
        for feature in inputs:
            for conv, bn, act in zip(self.convs, self.bns, self.acts):
                feature = act(bn(conv(feature), **kwargs))
            cls = self.cls_head(feature)
            cls_list.append(cls)

        cls_reshape_list = list(map(lambda x: keras.backend.reshape(x, (batch_size, -1, self.num_classes)), cls_list))
        cls_concat = keras.backend.concatenate(cls_reshape_list, axis=1)
        return keras.backend.sigmoid(cls_concat)


if __name__ == '__main__':
    import os
    import sys
    import numpy as np

    sys.path.append('..')
    from config import get_config
    os.environ['CUDA_VISIBLE_DEVICES'] = '-1'

    config = get_config()

    inputs = [
        np.random.normal(size=(4, n, n, 64)) for n in [64, 32, 16, 8, 4]
    ]

    boxnet = BoxNet(config)
    clsnet = ClsNet(config)
    result_box = boxnet(inputs, training=False)
    result_cls = clsnet(inputs, training=False)
    print(result_box.shape, result_cls.shape)
