from typing import Tuple
from math import ceil
import tensorflow as tf
import tensorflow.keras as keras


class ResampleOp(keras.layers.Layer):
    def __init__(self, target_ch: int):
        super().__init__()
        self.target_ch = target_ch

    def build(self, input_shape):
        if input_shape[-1] != self.target_ch:
            self.conv = keras.layers.Conv2D(
                filters=self.target_ch,
                kernel_size=1,
                padding='same'
            )
            self.bn = keras.layers.experimental.SyncBatchNormalization()

    def call(self, inputs, target_size=None, **kwargs):
        # h, w, c = tf.shape(inputs)[1:]
        h, w, c = inputs.shape.as_list()[1:]

        # matching channel
        if c != self.target_ch:
            inputs = self.bn(self.conv(inputs), **kwargs)

        # set target size
        if target_size is None:
            target_h, target_w = ceil(h/2), ceil(w/2)
        else:
            # target_h, target_w = tf.shape(all_feats[self.feat_level])[1:3]
            target_h, target_w = target_size

        # down-sample
        if h > target_h and w > target_w:
            strides = (ceil(h / target_h), ceil(w / target_w))
            pool_size = tuple(stride + 1 for stride in strides)
            inputs = keras.layers.MaxPool2D(
                pool_size=pool_size,
                strides=strides,
                padding='same')(inputs)
        # up-sample or keep
        elif h <= target_h and w <= target_w:
            # up-sample
            if h < target_h or w < target_w:
                inputs = tf.image.resize(
                    images=inputs,
                    size=(target_h, target_w),
                    method='nearest')
            # else keep

        return inputs


class Fusion(keras.layers.Layer):
    def __init__(self, act_fn, target_ch):
        super().__init__()
        self.act_fn = act_fn
        self.target_ch = target_ch

    def build(self, input_shape):
        num_feat = len(input_shape)
        self.feat_weight = self.add_weight(
            shape=(num_feat,),
            initializer='ones',
            trainable=True,
        )

        self.conv = keras.layers.SeparableConv2D(
            filters=self.target_ch,
            padding='same',
            use_bias=True
        )
        self.bn = keras.layers.experimental.SyncBatchNormalization()

    def call(self, inputs, **kwargs):
        w_sum = tf.reduce_sum(self.feat_weight)
        inputs_stack = tf.add_n([feat * self.feat_weight[i] for i, feat in enumerate(inputs)])
        fusion = inputs_stack / (w_sum + 1e-4)

        fusion = self.act_fn(self.bn(self.conv(fusion), **kwargs))
        return fusion
