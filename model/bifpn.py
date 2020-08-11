from math import ceil
from itertools import count

import tensorflow as tf
import tensorflow.keras as keras


class ResampleOp(keras.layers.Layer):
    def __init__(self, H_target, W_target, C_target, **kwargs):
        """
        Adjust the size of the feature. If the depth is different with target channel, adjust it by convolution.
        :param H_target: int : target height
        :param W_target: int : target width
        :param C_target: int : target channel
        """
        super(ResampleOp, self).__init__(**kwargs)
        self.H_target = H_target
        self.W_target = W_target
        self.C_target = C_target

        self.conv_1x1_op = None
        self.resample_op = None

    def build(self, input_shape):
        H, W, C = input_shape[1:]

        # channels not matched
        if C != self.C_target:
            self.conv_1x1_op = keras.layers.SeparableConv2D(
                filters=self.C_target,
                kernel_size=1,
                padding='same',
            )
            self.bn_op = keras.layers.BatchNormalization()

        # down-sample
        if H > self.H_target and W > self.W_target:
            pool_strides = (ceil(H/self.H_target), ceil(W/self.W_target))
            pool_size = tuple(stride + 1 for stride in pool_strides)
            self.resample_op = keras.layers.MaxPool2D(
                pool_size=pool_size,
                strides=pool_strides,
                padding='same',
            )
        # up-sample
        elif H < self.H_target or W < self.W_target:
            self.resample_op = keras.layers.experimental.preprocessing.Resizing(
                height=self.H_target,
                width=self.W_target,
                interpolation='nearest',
            )
        super(ResampleOp, self).build(input_shape)

    def call(self, inputs, **kwargs):
        if self.conv_1x1_op is not None:
            inputs = self.conv_1x1_op(inputs)
            inputs = self.bn_op(inputs, **kwargs)

        if self.resample_op is not None:
            inputs = self.resample_op(inputs)

        return inputs


class FastFusion(keras.layers.Layer):
    def __init__(self, epsilon=1e-4, **kwargs):
        super(FastFusion, self).__init__(**kwargs)
        self._epsilon = epsilon

    def build(self, feature_list):
        num_features = len(feature_list)
        self.feature_w = self.add_weight(
            name=self.name+'_feature_w',
            shape=(1, 1, 1, 1, num_features),
            initializer=keras.initializers.ones(),
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


class BiFPNLayer(keras.layers.Layer):
    def __init__(self, config, feats_size, **kwargs):
        super(BiFPNLayer, self).__init__(**kwargs)
        self.width = config.fpn_width
        self.act_fn = config.act_fn
        self.feats_size = feats_size

    def build(self, input_shape):
        num_level = len(input_shape)
        self.op_list = self.get_operation_map(num_level)

        for op_dict in self.op_list:
            feat_size = op_dict['feat_size']
            node_num = len(op_dict['node'])
            op_dict['resample'] = [ResampleOp(*feat_size, self.width) for _ in range(node_num)]
            op_dict['fusion'] = FastFusion()
            op_dict['conv'] = keras.Sequential([
                keras.layers.Activation(self.act_fn),
                keras.layers.SeparableConv2D(self.width, 3, padding='same'),
                keras.layers.BatchNormalization()
            ])

        super(BiFPNLayer, self).build(input_shape)

    def call(self, inputs, **kwargs):
        num_level = len(inputs)
        for op in self.op_list:
            resamples = [resample_op(inputs[node], **kwargs) for resample_op, node in zip(op['resample'], op['node'])]
            fusion = op['fusion'](resamples)
            feat = op['conv'](fusion, **kwargs)
            inputs.append(feat)

        return inputs[-num_level:]

    def get_operation_map(self, num_level):
        max_level = num_level - 1
        node_id = {i: [i] for i in range(num_level)}
        id_cnt = count(num_level)
        op_list = list()

        for i in range(max_level-1, -1, -1):
            op_list.append({'feat_size': self.feats_size[i], 'node': [node_id[i+1][-1], node_id[i][-1]]})
            node_id[i].append(next(id_cnt))

        for i in range(1, max_level+1):
            op_list.append({'feat_size': self.feats_size[i], 'node': node_id[i]+[node_id[i-1][-1]]})
            node_id[i].append(next(id_cnt))

        return op_list


def get_features_size(input_size, max_level):
    if isinstance(input_size, int):
        features_size = [(input_size, input_size)]
    else:
        features_size = [input_size]
    for level in range(1, max_level+1):
        f_size = (ceil(features_size[-1][0] / 2), ceil(features_size[-1][1] / 2))
        features_size.append(f_size)
    return features_size[1:]


class BiFPN(keras.layers.Layer):
    def __init__(self, config, **kwargs):
        super(BiFPN, self).__init__(**kwargs)
        self.config = config
        self.width = config.fpn_width
        self.repeat = config.fpn_repeat
        self.min_level = config.min_level
        self.max_level = config.max_level
        self.num_levels = self.max_level - self.min_level + 1
        self.feats_size = get_features_size(config.input_size, config.max_level)

    def build(self, input_shape):
        input_shape = input_shape[self.min_level-1: self.max_level]
        if len(input_shape) < self.num_levels:
            self.downsample_ops = [
                ResampleOp(*size, self.width) for size in self.feats_size[len(input_shape)-self.num_levels:]
            ]

        feats_size = self.feats_size[self.min_level-1: self.max_level]
        self.bifpn_ops = [BiFPNLayer(self.config, feats_size) for _ in range(self.repeat)]

    def call(self, inputs, **kwargs):
        inputs = inputs[self.min_level-1: self.max_level]
        if len(inputs) < self.num_levels:
            # down-sample
            for ds_op in self.downsample_ops:
                inputs.append(ds_op(inputs[-1], **kwargs))

        assert len(inputs) == self.num_levels

        for bifpn_op in self.bifpn_ops:
            inputs = bifpn_op(inputs, **kwargs)

        return inputs


if __name__ == '__main__':
    import sys
    import numpy as np
    sys.path.append('..')
    from config import get_config

    c = get_config()
    bifpn = BiFPN(c)

    inputs = [np.random.normal(size=(3, size, size, 8)) for size in [256, 128, 64, 32, 16]]
    result = bifpn(inputs)
    print([r.shape for r in result])
