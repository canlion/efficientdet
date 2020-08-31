from functools import partial
from math import ceil, log
from itertools import count
import tensorflow as tf
import tensorflow.keras as keras

from backbone import get_effnet_params, build_effnet


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
        super().build(input_shape)

    def call(self, inputs, target_size=None, **kwargs):
        h, w, c = inputs.shape.as_list()[1:]

        # matching channel
        if c != self.target_ch:
            inputs = self.bn(self.conv(inputs), **kwargs)

        # set target size
        if target_size is None:
            target_h, target_w = ceil(h/2), ceil(w/2)
        else:
            target_h, target_w = target_size

        # down-sample
        if h > target_h and w > target_w:
            strides = (ceil(h / target_h), ceil(w / target_w))
            pool_size = tuple(stride + 1 for stride in strides)
            inputs = keras.backend.pool2d(
                x=inputs,
                pool_size=pool_size,
                strides=strides,
                padding='same',
                pool_mode='max')
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
            kernel_size=3,
            padding='same',
            use_bias=True
        )
        self.bn = keras.layers.experimental.SyncBatchNormalization()
        super().build(input_shape)

    def call(self, inputs, **kwargs):
        w_sum = tf.reduce_sum(self.feat_weight)
        inputs_stack = tf.add_n([feat * self.feat_weight[i] for i, feat in enumerate(inputs)])
        fusion = inputs_stack / (w_sum + 1e-4)

        fusion = self.bn(self.conv(self.act_fn(fusion)), **kwargs)
        return fusion


class BiFPN(keras.layers.Layer):
    def __init__(self, target_ch, act_fn=keras.activations.swish):
        super().__init__()
        self.target_ch = target_ch
        self.act_fn = act_fn

    def build(self, input_shape):
        self.set_op_map(len(input_shape))
        self.resample_ops = [
            [ResampleOp(self.target_ch) for _ in range(len(op_dict['feat_idx']))]
            for op_dict in self.op_list
        ]
        self.fusion_ops = [Fusion(self.act_fn, self.target_ch) for _ in range(len(self.op_list))]
        super().build(input_shape)

    def set_op_map(self, num_feats):
        node_dict = {i: [i] for i in range(num_feats)}
        node_id = count(num_feats)
        self.op_list = list()
        for i in range(num_feats-2, -1, -1):
            self.op_list.append(
                {'level': i, 'feat_idx': [node_dict[i+1][-1], node_dict[i][-1]]}
            )
            node_dict[i].append(next(node_id))
        for i in range(1, num_feats):
            self.op_list.append(
                {'level': i, 'feat_idx': [node_dict[i-1][-1], *node_dict[i]]}
            )
            node_dict[i].append(next(node_id))

    def call(self, inputs, **kwargs):
        num_feats = len(inputs)
        for op_dict, resample_ops, fusion_op in zip(self.op_list, self.resample_ops, self.fusion_ops):
            resampled_feats = list()
            target_size = inputs[op_dict['level']].shape.as_list()[1:3]
            for feat_idx, resample_op in zip(op_dict['feat_idx'], resample_ops):
                resampled_feats.append(
                    resample_op(inputs[feat_idx], target_size, **kwargs)
                )
            feat_fusion = fusion_op(resampled_feats, **kwargs)
            inputs.append(feat_fusion)

        return inputs[-num_feats:]


class BiFPNS(keras.layers.Layer):
    def __init__(self, config):
        super().__init__()
        self.target_ch = config.fpn_ch
        self.repeat = config.fpn_repeat
        self.act_fn = config.act_fn
        self.min_level = config.min_level
        self.max_level = config.max_level

    def build(self, input_shape):
        num_feats = len(input_shape)
        self.resample_ops = [ResampleOp(self.target_ch) for _ in range(self.max_level-num_feats)]
        self.fpns = [BiFPN(self.target_ch, self.act_fn) for _ in range(self.repeat)]
        super().build(input_shape)

    def call(self, inputs, **kwargs):
        for resample_op in self.resample_ops:
            inputs.append(resample_op(inputs[-1], **kwargs))

        inputs = inputs[self.min_level-1: self.max_level]

        for fpn in self.fpns:
            inputs = fpn(inputs, **kwargs)

        return inputs


class ClassNet(keras.layers.Layer):
    def __init__(self,
                 width,
                 repeat,
                 num_feats,
                 num_classes,
                 num_anchors,
                 act_fn):
        super().__init__()
        self.act_fn = act_fn

        self.conv_ops = [
            keras.layers.SeparableConv2D(
                filters=width,
                kernel_size=3,
                strides=1,
                padding='same',
                depthwise_initializer=keras.initializers.VarianceScaling(),
                pointwise_initializer=keras.initializers.VarianceScaling(),
            ) for _ in range(repeat)
        ]

        self.bn_ops = [
            [keras.layers.experimental.SyncBatchNormalization() for _ in range(num_feats)] for _ in range(repeat)
        ]

        self.head = keras.layers.SeparableConv2D(
            filters=num_anchors*num_classes,
            kernel_size=3,
            strides=1,
            padding='same',
            bias_initializer=keras.initializers.Constant(-log((1-0.01)/0.01)),
        )

    def call(self, inputs, **kwargs):
        outputs = list()
        for feat_idx, feat in enumerate(inputs):
            for conv_op, bn_ops in zip(self.conv_ops, self.bn_ops):
                feat = conv_op(feat)
                feat = bn_ops[feat_idx](feat, **kwargs)
                feat = self.act_fn(feat)
            outputs.append(self.head(feat))
        return outputs


class BoxNet(keras.layers.Layer):
    def __init__(self,
                 width,
                 repeat,
                 num_feats,
                 num_anchors,
                 act_fn):
        super().__init__()
        self.act_fn = act_fn

        self.conv_ops = [
            keras.layers.SeparableConv2D(
                filters=width,
                kernel_size=3,
                strides=1,
                padding='same',
                depthwise_initializer=keras.initializers.VarianceScaling(),
                pointwise_initializer=keras.initializers.VarianceScaling(),
            ) for _ in range(repeat)
        ]

        self.bn_ops = [
            [keras.layers.experimental.SyncBatchNormalization() for _ in range(num_feats)] for _ in range(repeat)
        ]

        self.head = keras.layers.SeparableConv2D(
            filters=num_anchors*4,
            kernel_size=3,
            strides=1,
            padding='same',
            depthwise_initializer=keras.initializers.VarianceScaling(),
            pointwise_initializer=keras.initializers.VarianceScaling(),
        )

    def call(self, inputs, **kwargs):
        outputs = list()
        for feat_idx, feat in enumerate(inputs):
            for conv_op, bn_ops in zip(self.conv_ops, self.bn_ops):
                feat = conv_op(feat)
                feat = bn_ops[feat_idx](feat, **kwargs)
                feat = self.act_fn(feat)
            outputs.append(self.head(feat))
        return outputs


class EffDet(keras.Model):
    def __init__(self, config):
        super().__init__()

        backbone_config = get_effnet_params(config)
        self.backbone = build_effnet(backbone_config, feature_only=True, name=config.backbone_name)

        self.fpn = BiFPNS(config)

        self.clsnet = ClassNet(width=config.fpn_ch,
                               repeat=config.fpn_repeat,
                               num_feats=config.max_level-config.min_level+1,
                               num_classes=config.num_classes,
                               num_anchors=len(config.anchor_ratios)*len(config.anchor_scales),
                               act_fn=config.act_fn)
        self.boxnet = BoxNet(width=config.fpn_ch,
                             repeat=config.fpn_repeat,
                             num_feats=config.max_level-config.min_level+1,
                             num_anchors=len(config.anchor_ratios)*len(config.anchor_scales),
                             act_fn=config.act_fn)

    def call(self, inputs, training=None, mask=None):
        outputs = self.backbone(inputs, training=training)
        outputs = self.fpn(outputs, training=training)

        cls_pred = self.clsnet(outputs, training=training)
        box_pred = self.boxnet(outputs, training=training)

        return cls_pred, box_pred
