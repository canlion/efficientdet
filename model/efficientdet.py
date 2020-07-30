from collections import namedtuple

import tensorflow as tf
import tensorflow.keras as keras

from .bifpn import BiFPN
from .efficientnet import EfficientNetBase
from .subnet import BoxNet, ClsNet


DetArgs = namedtuple('EfficientDetArgs', ['input_size', 'backbone', 'net_filters', 'biFPN_repeat', 'subnet_repeat'])


class EfficientDet(keras.Model):
    EFFICIENTDET_ARGS = {
        'D0': DetArgs(input_size=512, backbone='B0', net_filters=64, biFPN_repeat=3, subnet_repeat=3),
        'D1': DetArgs(input_size=640, backbone='B1', net_filters=88, biFPN_repeat=4, subnet_repeat=3),
        'D2': DetArgs(input_size=768, backbone='B2', net_filters=112, biFPN_repeat=5, subnet_repeat=3),
        'D3': DetArgs(input_size=896, backbone='B3', net_filters=160, biFPN_repeat=6, subnet_repeat=4),
        'D4': DetArgs(input_size=1024, backbone='B4', net_filters=224, biFPN_repeat=7, subnet_repeat=4),
        'D5': DetArgs(input_size=1280, backbone='B5', net_filters=288, biFPN_repeat=7, subnet_repeat=4),
        'D6': DetArgs(input_size=1280, backbone='B6', net_filters=384, biFPN_repeat=8, subnet_repeat=5),
        'D7': DetArgs(input_size=1536, backbone='B6', net_filters=384, biFPN_repeat=8, subnet_repeat=5),
    }

    def __init__(self, D, num_anchors=9, classes=20, freeze_backbone=False, freeze_bn=False):
        super(EfficientDet, self).__init__()
        self.effdet_args = self.EFFICIENTDET_ARGS[D]
        self.num_anchors = num_anchors
        self.classes = classes
        self.freeze_backbone = freeze_backbone
        self.freeze_bn = freeze_bn

        self.imgnet_mean = tf.constant([0.485, 0.456, 0.406], shape=(1, 1, 1, 3), name='imagenet_mean')
        self.imgnet_std = tf.constant([0.229, 0.224, 0.225], shape=(1, 1, 1, 3), name='imagenet_std')

    def build(self, shape_list):
        args = self.effdet_args
        input_shape = (args.input_size, args.input_size, 3)
        inputs = keras.layers.Input(shape=input_shape)
        self.effnet = keras.Model(inputs, EfficientNetBase(args.backbone, freeze_bn=self.freeze_bn).outputs(inputs))
        if self.freeze_backbone:
            self.effnet.trainable = False

        self.bifpn = BiFPN(args.net_filters, args.biFPN_repeat, 3, 5)
        self.boxnet = BoxNet(args.subnet_repeat, args.net_filters, num_anchor=self.num_anchors)
        self.clsnet = ClsNet(args.subnet_repeat, args.net_filters, num_anchor=self.num_anchors, num_cls=self.classes)


        self.effnet.load_weights(
            'model/efficientnet_weights/'
            'efficientnet-{}_weights_tf_dim_ordering_tf_kernels_autoaugment_notop.h5'.format(args.backbone.lower()),
            by_name=True
        )

        super(EfficientDet, self).build(shape_list)

    def call(self, inputs, training=None, mask=None):
        outputs = self.normalizer(inputs)
        outputs = self.effnet(outputs, training=training)
        outputs = self.bifpn(outputs, training=training)[::-1]

        box = self.boxnet(outputs, training=training)
        cls = self.clsnet(outputs, training=training)

        return box, cls

    def normalizer(self, inputs):
        inputs = tf.cast(inputs, tf.float32) / 255.
        return (inputs - self.imgnet_mean) / self.imgnet_std


if __name__ == '__main__':
    import os
    import numpy as np

    os.environ['CUDA_VISIBLE_DEVICES'] = '-1'

    effdet = EfficientDet('D0')

    inputs = np.random.normal(size=(4, 512, 512, 3))
    outputs = effdet(inputs, training=False)

    print([x.shape for x in outputs])
