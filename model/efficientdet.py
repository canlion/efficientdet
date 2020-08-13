from collections import namedtuple

import tensorflow as tf
import tensorflow.keras as keras

from model.bifpn import BiFPN
from model.efficientnet import EfficientNetBase
from model.subnet import BoxNet, ClsNet


DetArgs = namedtuple('EfficientDetArgs', ['input_size', 'backbone', 'net_filters', 'biFPN_repeat', 'subnet_repeat'])


class EfficientDet(keras.Model):
    def __init__(self, config):
        super(EfficientDet, self).__init__()
        self.config = config

        self.imgnet_mean = tf.constant([0.485, 0.456, 0.406], shape=(1, 1, 1, 3), name='imagenet_mean')
        self.imgnet_std = tf.constant([0.229, 0.224, 0.225], shape=(1, 1, 1, 3), name='imagenet_std')

    def build(self):
        c = self.config
        input_shape = (None, c.input_size[0], c.input_size[1], 3)
        backbone_input = keras.layers.Input(shape=input_shape[1:])
        backbone_output = EfficientNetBase(c.backbone_scale, drop_rate=c.drop_rate).outputs(backbone_input)
        self.effnet = keras.Model(backbone_input, backbone_output)
        if c.freeze_backbone:
            self.effnet.trainable = False

        self.bifpn = BiFPN(self.config)
        self.boxnet = BoxNet(self.config)
        self.clsnet = ClsNet(self.config)

        if c.backbone_load_weights:
            self.effnet.load_weights(
                'model/efficientnet_weights/efficientnet-{}_weights_tf_dim_ordering_tf_kernels_autoaugment_notop.h5'
                .format(self.config.backbone_scale.lower()),
                by_name=True
            )

        super(EfficientDet, self).build(input_shape)

    def call(self, inputs, training=None, mask=None, inference=None):
        outputs = self.normalizer(inputs)
        outputs = self.effnet(outputs, training=training)
        outputs = self.bifpn(outputs, training=training)

        box = self.boxnet(outputs, training=training)
        cls = self.clsnet(outputs, training=training)

        return box, cls

    def normalizer(self, inputs):
        inputs = tf.cast(inputs, tf.float32) / 255.
        return (inputs - self.imgnet_mean) / self.imgnet_std


if __name__ == '__main__':
    import os
    import numpy as np
    import sys

    sys.path.append('..')
    from config import get_config

    config = get_config()
    config.backbone_load_w = False

    os.environ['CUDA_VISIBLE_DEVICES'] = '-1'

    effdet = EfficientDet(config)
    effdet.build()
    effdet.summary()

    inputs = np.random.normal(size=(4, 512, 512, 3)).astype(np.float32)
    outputs = effdet(inputs, training=False)

    print([x.shape for x in outputs])
