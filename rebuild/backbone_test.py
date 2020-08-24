import tensorflow as tf
from config import get_default_config
from backbone import get_effnet_params, EffNet


config = get_default_config()
effnet_params = get_effnet_params(config)
effnet = EffNet(effnet_params, name=config.backbone_name)

effnet.build((1, 512, 512, 3))
effnet.summary()

ckpt_path = tf.train.latest_checkpoint('/home/jws/efficientdet_keras/efficientnet-b0')
effnet.load_weights(ckpt_path)
#
# print(effnet.layers[5].trainable_weights)

