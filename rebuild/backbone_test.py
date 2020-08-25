import os
import tensorflow as tf

from config import get_default_config
from backbone import get_effnet_params, build_effnet

# gpus = tf.config.experimental.list_physical_devices('GPU')
# for gpu in gpus:
#     tf.config.experimental.set_memory_growth(gpu, True)
os.environ['CUDA_VISIBLE_DEVICES'] = '-1'


config = get_default_config()
effnet_params = get_effnet_params(config)
effnet = build_effnet(effnet_params, feature_only=True, name=config.backbone_name)

effnet.summary()
output = effnet(tf.random.normal(shape=(2, 512, 512, 3), dtype=tf.float32))
print([o.shape for o in output])
