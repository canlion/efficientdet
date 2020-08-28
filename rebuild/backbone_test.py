import os
import tensorflow as tf

from config import get_default_config
from backbone import get_effnet_params, build_effnet
from efficientdet import BiFPNS, ClassNet, BoxNet, EffDet

# gpus = tf.config.experimental.list_physical_devices('GPU')
# for gpu in gpus:
#     tf.config.experimental.set_memory_growth(gpu, True)
os.environ['CUDA_VISIBLE_DEVICES'] = '-1'


config = get_default_config()
# effnet_params = get_effnet_params(config)
# effnet = build_effnet(effnet_params, feature_only=True, name=config.backbone_name)
#
# effnet.summary()
# # output = effnet(tf.random.normal(shape=(2, 512, 512, 3), dtype=tf.float32))
# # print([o.shape for o in output])
#
# fpn = BiFPNS(config)
# fpn_pred = fpn(effnet.output)
# print([i.shape for i in fpn_pred])
#
# clsnet = ClassNet(config.predictor_width,
#                   config.predictor_repeat,
#                   len(fpn_pred),
#                   config.num_classes,
#                   len(config.anchor_ratios)*len(config.anchor_scales),
#                   config.act_fn)
#
# boxnet = BoxNet(config.predictor_width,
#                 config.predictor_repeat,
#                 len(fpn_pred),
#                 len(config.anchor_ratios)*len(config.anchor_scales),
#                 config.act_fn)
#
# cls_pred = clsnet(fpn_pred)
# box_pred = boxnet(fpn_pred)
#
# print([i.shape for i in cls_pred])
# print([i.shape for i in box_pred])

effdet = EffDet(config)

# effdet = build_effdet(config)
outputs = effdet(tf.random.normal(shape=(2, 500, 500, 3), dtype=tf.float32))
print([i.shape for i in outputs[0]])
effdet.summary()