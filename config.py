import tensorflow as tf
import tensorflow.keras as k


class Config:
    def __init__(self):
        pass

    def add_config(self, key):
        sub_config = Config()
        self.__dict__[key] = sub_config
        return sub_config


def get_config():
    c = Config()

    c.input_size = 512
    c.act_fn = k.activations.swish
    c.num_classes = 20
    c.weights_decay = 4e-5
    c.freeze_backbone = True
    c.backbone_scale = 'B0'
    c.backbone_load_w = True
    c.drop_rate = 0.

    c.min_level = 3
    c.max_level = 7
    c.fpn_width = 64
    c.fpn_repeat = 3
    c.method = 'fast'

    c.box_width = 64
    c.box_repeat = 3
    c.class_width = 64
    c.class_repeat = 3

    c.alpha = .25
    c.gamma = 1.5
    c.delta = .1
    c.box_loss_weight = 50.

    c.anchor_scale = 4.
    c.size_scale = (2**0, 2**(1/3), 2**(2/3),)
    c.ratio = (.5, 1., 2.)

    return c
