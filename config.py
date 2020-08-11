import tensorflow as tf
import tensorflow.keras as k


class Config:
    def update(self, key, val):
        if key in self.__dict__:
            self.__dict__[key] = val
        else:
            print('{} is not exist.'.format(key))

    def mass_update(self, config_dict):
        for key, val in config_dict.items():
            self.update(key, val)


def get_config():
    c = Config()

    # general
    c.input_size = (512, 512)
    c.act_fn = k.activations.swish
    c.num_classes = 20
    c.weights_decay = 4e-5
    c.freeze_backbone = True
    c.backbone_scale = 'B0'
    c.backbone_load_weights = True
    c.drop_rate = 0.

    # bifpn
    c.min_level = 3
    c.max_level = 7
    c.fpn_width = 64
    c.fpn_repeat = 3
    c.method = 'fast'

    # predictor
    c.box_width = 64
    c.box_repeat = 3
    c.class_width = 64
    c.class_repeat = 3

    # loss
    c.alpha = .25
    c.gamma = 1.5
    c.delta = .1
    c.cls_loss_weight = 1.
    c.box_loss_weight = 50.

    # anchor
    c.anchor_scale = 4.
    c.size_scale = (2**0, 2**(1/3), 2**(2/3),)
    c.ratio = (.5, 1., 2.)

    # training
    c.total_epoch = 300
    c.step_per_epoch = None

    c.optimizer = 'SGD'
    c.optimizer_config = {
        'momentum': .9
    }

    c.lr_method = 'cosine'
    c.lr_init = 1e-3
    c.lr_warmup = 1e-4
    c.lr_warmup_epoch = 1

    # c.lr_method = 'stepwise'
    # c.lr_stepwise_plan = [
    #     {'epoch': 10, 'lr': 1e-4},
    #     {'epoch': 100, 'lr': 1e-5},
    # ]
    # c.lr_init = 1e-3
    # c.lr_warmup = 1e-4
    # c.lr_warmup_epoch = 1



    return c
