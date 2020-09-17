from absl import logging
from typing import Any

import tensorflow.keras as keras


class Config:
    def update(self,
               key: str,
               val: Any):
        if key in self.__dict__:
            self.__dict__[key] = val
            logging.info('Config update - key : {} / val : {}'.format(key, val))
        else:
            raise KeyError('Key is not exist : {}'.format(key))

    def mass_update(self, override_dict: dict):
        for key, val in override_dict.items():
            self.update(key, val)

    def get_sub_dict(self, keys: list):
        sub_dict = dict()
        for key in keys:
            sub_dict[key] = self.__dict__[key]
        return sub_dict

    def add_sub_dict(self, sub_dict: dict):
        for key, val in sub_dict.items():
            if key not in self.__dict__:
                self.__dict__[key] = val
            else:
                raise KeyError('Key is exist : {}'.format(key))


def get_default_config():
    c = Config()
    c.act_fn = keras.activations.swish
    c.dataset_config = {
        'train': {
            'data_dir': '/mnt/hdd/jinwoo/sandbox_datasets/voc_download',
            'version_set_pair': [[2012, 'train'], [2007, 'train'], [2007, 'val'], ['2007Test', 'test']],
            'data_shuffle': True,
            'default_aug': True,
            'rescale_min': .1,
            'rescale_max': 2.,
            'batch_size': 4,
        },
        'valid': {
            'data_dir': '/mnt/hdd/jinwoo/sandbox_datasets/voc_download',
            'version_set_pair': [[2012, 'valid']],
            'data_shuffle': False,
            'default_aug': False,
            'batch_size': 4,
        },
    }

    c.backbone_name = 'efficientnet-b0'
    c.backbone_config = {}

    c.input_size = (512, 512)
    c.min_level = 3
    c.max_level = 7

    c.fpn_ch = 64
    c.fpn_repeat = 3

    c.predictor_width = 64
    c.predictor_repeat = 3
    c.num_classes = 20

    c.anchor_size_scale = 4
    c.anchor_ratios = [(1., 1.), (1.4, .7), (.7, 1.4)]
    c.anchor_scales = [2**(0/3), 2**(1/3), 2**(2/3)]

    return c

