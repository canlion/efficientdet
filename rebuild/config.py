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

    c.input_size = (512, 512)

    c.backbone_name = 'efficientnet-b0'
    c.backbone_config = {}

    c.min_level = 3
    c.max_level = 7

    c.fpn_ch = 64

    c.anchor_size_scale = 4
    c.anchor_ratios = [(1., 1.), (1.4, .7), (.7, 1.4)]
    c.anchor_scales = [2**(0/3), 2**(1/3), 2**(2/3)]


    return c

