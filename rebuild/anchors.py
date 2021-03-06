from itertools import product
import numpy as np
import tensorflow as tf

from utils import get_feature_sizes


class Anchor:
    def __init__(self, config):
        self.config = config
        self.input_size = config.input_size
        self.anchor_size_scale = config.anchor_size_scale
        self.anchor_ratios = config.anchor_ratios
        self.anchor_scales = config.anchor_scales
        self.feature_sizes = get_feature_sizes(self.input_size, config.min_level, config.max_level)

    def generate_anchor(self, img_size=None):
        if img_size is not None:
            input_h, input_w = img_size
            feature_sizes = get_feature_sizes(img_size, self.config.min_level, self.config.max_level)
        else:
            input_h, input_w = self.input_size
            feature_sizes = self.feature_sizes
        all_anchor = list()
        for feat_h, feat_w in feature_sizes:
            stride_h, stride_w = input_h/feat_h, input_w/feat_w

            center_x, center_y = np.meshgrid(
                np.arange(stride_w/2, input_w, stride_w),
                np.arange(stride_h/2, input_h, stride_h),
            )

            feat_anchor_list = list()
            for scale, ratio in product(self.anchor_scales, self.anchor_ratios):
                anchor_h = self.anchor_size_scale * stride_h * scale
                anchor_w = self.anchor_size_scale * stride_w * scale
                anchor_stretch_h = anchor_h * ratio[0]
                anchor_stretch_w = anchor_w * ratio[1]

                feat_anchor = np.stack([
                    center_x - anchor_stretch_w / 2,
                    center_y - anchor_stretch_h / 2,
                    center_x + anchor_stretch_w / 2,
                    center_y + anchor_stretch_h / 2,
                ], axis=-1)
                feat_anchor_list.append(feat_anchor)
            feat_all_anchor = np.stack(feat_anchor_list, axis=-2).reshape(-1, 4)
            all_anchor.append(feat_all_anchor)

        all_anchor = np.concatenate(all_anchor, axis=0)
        return tf.convert_to_tensor(all_anchor, dtype=tf.float32)


if __name__ == '__main__':
    from collections import namedtuple
    Config = namedtuple('config', ['input_size', 'anchor_size_scale', 'anchor_ratios',
                                   'anchor_scales', 'min_level', 'max_level'])
    config = Config((512, 512), 4, [(1, 1), (.7, 1.4), (1.4, .7)],
                    [0/3, 1/3, 2/3], 3, 7)

    a = Anchor(config)
    print(a.generate_anchor())
