from itertools import product
import math
import numpy as np
import tensorflow as tf


class Anchor:
    def __init__(self, config):
        self.img_size = config.input_size
        self.levels = tuple(range(config.min_level, config.max_level + 1))
        self.strides = tuple(2 ** level for level in self.levels)
        self.sizes = tuple(config.anchor_scale * stride for stride in self.strides)
        self.scales = config.size_scales
        self.ratios = config.ratios

    def generate_anchor(self):
        def generate_anchor_base(size):
            anchor_base = list()
            area = size ** 2
            for scale, ratio in product(self.scales, self.ratios, repeat=1):
                w = np.sqrt(area * scale / ratio)
                h = ratio * w
                anchor_base.append([h, w])
            anchor_half = np.array(anchor_base) / 2
            anchor_xyxy = np.concatenate([-1 * anchor_half, anchor_half], axis=-1)
            return anchor_xyxy

        h, w = self.img_size
        feature_sizes = [(math.ceil(h / stride), math.ceil(w / stride)) for stride in self.strides]

        anchor_list = list()
        for (f_h, f_w), stride, size in zip(feature_sizes, self.strides, self.sizes):
            xv, yv = np.meshgrid(range(f_w), range(f_h))
            grid = np.stack([xv, yv], axis=-1).reshape((-1, 1, 2))
            anchor_center = ((grid + .5) * stride)
            anchor_center = np.tile(anchor_center, (1, 1, 2))
            anchor_base = generate_anchor_base(size)
            anchor = anchor_center + anchor_base.reshape((1, -1, 4))
            anchor_list.append(anchor.reshape(-1, 4))

        return tf.concat(anchor_list, axis=0)
