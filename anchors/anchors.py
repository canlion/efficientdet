from itertools import product
import math
import numpy as np


def generate_anchor(img_size=(512, 512),
                    p_levels=(3, 4, 5, 6, 7),
                    strides=(8, 16, 32, 64, 128),
                    sizes=(32, 64, 128, 256, 512),
                    scales=(2**0, 2**(1/3), 2**(2/3)),
                    ratios=(.5, 1., 2.)):
    if isinstance(img_size, int):
        img_size = (img_size, img_size)
    H, W = img_size
    feature_sizes = [(math.ceil(H / (2**p)), math.ceil(W / (2**p))) for p in p_levels]

    anchor_list = list()
    for (H, W), stride, size in zip(feature_sizes, strides, sizes):
        grid = np.stack(np.meshgrid(range(H), range(W)), axis=-1).reshape((-1, 1, 2))
        anchor_center = ((grid + .5) * stride)
        anchor_center = np.tile(anchor_center, (1, 1, 2))
        anchor_base = generate_anchor_base(size, scales, ratios)
        anchor = anchor_center + anchor_base.reshape((1, -1, 4))
        anchor_list.append(anchor)

    return np.concatenate(anchor_list, axis=0).reshape(-1, 4)


def generate_anchor_base(size, scales, ratios):
    anchor_base = list()

    area = size ** 2

    for scale, ratio in product(scales, ratios, repeat=1):
        w = np.sqrt(area * scale / ratio)
        h = ratio * w
        anchor_base.append([h, w])

    anchor_half = np.array(anchor_base) / 2
    anchor_xyxy = np.concatenate([-1*anchor_half, anchor_half], axis=-1)
    return anchor_xyxy


if __name__ == '__main__':
    # generate_anchor_base(16, (2**0, 2**(1/3), 2**(2/3)), (.5, 1., 2.))
    anchor = generate_anchor()
    print(anchor[:3])
    print(anchor.shape)