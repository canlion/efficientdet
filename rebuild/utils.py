from typing import Tuple
from math import ceil
import tensorflow as tf


def get_feature_sizes(img_size: Tuple[int, int],
                      min_level: int,
                      max_level: int):
    feature_size_list = [img_size]
    for _ in range(max_level):
        feature_size_list.append((
            ceil(feature_size_list[-1][0] / 2),
            ceil(feature_size_list[-1][1] / 2),
        ))

    return feature_size_list[min_level: max_level+1]


def ltrb2xywh(arr):
    arr = tf.cast(arr, tf.float32)
    wh = arr[..., 2:] - arr[..., :2]
    xy = (arr[..., :2] + arr[..., 2:]) / 2
    return tf.concat([xy, wh], axis=-1)


def IOU(box, anchor):
    # box : (N_gt, 4), anchor : (N_anchor, 4)
    box = tf.cast(box, tf.float32)
    anchor = tf.cast(anchor, tf.float32)
    box_expand = tf.expand_dims(box, 0)  # (1, N_gt, 4)
    anchor_expand = tf.expand_dims(anchor, 1)  # (N_anchor, 1, 4)

    box_lt = box_expand[..., :2]  # (1, N_gt, 2)
    box_rb = box_expand[..., 2:]
    anchor_lt = anchor_expand[..., :2]  # (N_anchor, 1, 2)
    anchor_rb = anchor_expand[..., 2:]

    inter_lt = tf.maximum(box_lt, anchor_lt)  # (N_anchor, N_gt, 2)
    inter_rb = tf.minimum(box_rb, anchor_rb)

    inter_wh = tf.maximum(inter_rb - inter_lt + 1, 0)
    intersection = tf.reduce_prod(inter_wh, axis=-1)  # (N_anchor, N_gt)

    union = tf.reduce_prod(box_rb - box_lt + 1, axis=-1) \
        + tf.reduce_prod(anchor_rb - anchor_lt + 1, axis=-1) \
        - intersection

    return tf.clip_by_value(intersection / union, 0., 1.)  # (N_anchor, N_gt)
