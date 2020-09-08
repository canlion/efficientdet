import os
from typing import List, Callable
from math import floor, ceil
from random import shuffle
from functools import partial
import xml.etree.ElementTree as tree
import numpy as np
import cv2
import tensorflow as tf
from anchors import Anchor


VOC_MAP = {
    'aeroplane': 1,
    'bicycle': 2,
    'bird': 3,
    'boat': 4,
    'bottle': 5,
    'bus': 6,
    'car': 7,
    'cat': 8,
    'chair': 9,
    'cow': 10,
    'diningtable': 11,
    'dog': 12,
    'horse': 13,
    'motorbike': 14,
    'person': 15,
    'pottedplant': 16,
    'sheep': 17,
    'sofa': 18,
    'train': 19,
    'tvmonitor': 20,
}


class VOCDataset:
    def __init__(self,
                 data_dir: str,
                 version_set_pairs: List[List[str]],
                 image_extension: str = '.jpg',
                 skip_difficult: bool = False,
                 skip_truncated: bool = False,
                 data_shuffle: bool = True):
        """construct voc dataset

        :param data_dir: VOC data directory
        :param version_set_pairs: data version & set pair ex) [[2007, 'train'], [2012, ['valid']]
        :param image_extension: image extension. jpg / jpeg / png
        :param skip_difficult: skip difficult labeled data
        :param skip_truncated: skip truncated labeled data
        :param data_shuffle: shuffle data
        """
        self.data_dir = data_dir
        self.images = list()
        for version, set_ in version_set_pairs:
            with open(os.path.join(data_dir, 'VOC{}'.format(version), 'ImageSets', 'Main', set_+'.txt'), 'r') as f:
                image_names = [[version, img_name.strip()] for img_name in f.readlines()]
            self.images.extend(image_names)
        self.image_extension = image_extension if image_extension.startswith('.') else '.' + image_extension
        if self.image_extension not in ['.jpg', '.jpeg', '.png']:
            raise ValueError('image extension must be either .jpg, .jpeg, .png : {}'.format(self.image_extension))
        self.shuffle = shuffle
        self.skip_difficult = skip_difficult
        self.skip_truncated = skip_truncated
        self.data_shuffle = data_shuffle
        self.indices = list(range(len(self.images)))

    def __len__(self):
        return len(self.images)

    def data_idx_generator(self):
        if self.data_shuffle:
            shuffle(self.indices)
        for idx in self.indices:
            yield idx

    def load_image(self, version, name):
        img_path_format = os.path.join(self.data_dir, 'VOC{}', 'JPEGImages', '{}'+self.image_extension)
        image = cv2.imread(img_path_format.format(version, name))
        image = cv2.cvtColor(image, code=cv2.COLOR_BGR2RGB)
        return image

    def load_annotation(self, version, name):
        ann_path_format = os.path.join(self.data_dir, 'VOC{}', 'Annotations', '{}.xml')
        ann_path = ann_path_format.format(version, name)
        category, box = self.parse_annotation(ann_path)
        return category, box

    def parse_annotation(self, xml_path):
        ann_root = tree.parse(xml_path).getroot()
        category, boxes = [], []
        for obj in ann_root.iter('object'):
            if self.skip_truncated and int(obj.findtext('truncated')):
                continue
            if self.skip_difficult and int(obj.findtext('difficult')):
                continue
            category.append(VOC_MAP[obj.findtext('name')])
            box = obj.find('bndbox')
            box = [int(box.findtext(xy)) for xy in ['xmin', 'ymin', 'xmax', 'ymax']]
            boxes.append(box)

        return np.array(category), np.array(boxes)

    def load_data(self, idx):
        version, name = self.images[idx]
        image = self.load_image(version, name)
        category, box = self.load_annotation(version, name)

        return np.array(image), np.array(category), np.array(box)

    def get_dataset(self):
        ds = tf.data.Dataset.from_generator(generator=self.data_idx_generator,
                                            output_types=tf.int32)
        ds = ds.map(lambda idx: tf.py_function(func=self.load_data,
                                               inp=[idx],
                                               Tout=[tf.int32, tf.int32, tf.int32]),
                    num_parallel_calls=tf.data.experimental.AUTOTUNE)
        return ds


def box_anchor_matching(iou):
    # iou : (N_gt, N_anchor)
    max_iou = tf.reduce_max(iou, axis=0)  # (N_anchor,)
    max_iou_box_idx = tf.argmax(iou, axis=0)
    iou_pos_indicator = tf.cast(max_iou >= .5, tf.int32)

    # shape : (N_anchor,) -> box index if anchor's maximum iou >= 0.5 else -1 (negative)
    match = max_iou_box_idx * iou_pos_indicator + (1-iou_pos_indicator) * -1

    # TODO : force matching not-matched ground truth box

    return match


def assign_value(value, idx, negative_value):
    value_stack = tf.concat([tf.stack([negative_value]), value], axis=0)
    assigned_value = tf.gather(value_stack, idx+1)
    return assigned_value


if __name__ == '__main__':
    voc = VOCDataset(data_dir='/mnt/hdd/jinwoo/sandbox_datasets/voc_download',
                     version_set_pairs=[[2012, 'train']])
    ds = voc.get_dataset()

    for img, category, box in ds.take(5):
        print(img.shape)
        print(category)
        print(box)
