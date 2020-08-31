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
    'aeroplane': 0,
    'bicycle': 1,
    'bird': 2,
    'boat': 3,
    'bottle': 4,
    'bus': 5,
    'car': 6,
    'cat': 7,
    'chair': 8,
    'cow': 9,
    'diningtable': 10,
    'dog': 11,
    'horse': 12,
    'motorbike': 13,
    'person': 14,
    'pottedplant': 15,
    'sheep': 16,
    'sofa': 17,
    'train': 18,
    'tvmonitor': 19
}


class VOCDataset:
    def __init__(self,
                 data_dir: str,
                 version_set_pairs: List[List[str]],
                 preprocessing_unit: Callable = None,
                 image_extension: str = '.jpg',
                 batch_size: int = 8,
                 drop_remainder: bool = True,
                 skip_difficult: bool = False,
                 skip_truncated: bool = False,
                 data_shuffle: bool = True):
        self.data_dir = data_dir
        self.images = list()
        for version, set_ in version_set_pairs:
            with open(os.path.join(data_dir, 'VOC{}'.format(version), 'ImageSets', 'Main', set_+'.txt'), 'r') as f:
                image_names = [[version, img_name.strip()] for img_name in f.readlines()]
            self.images.extend(image_names)
        self.image_extension = image_extension if image_extension.startswith('.') else '.' + image_extension
        if self.image_extension not in ['.jpg', '.jpeg', '.png']:
            raise ValueError('image extension must be either .jpg, .jpeg, .png : {}'.format(self.image_extension))
        self.preprocessing_unit = preprocessing_unit
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.drop_remainder = drop_remainder
        self.skip_difficult = skip_difficult
        self.skip_truncated = skip_truncated
        self.data_shuffle = data_shuffle

    def __len__(self):
        ceil_or_floor = floor if self.drop_remainder else ceil
        return int(ceil_or_floor(len(self.images) / self.batch_size))

    @property
    def size(self):
        return len(self.images)

    def batch_path_generator(self):
        if self.data_shuffle:
            shuffle(self.images)
        for index in range(len(self)):
            batch_path = self.images[index * self.batch_size: (index+1) * self.batch_size]
            if len(batch_path) < self.batch_size and not self.drop_remainder:
                batch_path.extend(self.images[:self.batch_size-len(batch_path)])
            yield batch_path

    def load_images(self, batch_path):
        img_path_format = os.path.join(self.data_dir, 'VOC{}', 'JPEGImages', '{}'+self.image_extension)
        images = [cv2.imread(img_path_format.format(version, name)) for version, name in batch_path]
        images = list(map(partial(cv2.cvtColor, code=cv2.COLOR_BGR2RGB), images))
        return images

    def load_annotations(self, batch_path):
        ann_path_format = os.path.join(self.data_dir, 'VOC{}', 'Annotations', '{}.xml')
        labels, boxes = [], []
        for version, name in batch_path:
            ann_path = ann_path_format.format(version, name)
            label, box = self.parse_annotations(ann_path)
            labels.append(label)
            boxes.append(boxes)

        return labels, boxes

    def parse_annotations(self, xml_path):
        ann_root = tree.parse(xml_path).getroot()
        labels, boxes = [], []
        for obj in ann_root.iter('object'):
            if self.skip_truncated and int(obj.findtext('truncated')):
                continue
            if self.skip_difficult and int(obj.findtext('difficult')):
                continue
            labels.append(VOC_MAP[obj.findtext('name')])
            box = obj.find('bndbox')
            box = [int(box.findtext(xy))-1 for xy in ['xmin', 'ymin', 'xmax', 'ymax']]
            boxes.append(box)

        return np.array(labels), np.array(boxes)

    def load_and_preprocessing(self, batch_path):
        images = self.load_images(batch_path)
        labels, boxes = self.load_annotations(batch_path)

        images_prep, labels_prep, boxes_prep = [], [], []
        for img, label, box in zip(images, labels, boxes):
            preprocessed = self.preprocessing_unit(image=img, bboxes=box, labels=label)
            images_prep.append(preprocessed['images'])
            labels_prep.append(preprocessed['labels'])
            boxes_prep.append(preprocessed['bboxes'])

        images_batch = np.stack(images_prep, axis=0).astype(np.int32)
        labels_batch, boxes_batch = self.collate(labels_prep, boxes_prep)

        return (tf.convert_to_tensor(images_batch, tf.int32),
                tf.convert_to_tensor(labels_batch, tf.int32),
                tf.convert_to_tensor(boxes_batch, tf.int32))

    def collate(self, labels, boxes):
        """Make labels, boxes into one array each."""
        max_num = max(len(label) for label in labels)

        if max_num > 0:
            frame_boxes = np.zeros((self.batch_size, max_num, 4))
            frame_labels = np.ones((self.batch_size, max_num)) * -1

            for i, (label, box) in enumerate(zip(labels, boxes)):
                if len(label) > 0:
                    frame_labels[i, :len(label)] = label
                    frame_boxes[i, :len(label), :] = box
        else:
            frame_boxes = np.zeros((self.batch_size, 1, 4))
            frame_labels = np.ones((self.batch_size, 1)) * -1

        return frame_labels, frame_boxes
