from typing import List, Tuple, Callable
import os
import random
import math
from functools import partial
import xml.etree.ElementTree as tree

import numpy as np
import cv2
import tensorflow as tf


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


class VOCGenerator:
    """
    PASCAL VOC datasets generator.
    """
    def __init__(self,
                 data_dir: str,
                 version_set_pairs: List[List[str]],
                 augmentation_unit: Callable,
                 image_extension: str = '.jpg',
                 batch_size: int = 8,
                 drop_remainder: bool = True,
                 # image_size: Tuple[int, int] = (512, 512),
                 skip_difficult: bool = False,
                 skip_truncated: bool = False,
                 shuffle: bool = True,
                 ):
        """
        PASCAL VOC dataset.

        :param data_dir: VOC data path. (including VOC2012, VOC2007, etc.).
        :param version_set_pairs: The VOC version and set pair. e.g. [[2007, 'val'], [2012, 'train'], ...]
        :param augmentation_unit: Albumenation Compose object for data augmentation.
        :param skip_difficult: If True, ignore difficult labeled data.
        :param skip_truncated: If True, ignore truncated labeled data.
        :param shuffle: If True, shuffle data.

        :attr self.images: include [voc_version, image_name]. e.g. [[2007, image_name_0], [2007, image_name_1], ...]
        """
        self.data_dir = data_dir
        self.images = list()
        for version, set_ in version_set_pairs:
            with open(os.path.join(data_dir, 'VOC{}'.format(version), 'ImageSets', 'Main', set_+'.txt'), 'r') as f:
                image_names = [[version, img_name.strip()] for img_name in f.readlines()]
            self.images.extend(image_names)
        self.image_extension = image_extension if image_extension.startswith('.') else '.' + image_extension
        self.batch_size = batch_size
        self.drop_remainder = drop_remainder
        # self.image_size = image_size
        # if isinstance(self.image_size, int):
        #     self.image_size = (self.image_size, self.image_size)
        self.shuffle = shuffle
        self.augmentation_unit = augmentation_unit
        self.skip_difficult = skip_difficult
        self.skip_truncated = skip_truncated

        self.indices = list(range(len(self.images)))

    def __len__(self) -> int:
        """Return number of batches."""
        return int(math.ceil(len(self.indices) / self.batch_size))

    @property
    def size(self):
        return len(self.images)

    def gen(self) -> List[int]:
        """Generator for image's name and voc version."""
        random.shuffle(self.images)
        for index in range(len(self)):
            batch_indices = self.indices[index * self.batch_size: (index+1) * self.batch_size]
            if len(batch_indices) < self.batch_size and not self.drop_remainder:
                batch_indices.extend(self.indices[:self.batch_size-len(batch_indices)])

            yield batch_indices

    def map_fn(self, batch_indices: List[int]):
        """Get the images, labels and boundingboxes corresponding to image names indices."""
        list_images = self.load_images(batch_indices)
        list_labels, list_boxes = self.load_annotations(batch_indices)

        aug_images, aug_labels, aug_boxes = list(), list(), list()
        for img, label, box in zip(list_images, list_labels, list_boxes):
            aug = self.augmentation_unit(image=img, bboxes=box, labels=label)
            aug_images.append(aug['image'])
            aug_labels.append(aug['labels'])
            aug_boxes.append(aug['bboxes'])

        batch_images = np.stack(aug_images, axis=0).astype(np.int32)
        batch_labels, batch_boxes = self.collate(aug_labels, aug_boxes)
        return batch_images.astype(np.int32), batch_labels.astype(np.int32), batch_boxes.astype(np.float32)

    def get_dataset(self):
        """Return tf.data.Dataset"""
        ds = tf.data.Dataset.from_generator(generator=self.gen, output_types=tf.int32)
        ds = ds.map(lambda indices: tf.py_function(self.map_fn, inp=[indices], Tout=[tf.int32, tf.int32, tf.float32]),
                    num_parallel_calls=tf.data.experimental.AUTOTUNE)
        return ds

    def load_images(self, indices: List[int]) -> List[np.array]:
        img_path_format = os.path.join(self.data_dir, 'VOC{}', 'JPEGImages', '{}'+self.image_extension)
        image_names = [self.images[idx] for idx in indices]
        images = [cv2.imread(img_path_format.format(version, name)) for version, name in image_names]
        images = list(map(partial(cv2.cvtColor, code=cv2.COLOR_BGR2RGB), images))

        return images

    def load_annotations(self, indices: List[int]) -> Tuple[List, List]:
        ann_path_format = os.path.join(self.data_dir, 'VOC{}', 'Annotations', '{}.xml')
        image_names = [self.images[idx] for idx in indices]
        labels, boxes = list(), list()

        for version, name in image_names:
            ann_path = ann_path_format.format(version, name)
            label, box = self.parse_annotations(ann_path)
            labels.append(label)
            boxes.append(box)

        return labels, boxes

    def parse_annotations(self, xml_path: str) -> Tuple[np.array, np.array]:
        annotation_root = tree.parse(xml_path).getroot()
        labels, boxes = [], []
        for object in annotation_root.iter('object'):
            if self.skip_truncated and int(object.findtext('truncated')):
                continue
            if self.skip_difficult and int(object.findtext('difficult')):
                continue
            labels.append(VOC_MAP[object.findtext('name')])
            box = object.find('bndbox')
            box = [int(box.findtext(xy))-1 for xy in ['xmin', 'ymin', 'xmax', 'ymax']]
            boxes.append(box)

        return np.array(labels), np.array(boxes)

    @staticmethod
    def collate(labels, boxes):
        """Make labels, boxes into one array each."""
        batch_size = len(labels)
        max_num = max(len(label) for label in labels)

        if max_num > 0:
            frame_boxes = np.zeros((batch_size, max_num, 4))
            frame_labels = np.ones((batch_size, max_num)) * -1

            for i, (label, box) in enumerate(zip(labels, boxes)):
                if len(label) > 0:
                    frame_labels[i, :len(label)] = label
                    frame_boxes[i, :len(label), :] = box
        else:
            frame_boxes = np.zeros((batch_size, 1, 4))
            frame_labels = np.ones((batch_size, 1)) * -1

        return frame_labels, frame_boxes


if __name__ == '__main__':
    from data.augmenation import get_augmentator
    ds = VOCGenerator('/mnt/hdd/jinwoo/sandbox_datasets/voc_download',
                      [[2012, 'train'], [2007, 'val']],
                      get_augmentator('train', (512, 512), min_visibility=.2),
                      '.jpg',
                      4, )

    ds = ds.get_dataset()

    for i, (a, b, c) in enumerate(ds):
        if i > 5:
            print(a.shape, a.dtype, b, c)
            break
