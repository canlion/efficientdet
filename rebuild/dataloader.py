import os
from absl import logging
from typing import List, Callable, Tuple
import xml.etree.ElementTree as tree
import numpy as np
import cv2
import tensorflow as tf
from utils import ltrb2xywh, IOU
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
                 data_shuffle: bool = True,
                 comment: str = None):
        """construct voc dataset

        :param data_dir: VOC data directory
        :param version_set_pairs: data version & set pair ex) [[2007, 'train'], [2012, ['valid']]
        :param image_extension: image extension. jpg / jpeg / png
        :param skip_difficult: skip difficult labeled data
        :param skip_truncated: skip truncated labeled data
        :param data_shuffle: shuffle data
        """
        logging.info(comment)
        self.data_dir = data_dir
        logging.info('VOCDataset| VOC data directory: {}'.format(self.data_dir))
        self.images = list()
        for version, set_ in version_set_pairs:
            logging.info('VOCDataset| Add dataset : {} - {}'.format(version, set_))
            with open(os.path.join(data_dir, 'VOC{}'.format(version), 'ImageSets', 'Main', set_+'.txt'), 'r') as f:
                image_names = [[version, img_name.strip()] for img_name in f.readlines()]
            self.images.extend(image_names)
        self.image_extension = image_extension if image_extension.startswith('.') else '.' + image_extension
        if self.image_extension not in ['.jpg', '.jpeg', '.png']:
            raise ValueError('image extension must be either .jpg, .jpeg, .png : {}'.format(self.image_extension))
        self.skip_difficult = skip_difficult
        self.skip_truncated = skip_truncated
        self.data_shuffle = data_shuffle
        self.indices = list(range(len(self.images)))
        logging.info('VOCDataset| Dataset is shuffled.' if self.data_shuffle
                     else 'VOCDataset| Dataset is not shuffled.')

    def __len__(self):
        return len(self.images)

    def data_idx_generator(self):
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
        if self.data_shuffle:
            ds = ds.shuffle(64)
        ds = ds.map(lambda idx: tf.py_function(func=self.load_data,
                                               inp=[idx],
                                               Tout=[tf.int32, tf.int32, tf.int32]),
                    num_parallel_calls=tf.data.experimental.AUTOTUNE)
        return ds


class Preprocessor:
    def __init__(self,
                 output_size: Tuple[int, int],
                 is_training: bool = True,
                 rescale_min: float = None,
                 rescale_max: float = None,
                 augmentation_func: Callable = None):
        self.output_size = output_size
        self.is_training = is_training
        self.augmentation_func = augmentation_func
        self.rescale_min = rescale_min
        self.rescale_max = rescale_max

        logging.info('Preprocessor| output size : {}'.format(output_size))
        logging.info('Preprocessor| training : {}'.format(is_training))
        logging.info('Preprocessor| rescale factor : {} / {}'.format(rescale_min, rescale_max))

    def resize_crop(self, img, box, category):
        H, W = tf.cast(tf.shape(img)[:2], tf.float32)  # f32
        target_H, target_W = tf.cast(self.output_size, tf.float32)  # f32

        # get accurate factor
        if self.is_training:
            factor = tf.random.uniform((), self.rescale_min, self.rescale_max, dtype=tf.float32)  # f32
            scaled_H = factor * tf.cast(target_H, tf.float32)  # f32
            scaled_W = factor * tf.cast(target_W, tf.float32)  # f32
            scaled_H = tf.math.floor(scaled_H)  # f32
            scaled_W = tf.math.floor(scaled_W)  # f32
        else:
            scaled_H, scaled_W = target_H, target_W  # f32

        factor = tf.minimum(scaled_H / H, scaled_W / W)  # f32
        tf.print(factor)
        # get new, scaled size
        new_H, new_W = H * factor, W * factor  # f32
        tf.print(new_H, new_W)

        # get offset if new size is larger than output_size. we will crop image.
        if self.is_training:
            offset_range_y = tf.maximum(0., new_H-target_H)  # f32
            offset_range_x = tf.maximum(0., new_W-target_W)  # f32
            offset_y = tf.cast(offset_range_y * tf.random.uniform((), 0., 1.), tf.int32)  # i32
            offset_x = tf.cast(offset_range_x * tf.random.uniform((), 0., 1.), tf.int32)  # i32
        else:
            offset_y, offset_x = 0, 0  # i32? 64?
        img = tf.image.resize(img, (new_H, new_W))
        img = tf.cast(img, tf.int32)
        img = img[offset_y:offset_y+self.output_size[0], offset_x:offset_x+self.output_size[1]]
        tf.print(tf.shape(img))
        img = tf.image.pad_to_bounding_box(img, 0, 0, self.output_size[0], self.output_size[1])

        box_offset = tf.convert_to_tensor([[offset_x, offset_y, offset_x, offset_y]])
        box = tf.cast(box, tf.float32)
        box = tf.cast(box * factor, tf.int32)
        box = box - box_offset

        valid_range_x = self.output_size[1]-1 if target_W < new_W else new_W-1
        valid_range_y = self.output_size[0]-1 if target_H < new_H else new_H-1
        box = self.clip_box(box, valid_range_x, valid_range_y)

        box_indicator = tf.where(
            tf.reduce_prod(box[..., 2:] - box[..., :2], axis=1) > 0
        )
        box = tf.gather_nd(box, box_indicator)
        box = tf.cast(box, tf.int32)
        category = tf.gather_nd(category, box_indicator)

        return img, category, box

    def clip_box(self, box, valid_range_x, valid_range_y):
        box = tf.clip_by_value(box,
                               clip_value_min=[0, 0, 0, 0],
                               clip_value_max=[valid_range_x, valid_range_y, valid_range_x, valid_range_y])
        return box


class Labeler:
    def __init__(self, anchor_args):
        self.anchor = Anchor(**anchor_args).generate_anchor()

    def anchor_box_pairing(self, box, category):
        iou = IOU(box, self.anchor)
        iou_max = tf.reduce_max(iou, axis=-1)
        iou_argmax = tf.argmax(iou, axis=-1)

        box_indicator = tf.where(iou_max >= .5, iou_argmax, -1)

        box_concat = tf.concat([tf.zeros((1, 4)), box], axis=0)
        category_concat = tf.concat([tf.stack([0]), category], axis=-1)

        anchor_box_pair = tf.gather(box_concat, box_indicator+1)
        anchor_category_pair = tf.gather(category_concat, box_indicator+1)

        reg_target = self.encoding_delta(self.anchor, anchor_box_pair, box_indicator)
        return reg_target, anchor_category_pair

    def encoding_delta(self, anchor, box, indicator):
        anchor_xywh, box_xywh = ltrb2xywh(anchor), ltrb2xywh(box)
        xy_encoding = (box_xywh[..., :2] - anchor_xywh[..., :2]) / anchor_xywh[..., 2:]
        wh_encoding = tf.math.log(box_xywh[..., 2:] / anchor_xywh[..., 2:])
        delta = tf.concat([xy_encoding, wh_encoding], axis=-1)
        delta = tf.where(indicator > -1, delta, box)
        return delta


if __name__ == '__main__':
    voc = VOCDataset(data_dir='/mnt/hdd/jinwoo/sandbox_datasets/voc_download',
                     version_set_pairs=[[2012, 'train']])
    ds = voc.get_dataset()

    prep = Preprocessor((512, 512), True, .1, 2.)

    for img, category, box in ds.take(5):
        print(img.shape)
        print(category)
        print(box)

        img, category, box = prep.resize_crop(img, box, category)
        print(img.shape)
        print(category)
        print(box)
