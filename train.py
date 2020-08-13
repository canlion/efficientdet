import argparse
import json

import tensorflow as tf
import tensorflow.keras as keras
from tqdm import tqdm

from model.efficientdet import EfficientDet
from data.data_generator import VOCGenerator
from data.augmenation import get_agumentator
from train_lib import get_optimizer
from losses import EffdetLoss
from config import get_config


def parse_args(args):
    parser = argparse.ArgumentParser(description='EfficientDet training arguments.')
    parser.add_argument('--voc_path', help='VOC pascal dataset directory path.')
    parser.add_argument('--train_pair', help='Add VOC year-set pair for training.',
                        type=json.loads)
    parser.add_argument('--valid_pair', help='Add VOC year-set pair for validation.',
                        type=json.loads)
    parser.add_argument('--allow_growth', help='allow gpu memory growth.',
                        action='store_true', default=False)
    return parser.parse_args(args)


def main(args=None):
    def l2_reg(model, w=4e-5):
        return w * tf.add_n([tf.nn.l2_loss(v) for v in model.trainable_variables if v.name.find('kernel') > 0])

    @tf.function(
        input_signature=[
            tf.TensorSpec(shape=None, dtype=tf.int32),
            tf.TensorSpec(shape=None, dtype=tf.int32),
            tf.TensorSpec(shape=None, dtype=tf.float32)
        ]
    )
    def train(imgs, cls_true, box_true):
        with tf.GradientTape() as tape:
            pred = model(imgs, training=True)
            loss = effdet_loss((box_true, cls_true), pred)
            loss_l2 = loss + l2_reg(model)
        grad = tape.gradient(loss_l2, model.trainable_variables)
        optimizer.apply_gradients(zip(grad, model.trainable_variables))
        loss_train.update_state(loss)

    @tf.function(
        input_signature=[
            tf.TensorSpec(shape=None, dtype=tf.int32),
            tf.TensorSpec(shape=None, dtype=tf.int32),
            tf.TensorSpec(shape=None, dtype=tf.float32),
        ]
    )
    def eval(imgs, cls_true, box_true):
        pred = model(imgs, training=False)
        loss = effdet_loss((box_true, cls_true), pred)
        loss_valid.update_state(loss)

    args = parse_args(args)

    gpus = tf.config.experimental.list_physical_devices('GPU')
    if gpus and args.allow_growth:
        try:
            for gpu in gpus:
                tf.config.experimental.set_memory_growth(gpu, True)
            logical_gpus = tf.config.experimental.list_logical_devices('GPU')
            print(len(gpus), "Physical GPUs,", len(logical_gpus), "Logical GPUs")
        except RuntimeError as e:
            print(e)

    config = get_config()
    print('freeze backbone : {}'.format(config.freeze_backbone))

    model = EfficientDet(config)
    model.build()
    model.summary()

    voc_train = VOCGenerator(data_dir=args.voc_path,
                             version_set_pairs=args.train_pair,
                             augmentation_unit=get_agumentator(train=True,
                                                               img_size=config.input_size,
                                                               min_visibility=.3),
                             batch_size=config.train_batch_size,
                             drop_remainder=True,
                             )
    step_per_epoch = len(voc_train)
    config.update('step_per_epoch', step_per_epoch)
    train_ds = voc_train.get_dataset()

    voc_valid = VOCGenerator(data_dir=args.voc_path,
                             version_set_pairs=args.valid_pair,
                             augmentation_unit=get_agumentator(train=False,
                                                               img_size=config.input_size,),
                             batch_size=config.valid_batch_size,
                             drop_remainder=False,
                             )
    valid_step = len(voc_valid)
    valid_ds = voc_valid.get_dataset()

    optimizer = get_optimizer(config)
    effdet_loss = EffdetLoss(config)

    loss_train = keras.metrics.Mean(name='loss_train')
    loss_valid = keras.metrics.Mean(name='loss_valid')

    train_pbar = tqdm(total=step_per_epoch)
    valid_pbar = tqdm(total=valid_step)

    min_loss = None

    for epoch in range(config.total_epoch):
        for step, data in enumerate(train_ds):
            train(*data)
            train_pbar.update()
            if (step+1) % (step_per_epoch//10) == 0:
                train_pbar.set_description('train loss : {}'.format(loss_train.result()))
                loss_train.reset_states()
        train_pbar.reset()

        for data in valid_ds:
            eval(*data)
            valid_pbar.update()
        print(':+:+:+: {} epoch :+:+:+: loss : {}'.format(epoch, loss_valid.result()))
        if min_loss is None or loss_valid.result() < min_loss:
            model.save_weights('saved_weights/min_loss.weight')
            min_loss = loss_valid.result()
            print('weights saved.')
        loss_valid.reset_states()

        valid_pbar.reset()


if __name__ == '__main__':
    main()
