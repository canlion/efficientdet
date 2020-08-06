import argparse
import json

import tensorflow as tf
import tensorflow.keras as keras
from tqdm import tqdm

from model.efficientdet import EfficientDet
from data.data_generator import VOCGenerator
from data.augmenation import get_agumentator
from lr import CosineLRDecay, LRWarmup
from anchors.anchors import generate_anchor
from losses import effdet_loss
from config import get_config


def parse_args(args):
    parser = argparse.ArgumentParser(description='EfficientDet training arguments.')
    parser.add_argument('--voc_path', help='VOC pascal dataset directory path.')
    parser.add_argument('--train_pair', help='Add VOC year-set pair for training.',
                        type=json.loads)
    parser.add_argument('--valid_pair', help='Add VOC year-set pair for validation.',
                        type=json.loads)
    parser.add_argument('--freeze_backbone', help='EfficientNet backbone weights freezing.',
                        action='store_true', default=False),
    parser.add_argument('--freeze_bn', help='EfficientNet batch normalization layer weights freezing.',
                        action='store_true', default=False)
    parser.add_argument('--lr', help='learning rate', type=float, default=1e-4)
    parser.add_argument('--batch_size', help='Batch size.', type=int, default=16)
    parser.add_argument('--valid_batch_size', help='Validation batch size.', type=int, default=32)
    parser.add_argument('--epochs', help='Number of epochs to train.', type=int, default=100)
    parser.add_argument('--allow_growth', help='allow gpu memory growth.',
                        action='store_true', default=False)
    # parser.add_argument('--train_steps', help='Steps per epoch', type=int, default=1000)
    # parser.add_argument('--steps', help='Number of steps per epoch.', type=int, default=2000)
    return parser.parse_args(args)


def main(args=None):
    @tf.function(
        input_signature=[
            tf.TensorSpec(shape=None, dtype=tf.int32),
            tf.TensorSpec(shape=None, dtype=tf.int32),
            tf.TensorSpec(shape=None, dtype=tf.float32)
        ]
    )
    def train(imgs, cls_true, box_true):
        with tf.GradientTape() as tape:
            box_pred, cls_pred = model(imgs, training=True)
            loss = effdet_loss(box_pred, cls_pred, box_true, cls_true, anchor)
            loss_l2 = loss + tf.reduce_sum(model.losses)
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
        box_pred, cls_pred = model(imgs, training=False)
        loss = effdet_loss(box_pred, cls_pred, box_true, cls_true, anchor)
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

    print('freeze backbone / bn : {} / {}'.format(args.freeze_backbone, args.freeze_bn))
    config = get_config()
    model = EfficientDet(config)
    # model = EfficientDet(D='D0',
    #                      freeze_backbone=args.freeze_backbone,
    #                      freeze_bn=args.freeze_bn)
    model.build((None, config.input_size, config.input_size, 3))
    model.summary()

    voc_train = VOCGenerator(data_dir=args.voc_path,
                             version_set_pairs=args.train_pair,
                             augmentation_unit=get_agumentator(train=True,
                                                               img_size=config.input_size,
                                                               min_visibility=.3),
                             batch_size=args.batch_size,
                             drop_remainder=True,
                             )
    step_per_epoch = len(voc_train)
    train_ds = voc_train.get_dataset()

    voc_valid = VOCGenerator(data_dir=args.voc_path,
                             version_set_pairs=args.valid_pair,
                             augmentation_unit=get_agumentator(train=False,
                                                               img_size=config.input_size,),
                             batch_size=args.valid_batch_size,
                             drop_remainder=False,
                             )
    valid_step = len(voc_valid)
    valid_ds = voc_valid.get_dataset()

    lr = CosineLRDecay(init_lr=args.lr, total_batch=step_per_epoch*(args.epochs-1))
    lr = LRWarmup(scheduler=lr, step=step_per_epoch, init_lr=0.)

    # optimizer = keras.optimizers.Adam(learning_rate=lr)
    optimizer = keras.optimizers.SGD(learning_rate=lr, momentum=.9)

    loss_train = keras.metrics.Mean(name='loss_train')
    loss_valid = keras.metrics.Mean(name='loss_valid')

    train_pbar = tqdm(total=step_per_epoch)
    valid_pbar = tqdm(total=valid_step)

    min_loss = None

    anchor = tf.cast(generate_anchor(img_size=config.input_size), tf.float32)
    for epoch in range(args.epochs):
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
