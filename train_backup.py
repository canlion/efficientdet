import argparse
import json
from time import time

import tensorflow as tf
import tensorflow.keras as keras
from tqdm import tqdm

from model.efficientdet import EfficientDet
from data.data_generator import VOCGenerator
from data.augmenation import get_agumentator
from train_lib import CosineLRDecay, LRWarmup
from anchors.anchors import generate_anchor
from losses import effdet_loss


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
    parser.add_argument('--epochs', help='Number of epochs to train.', type=int, default=100)
    parser.add_argument('--allow_growth', help='allow gpu memory growth.',
                        action='store_true', default=False)
    # parser.add_argument('--train_steps', help='Steps per epoch', type=int, default=1000)
    # parser.add_argument('--steps', help='Number of steps per epoch.', type=int, default=2000)
    return parser.parse_args(args)


def main(args=None):
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
    model = EfficientDet(D='D0',
                         freeze_backbone=args.freeze_backbone,
                         freeze_bn=args.freeze_bn)
    model.build((model.effdet_args.input_size, model.effdet_args.input_size, 3))
    model.summary()

    voc_train = VOCGenerator(data_dir=args.voc_path,
                             version_set_pairs=args.train_pair,
                             augmentation_unit=get_agumentator(train=True,
                                                               img_size=model.effdet_args.input_size,
                                                               min_visibility=.3),
                             batch_size=args.batch_size,
                             # image_size=model.effdet_args.input_size,
                             )
    step_per_epoch = len(voc_train)
    train_ds = voc_train.get_dataset()

    voc_valid = VOCGenerator(data_dir=args.voc_path,
                             version_set_pairs=args.valid_pair,
                             augmentation_unit=get_agumentator(train=False,
                                                               img_size=model.effdet_args.input_size,),
                             batch_size=args.batch_size,
                             # image_size=model.effdet_args.input_size,
                             )
    valid_step = len(voc_valid)
    valid_ds = voc_valid.get_dataset()

    lr = CosineLRDecay(init_lr=args.lr, total_batch=step_per_epoch*(args.epochs-1))
    lr = LRWarmup(scheduler=lr, step=step_per_epoch, init_lr=args.lr/10)

    optimizer = keras.optimizers.Adam(learning_rate=lr)

    loss_train = keras.metrics.Mean(name='loss_train')
    loss_valid = keras.metrics.Mean(name='loss_valid')

    prog_bar_train = tqdm(total=step_per_epoch)
    prog_bar_valid = tqdm(total=valid_step)

    anchor = tf.cast(generate_anchor(img_size=model.effdet_args.input_size), tf.float32)
    for epoch in range(args.epochs):
        for step, (imgs, cls_true, box_true) in enumerate(train_ds):
            with tf.GradientTape() as tape:
                # t_infer_start = time()
                box_pred, cls_pred = model(imgs, training=True)
                # t_infer_end = time()
                loss = effdet_loss(box_pred, cls_pred, box_true, cls_true, anchor)
                # t_loss_end = time()
            grad = tape.gradient(loss, model.trainable_variables)
            # t_grad_end = time()
            optimizer.apply_gradients(zip(grad, model.trainable_variables))
            # t_apply_grad_end = time()
            #
            # print('t_infer : {} / t_loss : {} / t_grad : {} / t_apply : {}'.format(t_infer_end-t_infer_start,
            #                                                                        t_loss_end-t_infer_end,
            #                                                                        t_grad_end-t_loss_end,
            #                                                                        t_apply_grad_end-t_grad_end))

            loss_train.update_state(loss)
            prog_bar_train.update()

            if (step+1) % (step_per_epoch//10) == 0:
                prog_bar_train.set_description('train loss : {}'.format(loss_train.result()))
                loss_train.reset_states()
        prog_bar_train.reset()

        for imgs, cls_true, box_true in valid_ds:
            box_pred, cls_pred = model(imgs, training=False)
            loss = effdet_loss(box_pred, cls_pred, box_true, cls_true, anchor)

            loss_valid.update_state(loss)
            prog_bar_valid.update()
        print('---- mean loss ---- {}'.format(loss_valid.result()))
        loss_valid.reset_states()
        prog_bar_valid.reset()


if __name__ == '__main__':
    main()