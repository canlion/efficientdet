from math import pi
import tensorflow as tf
import tensorflow_addons as tfa

from efficientdet import EffDet


# learning rate
class LRWarmupSchedule(tf.keras.optimizers.schedules.LearningRateSchedule):
    def __init__(self, lr_init, lr_warmup, step_warmup):
        super().__init__()
        self.lr_init = lr_init
        self.lr_warmup = lr_warmup
        self.step_warmup = step_warmup

    def lr_fn(self, step):
        raise NotImplemented

    def __call__(self, step):
        warmup_lr = self.lr_warmup + (self.lr_init - self.lr_warmup) * (step / self.step_warmup)
        lr = self.lr_fn(step)
        return tf.where(step < self.step_warmup, warmup_lr, lr)

    def get_config(self):
        config = {
            'lr_init': self.lr_init,
            'lr_warmup': self.lr_warmup,
            'step_warmup': self.step_warmup,
        }
        return config


class CosineLRSchedule(LRWarmupSchedule):
    def __init__(self, lr_init, lr_warmup, step_warmup, step_total):
        super().__init__(lr_init=lr_init, lr_warmup=lr_warmup, step_warmup=step_warmup)
        self.step_total = step_total

    def lr_fn(self, step):
        step = step - self.step_warmup
        lr = self.lr_init * (1 + tf.math.cos(pi * step / (self.step_total - self.step_warmup))) / 2.
        return lr

    def get_config(self):
        config = super().get_config()
        config.update({'step_total': self.step_total})
        return config


def get_lr_schedule(config):
    if config.step_per_epoch is None:
        raise KeyError('step_per_epoch is not updated.')

    method = config.lr_method
    if method == 'cosine':
        return CosineLRSchedule(
            lr_init=config.lr_init,
            lr_warmup=config.lr_warmup,
            step_warmup=config.epoch_warmup*config.step_per_epoch,
            step_total=config.epoch_total*config.step_per_epoch,
        )
    else:
        raise ValueError(f'undefined lr schedule : {method}')


# optimizer
def get_optimizer(config):
    lr_schedule = get_lr_schedule(config)
    method = config.optimizer.lower()
    if method == 'sgd':
        opt = tf.keras.optimizers.SGD(learning_rate=lr_schedule, **config.optimizer_config)
    else:
        raise ValueError(f'undefined optimizer : {method}')

    if config.moving_average_decay is not None:
        opt = tfa.optimizers.MovingAverage(opt, average_decay=config.moving_average_decay)

    return opt


# training model
class EffDetTrainer(EffDet):
    def __init__(self, config):
        super().__init__(config)
        self.config = config

    def loss_fn(self, cls_output, box_output, cls_true, box_true, loss_dict):
        positive_indicator = tf.not_equal(cls_true, -1)
        num_positive = tf.reduce_sum(tf.cast(positive_indicator, tf.float32)) + 1.

        cls_loss_fn = self.loss.get('cls_loss')
        if cls_loss_fn:
            cls_true_one_hot = tf.one_hot(cls_true, self.config.num_classes)
            cls_loss = cls_loss_fn(cls_true_one_hot, cls_output) / num_positive
            loss_dict['cls_loss'] = cls_loss

        box_loss_fn = self.loss.get('box_loss')
        if box_loss_fn:
            box_loss = box_loss_fn(box_true, box_output)
            box_loss = box_loss / (4 * num_positive)
            loss_dict['box_loss'] = box_loss


        total_loss = loss_dict['cls_loss'] + self.config.box_loss_w * loss_dict['box_loss']
        loss_dict['total_loss'] = total_loss

        return total_loss

    def train_step(self, data):
        image_batch, cls_batch, box_batch = data

        loss_dict = {}
        with tf.GradientTape() as tape:
            cls_pred, box_pred = self(image_batch, training=True)
            loss = self.loss_fn(cls_pred, box_pred, cls_batch, box_batch, loss_dict)

        gradients = tape.gradient(loss, self.trainable_variables)
        self.optimizer.apply_gradients(zip(gradients, self.trainable_variables))
        return loss_dict

    def test_step(self, data):
        image_batch, cls_batch, box_batch = data
        image_batch_norm = self.normalize(image_batch)

        loss_dict = {}
        cls_pred, box_pred = self(image_batch_norm, training=False)
        _ = self.loss_fn(cls_pred, box_pred, cls_batch, box_batch, loss_dict)
        return loss_dict


