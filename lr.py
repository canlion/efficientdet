import math
import tensorflow as tf


class LRWarmup(tf.keras.optimizers.schedules.LearningRateSchedule):
    def __init__(self, scheduler, step, init_lr=0):
        super(LRWarmup, self).__init__()
        assert step > 0, 'learning rate warm-up step must be larger than zero.'

        self.init_lr = init_lr
        self.scheduler = scheduler
        self.lr = self.scheduler(0)
        self.step = step

    @tf.function
    def __call__(self, steps):
        if tf.less_equal(steps, self.step):
            return self.init_lr + (self.lr-self.init_lr) * (steps / self.step)
        else:
            return self.scheduler(steps - self.step)


class CosineLRDecay(tf.keras.optimizers.schedules.LearningRateSchedule):
    def __init__(self, init_lr, total_batch):
        super(CosineLRDecay, self).__init__()
        self.init_lr = init_lr
        self.T = total_batch

    @tf.function
    def __call__(self, steps):
        lr = self.init_lr * (1 + tf.math.cos(steps * math.pi / self.T)) / 2
        return lr