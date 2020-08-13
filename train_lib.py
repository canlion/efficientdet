import math
import tensorflow as tf


# learning rate
class LrSchedule(tf.keras.optimizers.schedules.LearningRateSchedule):
    def __init__(self, lr_init, lr_warmup, step_warmup):
        super(LrSchedule, self).__init__()
        self.lr_init = lr_init
        self.lr_warmup = lr_warmup
        self.step_warmup = step_warmup

    def lr_fn(self, step):
        raise NotImplementedError

    @tf.function
    def __call__(self, step):
        if tf.less_equal(step, self.step_warmup):
            return self.lr_warmup + (self.lr_init - self.lr_warmup) * (step / self.step_warmup)
        else:
            return self.lr_fn(step)

    def get_config(self):
        return {'lr_init': self.lr_init, 'lr_warmup': self.lr_warmup, 'step_warmup': self.step_warmup}


class CosineLrSchedule(LrSchedule):
    def __init__(self, lr_init, lr_warmup, step_warmup, step_total):
        super().__init__(lr_init, lr_warmup, step_warmup)
        self.step_total = step_total

    def lr_fn(self, step):
        step = step - self.step_warmup
        lr = self.lr_init * (1 + tf.math.cos(step * math.pi / self.step_total)) / 2.
        return lr

    def get_config(self):
        config = super().get_config()
        config.update({'step_total': self.step_total})
        return config


class StepwiseLrSchedule(LrSchedule):
    def __init__(self, lr_init, lr_warmup, step_warmup, lr_plan):
        super().__init__(lr_init, lr_warmup, step_warmup)
        self.lr_plan = sorted(lr_plan, key=lambda x: x['step'], reverse=True)

    def lr_fn(self, step):
        for plan in self.lr_plan:
            if tf.greater_equal(step, plan['step']):
                return plan['lr']
        return self.lr_init

    def get_config(self):
        config = super().get_config()
        config.update({'lr_plan': self.lr_plan})


def get_lr_schedule(config):
    if config.step_per_epoch is None:
        raise KeyError('step_per_epoch is not updated.')

    method = config.lr_method
    if method == 'cosine':
        return CosineLrSchedule(lr_init=config.lr_init,
                                lr_warmup=config.lr_warmup,
                                step_warmup=config.lr_warmup_epoch*config.step_per_epoch,
                                step_total=(config.total_epoch-config.lr_warmup_epoch)*config.step_per_epoch)
    elif method == 'stepwise':
        for plan in config.lr_stepwise_plan:
            plan['step'] = plan['epoch'] * config.step_per_epoch
        return StepwiseLrSchedule(lr_init=config.lr_init,
                                  lr_warmup=config.lr_warmup,
                                  step_warmup=config.lr_warmup_epoch*config.step_per_epoch,
                                  lr_plan=config.lr_stepwise_plan)
    else:
        raise ValueError('unknown lr_decay_method : {}'.format(method))


# optimizer
def get_optimizer(config):
    lr_schedule = get_lr_schedule(config)
    if config.optimizer.lower() == 'sgd':
        return tf.keras.optimizers.SGD(learning_rate=lr_schedule, **config.optimizer_config)
    elif config.optimizer.lower() == 'adam':
        return tf.keras.optimizers.Adam(learning_rate=lr_schedule, **config.optimizer_config)
    else:
        raise ValueError('undefined optimizer key : {}'.format(config.optimizer))
