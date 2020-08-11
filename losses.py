import tensorflow as tf
import tensorflow.keras as keras


def ltrb2xywh(arr):
    arr = tf.cast(arr, tf.float32)
    wh = arr[..., 2:] - arr[..., :2]
    xy = (arr[..., :2] + arr[..., 2:]) / 2
    return tf.concat([xy, wh], axis=-1)


def IOU(box, anchor):
    # box / anchor : (N, 4) / (n, 4) -> ltrb
    box = tf.expand_dims(box, 0)  # (1, N, 4)
    anchor = tf.expand_dims(anchor, 1)  # (n, 1, 4)

    box_left_top = box[..., :2]
    box_right_bot = box[..., 2:]
    anchor_left_top = anchor[..., :2]
    anchor_right_bot = anchor[..., 2:]

    inter_lt = tf.maximum(box_left_top, anchor_left_top)  # (n, N, 2)
    inter_rb = tf.minimum(box_right_bot, anchor_right_bot)  # (n, N, 2)

    inter_wh = tf.maximum(inter_rb - inter_lt + 1, 0)
    intersection = tf.reduce_prod(inter_wh, axis=-1)  # (n, N)

    union = tf.reduce_prod(box_right_bot - box_left_top + 1, axis=-1) \
        + tf.reduce_prod(anchor_right_bot - anchor_left_top + 1, axis=-1) \
        - intersection

    return tf.clip_by_value(intersection / union, 0., 1.)  # (n, N)


class EffdetLoss(tf.keras.losses.Loss):
    def __init__(self, config):
        super().__init__()
        self.classes = config.num_classes
        self.delta = config.delta
        self.alpha = config.alpha
        self.gamma = config.gamma
        self.cls_loss_weight = config.cls_loss_weight
        self.box_loss_weight = config.box_loss_weight

        # TODO: import generate anchor function and make anchor
        self.anchor = None

    def huber_loss(self, pred, box_true, anchor):
        box_true = ltrb2xywh(box_true)
        anchor = ltrb2xywh(anchor)

        box_true_xy, box_true_wh = box_true[..., :2], box_true[..., 2:]
        anchor_xy, anchor_wh = anchor[..., :2], anchor[..., 2:]

        pred_target_xy = (box_true_xy - anchor_xy) / anchor_wh
        pred_target_wh = tf.math.log(box_true_wh / anchor_wh)
        pred_target = tf.concat([pred_target_xy, pred_target_wh], axis=-1)

        x = pred_target - pred
        x_abs = tf.abs(x)
        huber_loss = tf.where(x_abs <= self.delta,
                              .5 * tf.pow(x_abs, 2),
                              .5 * tf.pow(self.delta, 2) + self.delta * (x_abs - self.delta))
        return keras.backend.sum(huber_loss)

    def focal_loss(self, pred, cls_true):
        alpha_factor = self.alpha * tf.ones_like(pred)
        alpha_factor = tf.where(tf.equal(cls_true, 1), alpha_factor, 1. - alpha_factor)

        focal_weight = tf.where(tf.equal(cls_true, 1), 1. - pred, pred)
        focal_weight = alpha_factor * tf.pow(focal_weight, self.gamma)

        bce = keras.backend.binary_crossentropy(cls_true, pred)
        focal_loss = focal_weight * bce

        return keras.backend.sum(focal_loss)

    def call(self, box_true, box_pred, cls_true, cls_pred):
        cls_pred = tf.clip_by_value(cls_pred, 1e-4, 1. - 1e-4)
        batch_size = tf.shape(box_true)[0]
        loss_sum = 0.

        for idx in range(batch_size):
            box_pred_ = box_pred[idx]
            box_true_ = box_true[idx]
            cls_true_ = tf.one_hot(cls_true[idx], depth=self.classes)
            cls_pred_ = cls_pred[idx]

            iou = IOU(box_true_, self.anchor)
            iou_max = tf.reduce_max(iou, axis=-1)
            iou_argmax = tf.argmax(iou, axis=-1)

            idx_pos = tf.where(iou_max >= .5)[..., -1]
            idx_neg = tf.where(iou_max < .4)[..., -1]

            idx_box_pos = tf.gather(iou_argmax, idx_pos)
            box_true_pos = tf.gather(box_true_, idx_box_pos)
            box_pred_pos = tf.gather(box_pred_, idx_pos)
            anchor_pos = tf.gather(self.anchor, idx_pos)

            cls_true_pos = tf.gather(cls_true_, idx_box_pos)
            cls_pred_pos = tf.gather(cls_pred_, idx_pos)
            cls_pred_neg = tf.gather(cls_pred_, idx_neg)

            box_loss = 0.
            cls_loss = 0.
            num_box_pos = tf.cast(tf.shape(box_true_pos)[0], tf.float32)
            if tf.greater(num_box_pos, 0):
                box_loss += self.huber_loss(box_pred_pos, box_true_pos, self.anchor)
                cls_loss += self.focal_loss(cls_pred_pos, cls_true_pos)
            cls_loss += self.focal_loss(cls_pred_neg, tf.zeros_like(cls_pred_neg))

            loss_sum += box_loss * self.box_loss_weight / tf.maximum(1., 4 * num_box_pos) \
                        + cls_loss * self.cls_loss_weight / tf.maximum(1., num_box_pos)

        return loss_sum / tf.cast(batch_size, tf.float32)
