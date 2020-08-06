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


def effdet_loss(box_preds, cls_preds, box_annos, cls_annos, anchor, classes=20, alpha=.25, gamma=2.):
    def smooth_l1_loss(box_pred, box_anno, anchor, delta=.1):
        # sigma_sq = sigma ** 2
        box_anno = ltrb2xywh(box_anno)
        anchor = ltrb2xywh(anchor)

        box_target_dxy = (box_anno[..., :2] - anchor[..., :2]) / anchor[..., 2:]
        box_target_dwh = tf.math.log(box_anno[..., 2:] / anchor[..., 2:])
        box_target = tf.concat([box_target_dxy, box_target_dwh], axis=-1)

        x = box_target - box_pred
        x_abs = tf.abs(x)
        huber_loss = tf.where(x_abs <= delta,
                              0.5 * tf.pow(x_abs, 2),
                              0.5 * tf.pow(delta, 2) + delta * (x_abs - delta))

        return keras.backend.sum(huber_loss)

        # box_target_normalize = tf.divide(box_target, tf.constant([[.1, .1, .2, .2]]))

        # x = box_target_normalize - box_pred
        # x_abs = tf.abs(x)
        # s_l1_loss = tf.where(x_abs > (1. / sigma_sq),
        #                      x_abs - (.5 / sigma_sq),
        #                      tf.pow(x_abs, 2) * sigma_sq * .5)
        #
        #
        # return tf.reduce_sum(s_l1_loss)

    def focal_loss(cls_target, cls_pred):
        alpha_factor = alpha * tf.ones_like(cls_pred)
        alpha_factor = tf.where(tf.equal(cls_target, 1), alpha_factor, 1. - alpha_factor)
        focal_weight = tf.where(tf.equal(cls_target, 1), 1. - cls_pred, cls_pred)
        focal_weight = alpha_factor * tf.pow(focal_weight, gamma)

        bce = keras.backend.binary_crossentropy(cls_target, cls_pred)
        loss = focal_weight * bce

        return tf.reduce_sum(loss)

    cls_preds = tf.clip_by_value(cls_preds, 1e-4, 1-1e-4)
    N = tf.shape(box_preds)[0]
    loss_sum = 0.
    for idx in range(N):
        box_pred = box_preds[idx]
        box_anno = box_annos[idx]
        cls_pred = cls_preds[idx]
        cls_anno = cls_annos[idx]
        cls_anno = tf.one_hot(cls_anno, depth=classes)

        iou = IOU(box_anno, anchor)
        iou_max = tf.reduce_max(iou, axis=-1)
        iou_argmax = tf.argmax(iou, axis=-1)

        pos_idx = tf.where(iou_max >= .5)[..., -1]
        neg_idx = tf.where(iou_max < .4)[..., -1]

        pos_box_idx = tf.gather(iou_argmax, pos_idx)
        pos_box = tf.gather(box_anno, pos_box_idx)
        pos_box_pred = tf.gather(box_pred, pos_idx)
        pos_anchor = tf.gather(anchor, pos_idx)

        pos_cls_target = tf.gather(cls_anno, pos_box_idx)
        pos_cls_pred = tf.gather(cls_pred, pos_idx)
        neg_cls_pred = tf.gather(cls_pred, neg_idx)

        box_loss = 0.
        cls_loss = 0.
        num_pos_box = tf.cast(tf.shape(pos_box)[0], tf.float32)
        if num_pos_box > 0:
            box_loss += smooth_l1_loss(pos_box_pred, pos_box, pos_anchor)
            cls_loss += focal_loss(pos_cls_target, pos_cls_pred)
        cls_loss += focal_loss(tf.zeros_like(neg_cls_pred), neg_cls_pred)

        box_loss = 50. * box_loss / tf.maximum(1., 4 * num_pos_box)
        cls_loss = cls_loss / tf.maximum(1., num_pos_box)
        loss = box_loss + cls_loss
        loss_sum += loss

    return loss_sum / tf.cast(N, tf.float32)


if __name__ == '__main__':
    import numpy as np

    print(ltrb2xywh(np.array([[5, 5, 10, 10]])))
