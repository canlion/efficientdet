import tensorflow as tf


class FocalLoss(tf.keras.losses.Loss):
    def __init__(self, alpha, gamma):
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma

    def call(self, y_true, y_pred):
        y_pred_sigmoid = tf.nn.sigmoid(y_pred)
        p_t = tf.where(tf.equal(y_true, 1.), y_pred_sigmoid, 1 - y_pred_sigmoid)
        mod_factor = (1 - p_t) ** self.gamma
        alpha_factor = tf.where(tf.equal(y_true, 1.), self.alpha, 1-self.alpha)

        bce = tf.nn.sigmoid_cross_entropy_with_logits(y_true, y_pred)
        loss = tf.reduce_sum(alpha_factor * mod_factor * bce)

        return loss


class BoxLoss(tf.keras.losses.Loss):
    def __init__(self, delta):
        super().__init__()
        self.huber = tf.keras.losses.Huber(delta=delta, reduction=tf.keras.losses.Reduction.NONE)

    def call(self, y_true, y_pred):
        mask = tf.cast(y_true != 0., tf.float32)
        y_true = tf.expand_dims(y_true, -1)
        y_pred = tf.expand_dims(y_pred, -1)
        loss = self.huber(y_true, y_pred) * mask
        loss = tf.reduce_sum(loss)
        return loss
#
# class FocalLoss(tf.keras.losses.Loss):
#     """Compute the focal loss between `logits` and the golden `target` values.
#     Focal loss = -(1-pt)^gamma * log(pt)
#     where pt is the probability of being classified to the true class.
#     """
#
#     def __init__(self, alpha, gamma, label_smoothing=0.0, **kwargs):
#         """Initialize focal loss.
#         Args:
#           alpha: A float32 scalar multiplying alpha to the loss from positive
#             examples and (1-alpha) to the loss from negative examples.
#           gamma: A float32 scalar modulating loss from hard and easy examples.
#           label_smoothing: Float in [0, 1]. If > `0` then smooth the labels.
#           **kwargs: other params.
#         """
#         super().__init__(**kwargs)
#         self.alpha = alpha
#         self.gamma = gamma
#         self.label_smoothing = label_smoothing
#
#     @tf.autograph.experimental.do_not_convert
#     def call(self, y, y_pred):
#         """Compute focal loss for y and y_pred.
#         Args:
#           y: A tuple of (normalizer, y_true), where y_true is the target class.
#           y_pred: A float32 tensor [batch, height_in, width_in, num_predictions].
#         Returns:
#           the focal loss.
#         """
#         normalizer, y_true = y
#         alpha = tf.convert_to_tensor(self.alpha, dtype=y_pred.dtype)
#         gamma = tf.convert_to_tensor(self.gamma, dtype=y_pred.dtype)
#
#         # compute focal loss multipliers before label smoothing, such that it will
#         # not blow up the loss.
#         pred_prob = tf.sigmoid(y_pred)
#         p_t = (y_true * pred_prob) + ((1 - y_true) * (1 - pred_prob))
#         alpha_factor = y_true * alpha + (1 - y_true) * (1 - alpha)
#         modulating_factor = (1.0 - p_t)**gamma
#
#         # apply label smoothing for cross_entropy for each entry.
#         y_true = y_true * (1.0 - self.label_smoothing) + 0.5 * self.label_smoothing
#         ce = tf.nn.sigmoid_cross_entropy_with_logits(labels=y_true, logits=y_pred)
#
#         # compute the final loss and return
#         return alpha_factor * modulating_factor * ce / normalizer
#
#
# class BoxLoss(tf.keras.losses.Loss):
#     """L2 box regression loss."""
#
#     def __init__(self, delta=0.1, **kwargs):
#         """Initialize box loss.
#         Args:
#           delta: `float`, the point where the huber loss function changes from a
#             quadratic to linear. It is typically around the mean value of regression
#             target. For instances, the regression targets of 512x512 input with 6
#             anchors on P3-P7 pyramid is about [0.1, 0.1, 0.2, 0.2].
#           **kwargs: other params.
#         """
#         super().__init__(**kwargs)
#         self.huber = tf.keras.losses.Huber(
#             delta, reduction=tf.keras.losses.Reduction.NONE)
#
#     @tf.autograph.experimental.do_not_convert
#     def call(self, y_true, box_outputs):
#         num_positives, box_targets = y_true
#         normalizer = num_positives * 4.0
#         mask = tf.cast(box_targets != 0.0, tf.float32)
#         box_targets = tf.expand_dims(box_targets, axis=-1)
#         box_outputs = tf.expand_dims(box_outputs, axis=-1)
#         box_loss = self.huber(box_targets, box_outputs) * mask
#         box_loss = tf.reduce_sum(box_loss)
#         box_loss /= normalizer
#         return box_loss