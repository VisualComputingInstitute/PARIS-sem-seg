import tensorflow as tf

LOSS_CHOICES = [
    'cross_entropy_loss',
    'bootstrapped_cross_entropy_loss',
    'focal_loss'
]

def cross_entropy_loss(logits, target, void=-1):
    logits_flat = tf.reshape(logits, [-1, logits.shape[-1]])
    target_flat = tf.reshape(target, [-1])
    mask = tf.not_equal(target_flat, void)
    logits_masked = tf.boolean_mask(logits_flat, mask)
    target_masked = tf.boolean_mask(target_flat, mask)
    return tf.nn.sparse_softmax_cross_entropy_with_logits(labels=target_masked, logits=logits_masked)


def bootstrapped_cross_entropy_loss(logits, target, bootstrap_factor=4, void=-1):
    top_count = tf.cast(tf.size(target) / bootstrap_factor, tf.int32)
    losses = cross_entropy_loss(logits, target, void)
    losses, _ = tf.nn.top_k(losses, k=top_count, sorted=False)
    return losses

def focal_loss(logits, target, correction_alpha=1, gamma=2, void=-1):
    losses = cross_entropy_loss(logits, target, void)
    target_probabilities = tf.exp(-losses)
    weight = correction_alpha * tf.pow(1.0 - target_probabilities, gamma)
    return weight * losses