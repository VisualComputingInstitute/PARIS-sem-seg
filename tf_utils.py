import tensorflow as tf
from tensorflow.contrib import slim
from tensorflow.contrib.framework.python.ops import add_arg_scope


def string_tuple_to_image_pair(image_file, label_file, label_mapping=None):
    image_encoded = tf.read_file(image_file)
    image_decoded = tf.to_float(tf.image.decode_png(image_encoded, channels=3))

    labels_encoded = tf.read_file(label_file)
    labels_decoded = tf.cast(
        tf.image.decode_png(labels_encoded, channels=1), tf.int32)

    if label_mapping is not None:
        label_mapping = tf.constant(label_mapping, dtype=tf.int32)
        labels_decoded = tf.gather(label_mapping, labels_decoded)

    return image_decoded, labels_decoded


def flip_augment(image, labels):
    flip = tf.random_uniform((1,))[0]
    image, labels = tf.cond(
        flip < 0.5,
        lambda: (image, labels),
        lambda: (tf.reverse(image, [1]), tf.reverse(labels, [1])))
    return image, labels


def gamma_augment(image, labels, gamma_range=0.1):
    # See Full-Resolution Residual Networks for Semantic Segmentation in Street Scenes
    # CVPR'17 for a discussion on this.
    scaled_image = image / 255.0
    factor = tf.random_uniform(
        shape=[], minval=-gamma_range, maxval=gamma_range, dtype=tf.float32)
    gamma = (tf.log(0.5 + 1.0 / tf.sqrt(2.0) * factor) /
        tf.log(0.5 - 1.0 / tf.sqrt(2.0) * factor))
    image = tf.pow(scaled_image, gamma) * 255.0
    return image, labels


def crop_augment(image, labels, pixel_to_remove):
    if not isinstance(pixel_to_remove, tuple):
        pixel_to_remove = (pixel_to_remove, pixel_to_remove)

    # Compute the corners.
    begin_h = tf.random_uniform(shape=[], minval=0, maxval=pixel_to_remove[0], dtype=tf.int32)
    begin_w = tf.random_uniform(shape=[], minval=0, maxval=pixel_to_remove[1], dtype=tf.int32)
    end_h = tf.shape(image)[0] - pixel_to_remove[0] + begin_h
    end_w = tf.shape(image)[1] - pixel_to_remove[1] + begin_w

    # Compute the new width.
    h = image.get_shape()[0] - pixel_to_remove[0]
    w = image.get_shape()[1] - pixel_to_remove[1]

    # Actually cut out the crop and fix the shapes.
    image = image[begin_h:end_h, begin_w:end_w]
    image.set_shape([h, w, 3])
    labels = labels[begin_h:end_h, begin_w:end_w]
    labels.set_shape([h, w, 1])

    return image, labels


@add_arg_scope
def group_normalization(input, group_count=None, channel_count=None,
                        epsilon=1e-5, scope=None):
    # Check that the provided parameters make sense.
    if group_count is not None and channel_count is not None:
        raise ValueError('You cannot specify both the group and channel count '
                         'for group normalization.')

    if group_count is None and channel_count is None:
        raise ValueError('You have to specify either the group or the channel '
                         'count for group normalization.')

    # Check that the number of channels can be divided as specified.
    # Here we need the static shape to do actual computations with.
    C = input.shape[-1].value
    if group_count is not None:
        if C % group_count:
            raise ValueError(
                'An input channel count of {} cannot be divided into {} groups.'
                ''.format(C, group_count))
        else:
            groups = group_count
            channels = C // group_count
    else:
        if C % channel_count:
            raise ValueError(
                'An input channel count of {} cannot be divided into groups of '
                '{} channels.'.format(C, channel_count))
        else:
            groups = C // channel_count
            channels = channel_count

    with tf.variable_scope(scope, 'group_normalization'):
        # This implements Group Normalization as introduced in:
        #   "Group Normalization", Yuxin Wu, Kaiming He
        #   https://arxiv.org/abs/1803.08494.
        # For reshaping we need the dynamic shapes. This is an important detail
        # done wrong in the original code snippet and the two implementations I
        # found in GitHub. When using the static shape the dimensions need to be
        # fully specified which doesn't make any sense for dynamic image and/or
        # batch sizes.
        # This also seems to be the case for the TF implementation found in
        # tf contrib.
        # However, this implementation also seems to seriously benefit from
        # having a fixed input and batch size at which point it is approximately
        # as fast as the TF contrib implementation. However, in turn this can
        # be used without fixed input sizes during deployment.
        # In any case they are both seriously slow compared to vanilla batch
        # normalization, easily increasing training time by a factor of two :(ZZ

        N = tf.shape(input)[0]
        H = tf.shape(input)[1]
        W = tf.shape(input)[2]
        grouped = tf.reshape(input, [N, H, W, channels, groups])

        # Compute the group statistics.
        mean, var = tf.nn.moments(grouped, [1, 2, 3], keep_dims=True)

        # Reshape them so that they can first me multiplied together
        # with gamma and beta, before applying them to the input.
        mean = tf.tile(mean, [1, 1, 1, 1, channels])
        mean = tf.squeeze(mean, -2)
        var = tf.tile(var, [1, 1, 1, 1, channels])
        var = tf.squeeze(var, -2)

        # Setup the scale and offset parameters
        gamma = tf.get_variable(
            'gamma', [1, 1, 1, C], initializer=tf.constant_initializer(1.0))
        beta = tf.get_variable(
            'beta', [1, 1, 1, C], initializer=tf.constant_initializer(0.0))

        inv_std = tf.rsqrt(var + epsilon)
        gamma = gamma * inv_std
        beta = beta - mean * gamma

        return input * gamma + beta
