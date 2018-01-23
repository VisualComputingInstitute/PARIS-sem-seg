import tensorflow as tf
from tensorflow.contrib import slim


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

    # Remove the pixel in height and width.
    begin_h = tf.random_uniform(shape=[], minval=0, maxval=pixel_to_remove[0], dtype=tf.int32)
    begin_w = tf.random_uniform(shape=[], minval=0, maxval=pixel_to_remove[1], dtype=tf.int32)

    end_h = tf.shape(image)[0] - pixel_to_remove[0] + begin_h
    end_w = tf.shape(image)[1] - pixel_to_remove[1] + begin_w

    return (image[begin_h:end_h, begin_w:end_w],
        labels[begin_h:end_h, begin_w:end_w])
