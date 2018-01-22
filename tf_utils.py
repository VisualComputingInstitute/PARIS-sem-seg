import tensorflow as tf
from tensorflow.contrib import slim


def string_tuple_to_image_pair(image_file, label_file, label_offset):
    image_encoded = tf.read_file(image_file)
    image_decoded = tf.to_float(tf.image.decode_png(image_encoded, channels=3))

    labels_encoded = tf.read_file(label_file)
    labels_decoded = tf.cast(
        tf.image.decode_png(labels_encoded, channels=1), tf.int32)
    labels_decoded = labels_decoded - label_offset

    return image_decoded, labels_decoded

