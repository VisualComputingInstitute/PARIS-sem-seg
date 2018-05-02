#!/usr/bin/env python3
from argparse import ArgumentParser
from importlib import import_module
import os
import sys
import time

import cv2
import glob
import json
import numpy as np
import tensorflow as tf
from tensorflow.contrib import slim

import tf_utils
import utils


parser = ArgumentParser(description='Evaluate a semantic segmentation network.')

parser.add_argument(
    '--experiment_root', required=True, type=utils.writeable_directory,
    help='Location used to load checkpoints and store images.')

parser.add_argument(
    '--image_glob', required=True, type=str,
    help='Pattern used to match the images for which the predictions should be '
         'made. For example `/root/folder/frames*.jpg`.')

parser.add_argument(
    '--result_directory', required=True, type=utils.writeable_directory,
    help='Directory in which the results will be saved. The base filename will '
         'be used to save it, where the extension is replaced by .png.')

parser.add_argument(
    '--rescale_h', type=utils.positive_int, default=None,
    help='The intermediate height to which input images will be rescaled.')

parser.add_argument(
    '--rescale_w', type=utils.positive_int, default=None,
    help='The intermediate width to which input images will be rescaled.')

parser.add_argument(
    '--checkpoint_iteration', type=int, default=-1,
    help='Iteration from which the checkpoint will be loaded. Defaults to -1, '
         'which results in the last checkpoint being used.')


def main():
    args = parser.parse_args()

    # Fetch the image glob
    image_filenames = sorted(glob.glob(args.image_glob))

    # Parse original info from the experiment root and add new ones.
    args_file = os.path.join(args.experiment_root, 'args.json')
    if not os.path.isfile(args_file):
        raise IOError('`args.json` not found in {}'.format(args_file))
    print('Loading args from {}.'.format(args_file))
    with open(args_file, 'r') as f:
        args_resumed = json.load(f)
    for key, value in args_resumed.items():
        if key not in args.__dict__:
            args.__dict__[key] = value

    # Load the config for the dataset.
    with open(args.dataset_config, 'r') as f:
        dataset_config = json.load(f)

    # Compute the label to color map
    id_to_rgb = np.asarray(
        dataset_config['rgb_colors'] + [(0, 0, 0)], dtype=np.uint8)[:,::-1]

    # Setup the input
    image_file_tensor = tf.data.Dataset.from_tensor_slices(image_filenames)

    dataset = image_file_tensor.map(
        lambda fn: tf.image.decode_png(tf.read_file(fn), channels=3))

    dataset = dataset.map(lambda x: ((tf.to_float(x) - 128.0) / 128.0))

    dataset = tf.data.Dataset.zip((dataset, image_file_tensor))

    dataset = dataset.batch(1)

    dataset = dataset.prefetch(1)

    image_input, image_filename = dataset.make_one_shot_iterator().get_next()

    if args.rescale_h is not None or args.rescale_w is not None:
        if args.rescale_h is not None and args.rescale_w is not None:
            image_input_resized = tf.image.resize_images(
                image_input, (args.rescale_h, args.rescale_w))
        else:
            raise ValueError('Either both rescale_h and rescale_w should be '
                             'left undefined or both should be set. Got {} and '
                             '{}'.format(args.rescale_h, args.rescale_w))
    else:
        image_input_resized = image_input

    # Setup the network for simple forward passing.
    model = import_module('networks.' + args.model_type)
    with tf.name_scope('model'):
        net = model.network(image_input_resized, is_training=False,
            **args.model_params)
        logits = slim.conv2d(net, len(dataset_config['class_names']),
            [3,3], scope='output_conv', activation_fn=None,
            weights_initializer=slim.variance_scaling_initializer(),
            biases_initializer=tf.zeros_initializer())
        predictions = tf.nn.softmax(logits)

        predictions_full = tf.image.resize_images(
            predictions, tf.shape(image_input)[1:3])

    with tf.Session() as sess:
        # Determine the checkpoint location.
        checkpoint_loader = tf.train.Saver()
        if args.checkpoint_iteration == -1:
            # The default TF way to do this fails when moving folders.
            checkpoint = os.path.join(
                args.experiment_root,
                'checkpoint-{}'.format(args.train_iterations))
        else:
            checkpoint = os.path.join(
                args.experiment_root,
                'checkpoint-{}'.format(args.checkpoint_iteration))
        iteration = int(checkpoint.split('-')[-1])
        print('Restoring from checkpoint: {}'.format(checkpoint))
        checkpoint_loader.restore(sess, checkpoint)

        # Loop over all images
        timings = []
        while True:
            try:
                start = time.time()
                preds, fn = sess.run(
                    [predictions_full, image_filename])
                timings.append(time.time() - start)
                pred_class = np.argmax(preds[0], -1)
                pred_out = id_to_rgb[pred_class]

                in_filename = fn[0].decode("utf-8")
                base_dir = os.path.dirname(in_filename)
                out_filename = in_filename.replace(
                    base_dir, args.result_directory)
                extension = os.path.splitext(out_filename)[1]
                out_filename = out_filename.replace(extension, '.png')
                if not os.path.isdir(args.result_directory):
                    os.makedirs(args.result_directory)
                cv2.imwrite(out_filename, pred_out)

            except tf.errors.OutOfRangeError:
                print()  # Done!
                break

    # For the timings we skip the first frame since this is where Tensorflow
    # hides the compilation time.
    if len(timings) > 1:
        timings = timings[1:]
        print('Time for loading, resizing and forwarding per frame: '
              '{:7.4f}s±{:7.4f}s'.format(np.mean(timings), np.std(timings)))
    else:
        print('Loading and forwarding took {:7.4f}s. '
              'This includes compilation'.format(timings[0]))


if __name__ == '__main__':
    main()
