#!/usr/bin/env python3
from argparse import ArgumentParser
import glob
from importlib import import_module
import json
import os
import sys
import time

import cv2
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
    '--output_graph', type=str, default=None,
    help='Target file to store the graph protobuf to. If left blank it will be '
         'named based on the checkpoint name.')

parser.add_argument(
    '--checkpoint_iteration', type=int, default=-1,
    help='Iteration from which the checkpoint will be loaded. Defaults to -1, '
         'which results in the last checkpoint being used.')

# TODO specify fixed input sizes here to possibly create an optimized model.

def main():
    args = parser.parse_args()

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
    id_to_rgb = tf.constant(np.asarray(
        dataset_config['rgb_colors'] + [(0, 0, 0)], dtype=np.uint8)[:,::-1])

    # Setup the input
    image_input = tf.placeholder(
        tf.uint8, shape=(None, None, None, 3), name='input')

    image_input = (tf.to_float(image_input) - 128.0) / 128.0

    # Setup the network for simple forward passing.
    model = import_module('networks.' + args.model_type)
    with tf.name_scope('model'):
        net = model.network(image_input, is_training=False,
            **args.model_params)
        logits = slim.conv2d(net, len(dataset_config['class_names']),
            [3,3], scope='output_conv', activation_fn=None,
            weights_initializer=slim.variance_scaling_initializer(),
            biases_initializer=tf.zeros_initializer())
    predictions = tf.nn.softmax(logits, name='class_probabilities')

    # Add a color decoder to create nice color-coded images for this
    # dataset.
    colored_predictions = tf.gather(
        id_to_rgb, tf.argmax(predictions, -1), name='class_colors')

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

        # Possibly fix the output graph.
        if args.output_graph is None:
            output_graph = checkpoint + '_frozen.pb'
        else:
            output_graph = args.output_graph

        # Freeze all variables.
        frozen_graph_def = tf.graph_util.convert_variables_to_constants(
            sess,
            sess.graph_def,
            ['class_probabilities', 'class_colors'])

        with tf.gfile.GFile(output_graph, "wb") as f:
            f.write(frozen_graph_def.SerializeToString())

    print('Frozen graph saved to: {}'.format(output_graph))


if __name__ == '__main__':
    main()
