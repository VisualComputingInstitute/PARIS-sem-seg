#!/usr/bin/env python3
from argparse import ArgumentParser
import glob
from importlib import import_module
import json
import os

import numpy as np
import tensorflow as tf
from tensorflow.contrib import slim

import utils


parser = ArgumentParser(description='Evaluate a semantic segmentation network.')

parser.add_argument(
    '--experiment_root', required=True, type=utils.writeable_directory,
    help='Location used to load checkpoints and store images.')

parser.add_argument(
    '--dataset_config', type=str, default=None,
    help='Path to the json file containing the dataset config. When left blank,'
         'the first dataset used during training is used.')

parser.add_argument(
    '--output_graph', type=str, default=None,
    help='Target file to store the graph protobuf to. If left blank it will be '
         'named based on the checkpoint name.')

parser.add_argument(
    '--checkpoint_iteration', type=int, default=-1,
    help='Iteration from which the checkpoint will be loaded. Defaults to -1, '
         'which results in the last checkpoint being used.')

parser.add_argument(
    '--fixed_input_height', default=None, type=utils.nonnegative_int,
    help='A fixed value for the input height. If specified this will bake it '
         'into the frozen model, possibly increasing the speed.')

parser.add_argument(
    '--fixed_input_width', default=None, type=utils.nonnegative_int,
    help='A fixed value for the input width. If specified this will bake it '
         'into the frozen model, possibly increasing the speed.')

parser.add_argument(
    '--fixed_batch_size', default=None, type=utils.nonnegative_int,
    help='A fixed value for the batch size. If specified this will bake it into'
         ' the frozen model, possibly increasing the speed.')

# TODO(pandoro) generalize to use all output logits instead of a single one.
# This will make the freezing less redundant. Simple make a "default" additional
#  probability and color coding which is a duplicate of one of the dataset, but
# also support all datasets.


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

    # In case no dataset config was specified, we need to fix the argument here
    # since there will be a list of configs.
    if args.dataset_config is None:
        args.dataset_config = args_resumed['dataset_config'][0]

    # Load the config for the dataset.
    with open(args.dataset_config, 'r') as f:
        dataset_config = json.load(f)

    # Compute the label to color map
    id_to_rgb = tf.constant(np.asarray(
        dataset_config['rgb_colors'] + [(0, 0, 0)], dtype=np.uint8)[:,::-1])

    # Setup the input
    image_input = tf.placeholder(
        tf.uint8, shape=(
            args.fixed_batch_size, args.fixed_input_height,
            args.fixed_input_width, 3),
        name='input')

    image_input = (tf.to_float(image_input) - 128.0) / 128.0

    # Determine the checkpoint location.
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

    # Check if the checkpoint contains a specifically named output_conv. This is
    # needed for models trained with the older single dataset code.
    reader = tf.train.NewCheckpointReader(checkpoint)
    var_to_shape_map = reader.get_variable_to_shape_map()
    output_conv_name = 'output_conv_{}'.format(
        dataset_config.get('dataset_name'))
    output_conv_name_found = False

    for k in var_to_shape_map.keys():
        output_conv_name_found = output_conv_name in k
        if output_conv_name_found:
            break

    if not output_conv_name_found:
        print('Warning: An output for the specific dataset could not be found '
              'in the checkpoint. This likely means it\'s an old checkpoint. '
              'Revertig to the old default output name. This could cause '
              'issues if the dataset class count and output classes of the '
              'network do not match.')
        output_conv_name = 'output_conv'

    # Setup the network for simple forward passing.
    model = import_module('networks.' + args.model_type)
    with tf.name_scope('model'):
        net = model.network(image_input, is_training=False,
            **args.model_params)
        logits = slim.conv2d(net, len(dataset_config['class_names']),
            [3,3], scope=output_conv_name, activation_fn=None,
            weights_initializer=slim.variance_scaling_initializer(),
            biases_initializer=tf.zeros_initializer())
    predictions = tf.nn.softmax(logits, name='class_probabilities')

    # Add a color decoder to create nice color-coded images for this
    # dataset.
    colored_predictions = tf.gather(
        id_to_rgb, tf.argmax(predictions, -1), name='class_colors')

    with tf.Session() as sess:
        checkpoint_loader = tf.train.Saver()
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
