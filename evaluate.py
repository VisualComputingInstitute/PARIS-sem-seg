#!/usr/bin/env python3
from argparse import ArgumentParser
from importlib import import_module
from itertools import count
import os
import sys

import cv2
import json
import numpy as np
import tensorflow as tf
from tensorflow.contrib import slim

import confusion
import tf_utils
import utils


parser = ArgumentParser(description='Evaluate a semantic segmentation network.')

parser.add_argument(
    '--experiment_root', required=True, type=utils.writeable_directory,
    help='Location used to load checkpoints and store images.')

parser.add_argument(
    '--eval_set', required=True, type=str,
    help='Path to the validation set csv file.')

parser.add_argument(
    '--rgb_input_root', required=True, type=utils.readable_directory,
    help='Path that will be pre-pended to the RGB image filenames in the '
         'eval_set csv.')

parser.add_argument(
    '--full_res_label_root', required=True, type=utils.readable_directory,
    help='Path that will be pre-pended to the label image filenames in the '
         'eval_set csv. For a correct evaluation this should point to the '
         'full resolution images.')

parser.add_argument(
    '--save_predictions', type=str, default='none',
    choices=['none', 'full', 'out', 'full_id', 'out_id'],
    help='Whether to save color coded predictions and at which resolution. By '
         'default `none` means no predictions are stored. `full` results in '
         'full resolution predictions and `out` in network output resolution '
         'predictions being stored. Appending `_id` results in gray scale '
         'images encoding the label id.')

parser.add_argument(
    '--checkpoint_iteration', type=int, default=-1,
    help='Iteration from which the checkpoint will be loaded. Defaults to -1, '
         'which results in the last checkpoint being used.')

parser.add_argument(
    '--batch_size', type=utils.positive_int, default=10,
    help='Batch size used during forward passes.')

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
    id_to_rgb = np.asarray(
        dataset_config['rgb_colors'] + [(0, 0, 0)], dtype=np.uint8)[:,::-1]

    # If we map from original labels to train labels we have to invert this.
    original_to_train_mapping = dataset_config.get(
        'original_to_train_mapping', None)
    if original_to_train_mapping is None:
        # This results in an identity mapping.
        train_to_label_id = np.arange(len(id_to_rgb)-1, dtype=np.uint8)
    else:
        train_to_label_id = np.arange(len(id_to_rgb)-1, dtype=np.uint8)
        for label_id, label_train in enumerate(original_to_train_mapping):
            if label_train != -1:
                train_to_label_id[label_train] = label_id

    # Setup the input data.
    image_files, label_files = utils.load_dataset(
        args.eval_set, args.rgb_input_root, args.full_res_label_root)

    images = tf.data.Dataset.from_tensor_slices(image_files)
    labels = tf.data.Dataset.from_tensor_slices(label_files)
    dataset = tf.data.Dataset.zip((images, labels))

    dataset = dataset.map(
        lambda x,y: tf_utils.string_tuple_to_image_pair(
            x, y, original_to_train_mapping),
        num_parallel_calls=args.loading_threads)
    dataset = tf.data.Dataset.zip((dataset, labels))

    # Scale the input images
    dataset = dataset.map(lambda x, y: (((x[0] - 128.0) / 128.0, x[1]), y))

    dataset = dataset.batch(args.batch_size)

    # Overlap producing and consuming for parallelism.
    dataset = dataset.prefetch(1)

    # Since we repeat the data infinitely, we only need a one-shot iterator.
    (image_batch, label_batch), label_name_batch = dataset.make_one_shot_iterator().get_next()

    # Setup the network.
    model = import_module('networks.' + args.model_type)
    with tf.name_scope('model'):
        net = model.network(image_batch, is_training=False)
        logits = slim.conv2d(net, len(dataset_config['class_names']),
            [3,3], scope='output_conv', activation_fn=None,
            weights_initializer=slim.variance_scaling_initializer(),
            biases_initializer=tf.zeros_initializer())
        predictions = tf.nn.softmax(logits)

    with tf.Session() as sess:
        # Determine the checkpoint location.
        checkpoint_loader = tf.train.Saver()
        if args.checkpoint_iteration == -1:
             checkpoint = tf.train.latest_checkpoint(args.experiment_root)
        else:
            checkpoint = os.path.join(
                args.experiment_root,
                'checkpoint-{}'.format(args.checkpoint_iteration))
        iteration = int(checkpoint.split('-')[-1])
        print('Restoring from checkpoint: {}'.format(checkpoint))
        checkpoint_loader.restore(sess, checkpoint)

        # Setup storage if needed.
        result_directory = os.path.join(
            args.experiment_root, 'results-{}'.format(iteration))
        if (not os.path.isdir(result_directory) and
                args.save_predictions is not 'none'):
            os.makedirs(result_directory)

        # Initialize the evaluation.
        evaluation = confusion.Confusion(dataset_config['class_names'])

        # Loop over image batches.
        for start_idx in count(step=args.batch_size):
            try:
                print(
                    '\rEvaluating batch {}-{}/{}'.format(
                        start_idx, start_idx + args.batch_size,
                        len(image_files)), flush=True, end='')
                preds_batch, gt_batch, gt_fn_batch = sess.run(
                    [predictions, label_batch, label_name_batch])
                for pred, gt, gt_fn in zip(preds_batch, gt_batch, gt_fn_batch):
                    # Compute the scores.
                    pred_full = np.argmax(
                        cv2.resize(pred, gt.shape[:2][::-1]), -1)
                    evaluation.incremental_update(gt.squeeze(), pred_full)

                    # Possibly save result images.
                    if args.save_predictions == 'full':
                        pred_out = id_to_rgb[pred_full]

                    if args.save_predictions == 'out':
                        pred_out = id_to_rgb[np.argmax(pred, -1)]

                    if args.save_predictions == 'full_id':
                        pred_out = train_to_label_id[pred_full]

                    if args.save_predictions == 'out_id':
                        pred_out = train_to_label_id[np.argmax(pred, -1)]

                    if args.save_predictions != 'none':
                        out_filename = gt_fn.decode("utf-8").replace(
                            args.full_res_label_root, result_directory)
                        base_dir = os.path.dirname(out_filename)
                        if not os.path.isdir(base_dir):
                            os.makedirs(base_dir)
                        cv2.imwrite(out_filename, pred_out)

            except tf.errors.OutOfRangeError:
                print()  # Done!
                break

    # Print the evaluation.
    evaluation.print_confusion_matrix()

    # Save the results.
    result_file = os.path.join(args.experiment_root, 'results.json')
    try:
        with open(result_file, 'r') as f:
            result_log = json.load(f)
    except (FileNotFoundError, json.JSONDecodeError):
        result_log = {}

    result_log[str(iteration)] = {  # json keys cannot be integers.
        'confusion matrix' : evaluation.confusion_normalized_row.tolist(),
        'iou scores' : evaluation.iou_score.tolist(),
        'class scores' : evaluation.class_score.tolist(),
        'global score' : evaluation.global_score,
        'mean iou score' : evaluation.avg_iou_score,
        'mean class score' : evaluation.avg_score,
    }
    with open(result_file, 'w') as f:
        json.dump(result_log, f, ensure_ascii=False, indent=2, sort_keys=True)

if __name__ == '__main__':
    main()
