#!/usr/bin/env python3
from argparse import ArgumentParser
import json
import os
import time

import numpy as np
import tensorflow as tf

import utils

parser = ArgumentParser(description='Time the forward pass of a frozen network.')

parser.add_argument(
    '--experiment_root', required=True, type=utils.writeable_directory,
    help='Location used to load checkpoints and store resulting timings.')

parser.add_argument(
    '--frozen_graph_pb', type=str, default=None,
    help='Optional specification of a specific graph protobuf. If left blank '
         'the last will be used. This assumes a .pb file was created already.')

parser.add_argument(
    '--input_height', type=utils.positive_int, default=256,
    help='Input height used for forwarding.')

parser.add_argument(
    '--input_width', type=utils.positive_int, default=512,
    help='Input width used for forwarding.')

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

    # Determing the pb file to use.
    if args.frozen_graph_pb is None:
        frozen_model = os.path.join(
            args.experiment_root,
            'checkpoint-{}_frozen.pb'.format(args.train_iterations))
    else:
        frozen_model = args.frozen_graph_pb

   # Load the complete network
    with tf.gfile.GFile(frozen_model, 'rb') as f:
        restored_graph_def = tf.GraphDef()
        restored_graph_def.ParseFromString(f.read())

    with tf.Graph().as_default() as graph:
        tf.import_graph_def(restored_graph_def)

        # Fetch the input and output.
        input_tensor = graph.get_tensor_by_name('import/input:0')
        output_probs = graph.get_tensor_by_name('import/class_probabilities:0')

        sess = tf.Session(graph=graph)

    # Do 300 forward passes to "burn in" and the last 200 to measure.
    timings = []
    print()
    for i in range(500):
        random_input = np.random.uniform(0, 255, size=(
            1, args.input_height, args.input_width, 3)).astype(np.uint8)
        start = time.time()
        _ = sess.run(output_probs, feed_dict={input_tensor: random_input})
        timings.append(time.time() - start)
        print('Time for loading, resizing and forwarding per frame:'
                ' {:7.4f}ms±{:7.4f}ms, at iteration:{}'.format(
                    np.mean(timings[-200:])*1000,
                    np.std(timings[-200:])*1000,
                    i),
              end='\r')

    result_file = os.path.join(args.experiment_root, 'results.json')
    try:
        with open(result_file, 'r') as f:
            result_log = json.load(f)
    except (FileNotFoundError, json.JSONDecodeError):
        result_log = {}

    key = '{}x{}'.format(args.input_height, args.input_width)
    result_log[key] = {
            'runtime_ms_mean': np.mean(timings[-200:])*1000,
            'runtime_ms_std': np.std(timings[-200:])*1000,
    }
    with open(result_file, 'w') as f:
        json.dump(result_log, f, ensure_ascii=False, indent=2, sort_keys=True)

if __name__ == '__main__':
    main()
