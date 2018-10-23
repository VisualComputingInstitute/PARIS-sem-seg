#!/usr/bin/env python3
from argparse import ArgumentParser
from importlib import import_module
import functools
import json
import logging.config
import os
from signal import SIGINT, SIGTERM
import time

import numpy as np
import tensorflow as tf
from tensorflow.contrib import slim

import output_losses
from networks import NETWORK_CHOICES
import tf_utils
import utils


parser = ArgumentParser(description='Train a Semantic Segmentation network.')

parser.add_argument(
    '--experiment_root', required=True,
    type=lambda x: utils.writeable_directory(
        utils.select_existing_root(x, check_only_basedir=True)),
    help='Location used to store checkpoints and dumped data.')

parser.add_argument(
    '--train_set', type=str, nargs='+',
    help='Path to the train_set csv file. Can be multiple sets, but in that '
         'case the train set, dataset root and dataset config counts must match'
         '.')

parser.add_argument(
    '--dataset_root',
    type=lambda x: utils.readable_directory(
        utils.select_existing_root(x, check_only_basedir=True)), nargs='+',
    help='Path that will be pre-pended to the filenames in the train_set csv. '
         'Can be multiple sets, but in that case the train set, dataset root '
         'and dataset config counts must match.')

parser.add_argument(
    '--dataset_config', type=str, nargs='+',
    help='Path to the json file containing the dataset config. Can be multiple '
         'sets, but in that case the train set, dataset root and dataset config'
         ' counts must match.')

parser.add_argument(
    '--dataset_weights', type=float, nargs='+', default=None,
    help='A relative weight when multiple dataets are used. If not specified '
         'the weights are computed based on the dataset sizes.')

parser.add_argument(
    '--resume', action='store_true', default=False,
    help='When this flag is provided, all other arguments apart from the '
         'experiment_root are ignored and a previously saved set of arguments '
         'is loaded.')

parser.add_argument(
    '--auto_resume', action='store_true', default=False,
    help='When this flag is provided, training with either start or continue, '
         'regardless of the fact whether an experiment_root already exists.')

parser.add_argument(
    '--checkpoint_frequency', default=5000, type=utils.nonnegative_int,
    help='After how many iterations a checkpoint is stored. Set this to 0 to '
         'disable intermediate storing. This will result in only one final '
         'checkpoint.')

parser.add_argument(
    '--learning_rate', default=1e-3, type=utils.positive_float,
    help='The initial value of the learning-rate, before the decay kicks in.')

parser.add_argument(
    '--train_iterations', default=70000, type=utils.positive_int,
    help='Number of training iterations.')

parser.add_argument(
    '--decay_start_iteration', default=40000, type=int,
    help='At which iteration the learning-rate decay should kick-in.'
         'Set to -1 to disable decay completely.')

parser.add_argument(
    '--decay_multiplier', default=1e-3, type=float,
    help='How much the exponential decay, should reduce the learning rate.')

parser.add_argument(
    '--batch_size', default=5, type=utils.positive_int,
    help='The batch size used during training.')

parser.add_argument(
    '--loading_threads', default=8, type=utils.positive_int,
    help='Number of threads used for parallel loading.')

parser.add_argument(
    '--model_type', default='res_net_style', choices=NETWORK_CHOICES,
    help='Type of the model to use.')

parser.add_argument(
    '--model_params', default=None, type=str,
    help='Comma separated list of model parameters and values. Valid '
         'parameters differ from model to model.')

parser.add_argument(
    '--loss_type', default='cross_entropy_loss',
    choices=output_losses.LOSS_CHOICES,
    help='Loss used to train the network.')

parser.add_argument(
    '--flip_augment', action='store_true', default=False,
    help='When provided, random flip augmentation is used during training.')

parser.add_argument(
    '--gamma_augment', action='store_true', default=False,
    help='When provided, random gamma augmentation is used during training.')

parser.add_argument(
    '--crop_augment', default=0, type=utils.nonnegative_int,
    help='When not 0, a randomly located crop is taken from the input images, '
         'so that the removed border has the width and height as specified by '
         'this value. This clashes with `fixed_crop_augment_width` and '
         '`fixed_crop_augment_height`.')

parser.add_argument(
    '--fixed_crop_augment_height', default=0, type=utils.nonnegative_int,
    help='Perform a crop augmentation where a crop of fixed height and width '
         'is taken from the input images with the provided height. If this is '
         'used fixed_crop_augment_width also needs to be specified. This '
         'clashes with `crop_augment`.')

parser.add_argument(
    '--fixed_crop_augment_width', default=0, type=utils.nonnegative_int,
    help='Perform a crop augmentation where a crop of fixed height and width '
         'is taken from the input images with the provided width. If this is '
         'used fixed_crop_augment_height also needs to be specified. This '
         'clashes with `crop_augment`.')


# TODO(pandoro): loss parameters


def main():
    args = parser.parse_args()

    # We store all arguments in a json file. This has two advantages:
    # 1. We can always get back and see what exactly that experiment was
    # 2. We can resume an experiment as-is without needing to remember flags.
    if args.resume or args.auto_resume:
        args.experiment_root = utils.select_existing_root(args.experiment_root)
        args_file = os.path.join(args.experiment_root, 'args.json')
        if not os.path.isfile(args_file) and not args.auto_resume:
            # We are not auto_resuming and no existing file was found. This is
            # an error.
            raise IOError('`args.json` not found in {}'.format(args_file))
        elif not os.path.isfile(args_file) and args.auto_resume:
            # No existing args file was found, but we are auto resuming, so we
            # just start a new run.
            new_run = True
        else:
            # We found an existing args file, this can just be used.
            new_run = False
            print('Loading args from {}.'.format(args_file))
            with open(args_file, 'r') as f:
                args_resumed = json.load(f)
            args_resumed['resume'] = True  # This would be overwritten.

            # When resuming, we not only want to populate the args object with
            # the values from the file, but we also want to check for some
            # possible conflicts between loaded and given arguments.
            for key, value in args.__dict__.items():
                if key in args_resumed:
                    resumed_value = args_resumed[key]
                    if resumed_value != value:
                        print('Warning: For the argument `{}` we are using the'
                              ' loaded value `{}`. The provided value was `{}`'
                              '.'.format(key, resumed_value, value))
                        args.__dict__[key] = resumed_value
                else:
                    print('Warning: A new argument was added since the last run'
                          ': `{}`. Using the new value: `{}`.'
                          ''.format(key, value))
    else:
        # No resuming requested at all.
        new_run = True

    if new_run:
        # If the experiment directory exists already and we are not auto
        # resuming, we bail in fear.
        args.experiment_root = utils.select_existing_root(
                args.experiment_root, check_only_basedir=True)
        if os.path.exists(args.experiment_root) and not args.auto_resume:
            if os.listdir(args.experiment_root):
                print('The directory {} already exists and is not empty.'
                      ' If you want to resume training, append --resume or '
                      ' --auto_resume to your call.'
                      ''.format(args.experiment_root))
                exit(1)
        elif os.path.exists(args.experiment_root) and args.auto_resume:
            # If we are auto resuming, it is okay if the directory exists.
            pass
        else:
            # We create a new one if it does not exist.
            os.makedirs(args.experiment_root)
        args_file = os.path.join(args.experiment_root, 'args.json')


        # Make sure the required arguments are provided:
        # train_set, dataset_root, dataset_config
        if not args.train_set:
            parser.print_help()
            print('You did not specify the `train_set` argument!')
            exit(1)
        if not args.dataset_root:
            parser.print_help()
            print('You did not specify the required `dataset_root` argument!')
            exit(1)
        if not args.dataset_config:
            parser.print_help()
            print('You did not specify the required `dataset_config` argument!')
            exit(1)

        # Since multiple datasets can be used, we need to check that the
        # we got lists of the same length
        train_set_len = len(args.train_set)
        dataset_root_len = len(args.dataset_config)
        dataset_config_len = len(args.dataset_config)
        if args.dataset_weights is not None:
            dataset_weight_len = len(args.dataset_weights)
        else:
            # We'll set this manually later so just use a valid length here.
            dataset_weight_len = dataset_config_len

        if (train_set_len != dataset_root_len or
                train_set_len != dataset_config_len or
                train_set_len != dataset_weight_len):
            parser.print_help()
            print('The dataset specific argument lengths didn\'t match.')
            exit(1)


        # Parse the model parameters. This could be a bit cleaner in the future,
        # but it will do for now.
        if args.model_params is not None:
            #model_params = args.model_params.split(';')
            #if len(model_params) % 2 != 0:
            #    raise ValueError('`model_params` has to be a comma separated '
            #                     'list of even length.')
            #it = iter(model_params)
            #args.model_params = {p: eval(v) for p, v in zip(it,it)}
            args.model_params = eval(args.model_params)
        else:
            args.model_params = {}

        # Check some parameter clashes.
        if args.crop_augment > 0 and (args.fixed_crop_augment_width > 0 or
                                      args.fixed_crop_augment_height > 0):
            print('You cannot specified the use of both types of crop '
                  'augmentations. Either use the `crop_augment` argument to '
                  'remove a fixed amount of pixel from the borders, or use the '
                  '`fixed_crop_augment_height` arguments to provide a fixed '
                  'size window that will be cropped from the input images.')
            exit(1)
        if ((args.fixed_crop_augment_height > 0) !=
                (args.fixed_crop_augment_width > 0)):
            print('You need to specify both the `fixed_crop_augment_width` and '
                  '`fixed_crop_augment_height` arguments for a valid '
                  'augmentation.')
            exit(1)

        # Store the passed arguments for later resuming and grepping in a nice
        # and readable format.
        with open(args_file, 'w') as f:
            # Make sure not to store the auto_resume forever though.
            if 'auto_resume' in args.__dict__:
                del args.__dict__['auto_resume']
            json.dump(
                vars(args), f, ensure_ascii=False, indent=2, sort_keys=True)

    log_file = os.path.join(args.experiment_root, 'train')
    logging.config.dictConfig(utils.get_logging_dict(log_file))
    log = logging.getLogger('train')

    # Also show all parameter values at the start, for ease of reading logs.
    log.info('Training using the following parameters:')
    for key, value in sorted(vars(args).items()):
        log.info('{}: {}'.format(key, value))


    # Preload all the filenames and mappings.
    file_lists = []
    dataset_configs = []
    for i, (train_set, dataset_root, config) in enumerate(
            zip(args.train_set, args.dataset_root, args.dataset_config)):

        # Load the config for the dataset.
        with open(config, 'r') as f:
            dataset_configs.append(json.load(f))
        log.info('Training set {} based on a `{}` configuration.'.format(
            i, dataset_configs[-1]['dataset_name']))

        # Load the data from the CSV file.
        file_list = utils.load_dataset(train_set, dataset_root)
        file_lists.append(file_list)

    # if not None set based on size
    if args.dataset_weights is None:
        dataset_weights = [len(fl) for fl in file_lists]
    else:
        dataset_weights = args.dataset_weights

    # In order to keep the loading of images in tensorflow, we need to make some
    # quite ugly hacks where we merge all the dataset original to train mappings
    # into one tensor. Not nice but working.
    mappings = [d.get('original_to_train_mapping') for d in dataset_configs]
    mapping = np.zeros(
        (len(mappings), np.max([len(m) for m in mappings])), dtype=np.int32)
    for i, m in enumerate(mappings):
        mapping[i, :len(m)] = m
    original_to_train_mapping = tf.constant(mapping)

    dataset = tf.data.Dataset.from_generator(
        generator=functools.partial(
            utils.mixed_dataset_generator, file_lists, dataset_weights
        ),
        output_types=(tf.string, tf.string, tf.int32))

    # Convert filenames to actual image and label id tensors.
    dataset = dataset.map(
        lambda x, y, z: tf_utils.string_tuple_to_image_pair(
            x, y, tf.gather(original_to_train_mapping, z)) + (z,),
        num_parallel_calls=args.loading_threads)

    # Possible augmentations
    if args.flip_augment:
        dataset = dataset.map(
            lambda x, y, z: tf_utils.flip_augment(x, y) + (z,))
    if args.gamma_augment:
        dataset = dataset.map(
            lambda x, y, z: tf_utils.gamma_augment(x, y) + (z,))

    # TODO deprecate this. It doesn't file with many datasets. This needs to go.
    if args.crop_augment > 0:
        dataset = dataset.map(
            lambda x, y, z: tf_utils.crop_augment(
                x, y, args.crop_augment, args.crop_augment) + (z,))
    # TODO end

    if args.fixed_crop_augment_width > 0 and args.fixed_crop_augment_height > 0:
        dataset = dataset.map(
            lambda x, y, z: tf_utils.fixed_crop_augment(
                x, y, args.fixed_crop_augment_height,
                args.fixed_crop_augment_width) + (z,))

    # Re scale the input images
    dataset = dataset.map(lambda x, y, z: ((x - 128.0) / 128.0, y, z))

    # Group it into batches.
    dataset = dataset.batch(args.batch_size)

    # Overlap producing and consuming for parallelism.
    dataset = dataset.prefetch(1)

    # Since we repeat the data infinitely, we only need a one-shot iterator.
    image_batch, label_batch, dataset_ids = (
        dataset.make_one_shot_iterator().get_next())

    # This needs a fixed shape.
    dataset_ids.set_shape([args.batch_size])

    # Feed the image through a model.
    model = import_module('networks.' + args.model_type)
    with tf.name_scope('model'):
        net = model.network(image_batch, is_training=True, **args.model_params)

    # Generate a logit for every dataset.
    with tf.name_scope('logits'):
        logits = []
        for d in dataset_configs:
            logits.append(slim.conv2d(
                net, len(d['class_names']),[3,3],
                scope='output_conv_{}'.format(d['dataset_name']),
                activation_fn=None,
                weights_initializer=slim.variance_scaling_initializer(),
                biases_initializer=tf.zeros_initializer()))

    # Create the loss for every dataset.
    with tf.name_scope('losses'):
        loss_function = getattr(output_losses, args.loss_type)
        weighted_losses = []
        for i, dataset_config in enumerate(dataset_configs):
            mask = tf.equal(dataset_ids, i)
            weight = tf.cast(tf.reduce_sum(tf.cast(mask, tf.int32)), tf.float32)
            logit_subset = tf.boolean_mask(logits[i], mask)
            label_subset = tf.boolean_mask(label_batch, mask)

            # Do not evaluate the loss for those datasets without images in the
            # batch.
            zero_mask = tf.equal(weight, 0)
            loss = tf.cond(
                zero_mask,
                lambda: 0.0,
                lambda: tf.reduce_mean(
                    loss_function(logit_subset, label_subset,
                                  void=dataset_config['void_label'])))

            # Normalize with prior
            # loss = tf.divide(
            #    loss, tf.log(float(len(dataset_config['class_names']))))

            summary_loss = tf.cond(zero_mask, lambda: np.nan, lambda: loss)

            tf.summary.scalar(
                'loss_{}'.format(dataset_config['dataset_name']), summary_loss)
            tf.summary.scalar(
                'weight_{}'.format(dataset_config['dataset_name']), weight)

            weighted_losses.append(tf.multiply(loss, weight))

    # Merge all the losses together based on how frequent the underlying
    # datasets are in this batch.
    loss_mean = tf.divide(tf.add_n(weighted_losses), args.batch_size)

    # Some logging for tensorboard.
    tf.summary.scalar('loss', loss_mean)

    # Define the optimizer and the learning-rate schedule.
    # Unfortunately, we get NaNs if we don't handle no-decay separately.
    global_step = tf.Variable(0, name='global_step', trainable=False)
    if 0 <= args.decay_start_iteration < args.train_iterations:
        learning_rate = tf.train.exponential_decay(
            args.learning_rate,
            tf.maximum(0, global_step - args.decay_start_iteration),
            args.train_iterations - args.decay_start_iteration,
            args.decay_multiplier)
    else:
        learning_rate = args.learning_rate
    tf.summary.scalar('learning_rate', learning_rate)
    optimizer = tf.train.AdamOptimizer(learning_rate)

    # Update_ops are used to update batchnorm stats.
    with tf.control_dependencies(tf.get_collection(tf.GraphKeys.UPDATE_OPS)):
        train_op = optimizer.minimize(loss_mean, global_step=global_step)

    # Define a saver for the complete model.
    checkpoint_saver = tf.train.Saver(max_to_keep=0)

    with tf.Session() as sess:
        if args.resume:
            # In case we're resuming, simply load the full checkpoint to init.
            last_checkpoint = tf.train.latest_checkpoint(args.experiment_root)
            log.info('Restoring from checkpoint: {}'.format(last_checkpoint))
            checkpoint_saver.restore(sess, last_checkpoint)
        else:
            # Initialize all variables
            sess.run(tf.global_variables_initializer())

            # We also store this initialization as a checkpoint, such that we
            # could run exactly reproducible experiments.
            checkpoint_saver.save(sess, os.path.join(
                args.experiment_root, 'checkpoint'), global_step=0)

        merged_summary = tf.summary.merge_all()
        summary_writer = tf.summary.FileWriter(args.experiment_root, sess.graph)

        start_step = sess.run(global_step)
        log.info('Starting training from iteration {}.'.format(start_step))

        # Finally, here comes the main-loop. This `Uninterrupt` is a handy
        # utility such that an iteration still finishes on Ctrl+C and we can
        # stop the training cleanly.
        with utils.Uninterrupt(sigs=[SIGINT, SIGTERM], verbose=True) as u:
            for i in range(start_step, args.train_iterations):

                # Compute gradients, update weights, store logs!
                start_time = time.time()
                _, summary, step = sess.run(
                    [train_op, merged_summary, global_step])
                elapsed_time = time.time() - start_time

                # Compute the iteration speed and add it to the summary.
                # We did observe some weird spikes that we couldn't track down.
                summary2 = tf.Summary()
                summary2.value.add(
                    tag='secs_per_iter', simple_value=elapsed_time)
                summary_writer.add_summary(summary2, step)
                summary_writer.add_summary(summary, step)

                # Save a checkpoint of training every so often.
                if (args.checkpoint_frequency > 0 and
                        step % args.checkpoint_frequency == 0):
                    checkpoint_saver.save(sess, os.path.join(
                        args.experiment_root, 'checkpoint'), global_step=step)

                # Stop the main-loop at the end of the step, if requested.
                if u.interrupted:
                    log.info('Interrupted on request!')
                    break

        # Store one final checkpoint. This might be redundant, but it is crucial
        # in case intermediate storing was disabled and it saves a checkpoint
        # when the process was interrupted.
        checkpoint_saver.save(sess, os.path.join(
            args.experiment_root, 'checkpoint'), global_step=step)


if __name__ == '__main__':
    main()
