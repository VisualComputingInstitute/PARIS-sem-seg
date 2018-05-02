import tensorflow as tf
import tensorflow.contrib.slim as slim

import tf_utils

# ResNet baseline from the FRRN paper with some modifications by Jakob Bauer


# Each ResBlock follows the resnet v2 style
# input -> bn -> relu -> conv -> bn -> relu -> conv + input. Where the last is
# possible passed through a 1x1 conv if channel counts don't match. All convs
# have an input channel count and an possible multiplication factor, which is a
# global network parameter.


def res_block_v2(input, out_channel_count, conv_op, norm_op, scope,
                 channel_multiplier=1, bottleneck=False):
    '''ResNet v2 block

    Args:

    Returns:

    '''
    input_channel_count = input.shape[-1]
    out_channel_count *= channel_multiplier

    with tf.variable_scope(scope):
        if input_channel_count != out_channel_count:
            skip = conv_op(input, out_channel_count, [1,1],
                scope='res_skip1x1_conv')
        else:
            skip = input

        if not bottleneck:
            net = norm_op(input)
            net = tf.nn.relu(net)
            net = conv_op(net, out_channel_count, [3,3], scope='res_conv1')

            net = norm_op(net)
            net = tf.nn.relu(net)
            net = conv_op(net, out_channel_count, [3,3], scope='res_conv2')
        else:
            net = norm_op(input)
            net = tf.nn.relu(net)
            net = conv_op(
                net, out_channel_count / 4, [1,1], scope='res_conv1')

            net = norm_op(net)
            net = tf.nn.relu(net)
            net = conv_op(
                net, out_channel_count / 4, [3,3], scope='res_conv2')

            net = norm_op(net)
            net = tf.nn.relu(net)
            net = conv_op(net, out_channel_count, [1,1], scope='res_conv3')
        net = net + skip

    return net


def network(input, is_training, base_channel_count=48, bottleneck_blocks=False,
            separable_conv=False, gn_groups=None, gn_channels=None):
    '''ResNet v2 style semantic segmentation network with long range skips.

    Args:

    Returns:

    '''
    conv2d_params = {
        'padding': 'SAME',
        'weights_initializer': slim.variance_scaling_initializer(),
        'biases_initializer': None,
        'activation_fn': None,
        'normalizer_fn': None
    }

    if gn_groups is not None or gn_channels is not None:
        normalziation_params = {
            'group_count': gn_groups,
            'channel_count': gn_channels
        }
        norm_op = tf_utils.group_normalization
    else:
        normalziation_params = {
            'center': True,
            'scale': True,
            'decay': 0.9,
            'epsilon': 1e-5,
            'is_training': is_training
        }
        norm_op = slim.batch_norm

    if separable_conv:
        separable_conv2d_params = dict(conv2d_params)
        separable_conv2d_params['depth_multiplier'] = 1
        conv_op = slim.separable_conv2d
    else:
        separable_conv2d_params = {}
        conv_op = slim.conv2d

    with slim.arg_scope([slim.conv2d], **conv2d_params):
        with slim.arg_scope([slim.separable_conv2d], **separable_conv2d_params):
            with slim.arg_scope([norm_op], **normalziation_params):
                # First convolution to increase the channel count.
                net = slim.conv2d(input, base_channel_count, [3, 3],
                                  scope='input_conv')

                # 2 ResBlocks, store the output for the skip connection
                net = res_block_v2(
                    net, base_channel_count, conv_op, norm_op,
                    channel_multiplier=1, scope='resblock_v2_1',
                    bottleneck=bottleneck_blocks)
                net = res_block_v2(
                    net, base_channel_count, conv_op, norm_op,
                    channel_multiplier=1, scope='resblock_v2_2',
                    bottleneck=bottleneck_blocks)
                skip0 = net

                # Pooling -> 1/2 res
                net = slim.max_pool2d(net, [2, 2], padding='SAME')

                # 3 ResBlocks, store the output for the skip connection
                net = res_block_v2(
                    net, base_channel_count, conv_op, norm_op,
                    channel_multiplier=2, scope='resblock_v2_3',
                    bottleneck=bottleneck_blocks)
                net = res_block_v2(
                    net, base_channel_count, conv_op, norm_op,
                    channel_multiplier=2, scope='resblock_v2_4',
                    bottleneck=bottleneck_blocks)
                net = res_block_v2(
                    net, base_channel_count, conv_op, norm_op,
                    channel_multiplier=2, scope='resblock_v2_5',
                    bottleneck=bottleneck_blocks)
                skip1 = net

                # Pooling -> 1/4 res
                net = slim.max_pool2d(net, [2, 2], padding='SAME')

                # 4 ResBlocks, store the output for the skip connection
                net = res_block_v2(
                    net, base_channel_count, conv_op, norm_op,
                    channel_multiplier=4, scope='resblock_v2_6',
                    bottleneck=bottleneck_blocks)
                net = res_block_v2(
                    net, base_channel_count, conv_op, norm_op,
                    channel_multiplier=4, scope='resblock_v2_7',
                    bottleneck=bottleneck_blocks)
                net = res_block_v2(
                    net, base_channel_count, conv_op, norm_op,
                    channel_multiplier=4, scope='resblock_v2_8',
                    bottleneck=bottleneck_blocks)
                net = res_block_v2(
                    net, base_channel_count, conv_op, norm_op,
                    channel_multiplier=4, scope='resblock_v2_9',
                    bottleneck=bottleneck_blocks)
                skip2 = net

                # Pooling -> 1/8 res
                net = slim.max_pool2d(net, [2, 2], padding='SAME')

                # 2 ResBlocks, store the output for the skip connection
                net = res_block_v2(
                    net, base_channel_count, conv_op, norm_op,
                    channel_multiplier=8, scope='resblock_v2_10',
                    bottleneck=bottleneck_blocks)
                net = res_block_v2(
                    net, base_channel_count, conv_op, norm_op,
                    channel_multiplier=8, scope='resblock_v2_11',
                    bottleneck=bottleneck_blocks)
                skip3 = net

                # Pooling -> 1/16 res
                net = slim.max_pool2d(net, [2, 2], padding='SAME')

                # 2 ResBlocks
                net = res_block_v2(
                    net, base_channel_count, conv_op, norm_op,
                    channel_multiplier=8, scope='resblock_v2_12',
                    bottleneck=bottleneck_blocks)
                net = res_block_v2(
                    net, base_channel_count, conv_op, norm_op,
                    channel_multiplier=8, scope='resblock_v2_13',
                    bottleneck=bottleneck_blocks)

                # Unpool, crop and concatenate the skip connection
                net = tf.image.resize_nearest_neighbor(
                    net, [tf.shape(net)[1]*2, tf.shape(net)[2]*2])
                net = net[:, :skip3.shape[1], :skip3.shape[2], :]
                net = tf.concat([net, skip3], axis=-1)

                # 2 ResBlocks, store the output for the skip connection
                net = res_block_v2(
                    net, base_channel_count, conv_op, norm_op,
                    channel_multiplier=4, scope='resblock_v2_14',
                    bottleneck=bottleneck_blocks)
                net = res_block_v2(
                    net, base_channel_count, conv_op, norm_op,
                    channel_multiplier=4, scope='resblock_v2_15',
                    bottleneck=bottleneck_blocks)

                # Unpool, crop and concatenate the skip connection
                net = tf.image.resize_nearest_neighbor(
                    net, [tf.shape(net)[1]*2, tf.shape(net)[2]*2])
                net = net[:, :skip2.shape[1], :skip2.shape[2], :]
                net = tf.concat([net, skip2], axis=-1)

                # 2 ResBlocks, store the output for the skip connection
                net = res_block_v2(
                    net, base_channel_count, conv_op, norm_op,
                    channel_multiplier=4, scope='resblock_v2_16',
                    bottleneck=bottleneck_blocks)
                net = res_block_v2(
                    net, base_channel_count, conv_op, norm_op,
                    channel_multiplier=4, scope='resblock_v2_17',
                    bottleneck=bottleneck_blocks)

                # Unpool, crop and concatenate the skip connection
                net = tf.image.resize_nearest_neighbor(
                    net, [tf.shape(net)[1]*2, tf.shape(net)[2]*2])
                net = net[:, :skip1.shape[1], :skip1.shape[2], :]
                net = tf.concat([net, skip1], axis=-1)

                # 2 ResBlocks, store the output for the skip connection
                net = res_block_v2(
                    net, base_channel_count, conv_op, norm_op,
                    channel_multiplier=2, scope='resblock_v2_18',
                    bottleneck=bottleneck_blocks)
                net = res_block_v2(
                    net, base_channel_count, conv_op, norm_op,
                    channel_multiplier=2, scope='resblock_v2_19',
                    bottleneck=bottleneck_blocks)

                # Unpool, crop and concatenate the skip connection
                net = tf.image.resize_nearest_neighbor(
                    net, [tf.shape(net)[1]*2, tf.shape(net)[2]*2])
                net = net[:, :skip0.shape[1], :skip0.shape[2], :]
                net = tf.concat([net, skip0], axis=-1)

                # 2 ResBlocks, store the output for the skip connection
                net = res_block_v2(
                    net, base_channel_count, conv_op, norm_op,
                    channel_multiplier=1, scope='resblock_v2_20',
                    bottleneck=bottleneck_blocks)
                net = res_block_v2(
                    net, base_channel_count, conv_op, norm_op,
                    channel_multiplier=1, scope='resblock_v2_21',
                    bottleneck=bottleneck_blocks)

                # Final batchnorm and relu before the prediction.
                net = slim.batch_norm(net)
                net = tf.nn.relu(net)

                return net

