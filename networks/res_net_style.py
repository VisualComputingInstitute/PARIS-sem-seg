import tensorflow as tf
import tensorflow.contrib.slim as slim

# ResNet baseline from the FRRN paper with some modifications by Jakob Bauer


# Each ResBlock follows the resnet v2 style
# input -> bn -> relu -> conv -> bn -> relu -> conv + input. Where the last is
# possible passed through a 1x1 conv if channel counts don't match. All convs
# have an input channel count and an possible multiplication factor, which is a
# global network parameter.


def res_block_v2(input, out_channel_count, scope, channel_multiplier=1):
    '''ResNet v2 block

    Args:

    Returns:

    '''
    input_channel_count = input.shape[-1]
    out_channel_count *= channel_multiplier

    with tf.variable_scope(scope):
        if input_channel_count != out_channel_count:
            skip = slim.conv2d(input, out_channel_count, [1,1],
                scope='res_skip1x1_conv')
        else:
            skip = input

        net = slim.batch_norm(input)
        net = tf.nn.relu(net)
        net = slim.conv2d(net, out_channel_count, [3,3], scope='res_conv1')

        net = slim.batch_norm(net)
        net = tf.nn.relu(net)
        net = slim.conv2d(net, out_channel_count, [3,3], scope='res_conv2')

        net = net + skip

    return net


def network(input, is_training, base_channel_count=48):
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

    batch_norm_params = {
        'center': True,
        'scale': True,
        'decay': 0.9,
        'epsilon': 1e-5,
        'is_training': is_training
    }

    with slim.arg_scope([slim.conv2d], **conv2d_params):
        with slim.arg_scope([slim.batch_norm], **batch_norm_params):
            # First convolution to increase the channel count.
            net = slim.conv2d(input, base_channel_count, [3, 3],
                scope='input_conv')

            # 2 ResBlocks, store the output for the skip connection
            net = res_block_v2(net, base_channel_count, channel_multiplier=1, scope='resblock_v2_1')
            net = res_block_v2(net, base_channel_count, channel_multiplier=1, scope='resblock_v2_2')
            skip0 = net

            # Pooling -> 1/2 res
            net = slim.max_pool2d(net, [2, 2], padding='SAME')

            # 3 ResBlocks, store the output for the skip connection
            net = res_block_v2(net, base_channel_count, channel_multiplier=2, scope='resblock_v2_3')
            net = res_block_v2(net, base_channel_count, channel_multiplier=2, scope='resblock_v2_4')
            net = res_block_v2(net, base_channel_count, channel_multiplier=2, scope='resblock_v2_5')
            skip1 = net

            # Pooling -> 1/4 res
            net = slim.max_pool2d(net, [2, 2], padding='SAME')

            # 4 ResBlocks, store the output for the skip connection
            net = res_block_v2(net, base_channel_count, channel_multiplier=4, scope='resblock_v2_6')
            net = res_block_v2(net, base_channel_count, channel_multiplier=4, scope='resblock_v2_7')
            net = res_block_v2(net, base_channel_count, channel_multiplier=4, scope='resblock_v2_8')
            net = res_block_v2(net, base_channel_count, channel_multiplier=4, scope='resblock_v2_9')
            skip2 = net

            # Pooling -> 1/8 res
            net = slim.max_pool2d(net, [2, 2], padding='SAME')

            # 2 ResBlocks, store the output for the skip connection
            net = res_block_v2(net, base_channel_count, channel_multiplier=8, scope='resblock_v2_10')
            net = res_block_v2(net, base_channel_count, channel_multiplier=8, scope='resblock_v2_11')
            skip3 = net

            # Pooling -> 1/16 res
            net = slim.max_pool2d(net, [2, 2], padding='SAME')

            # 2 ResBlocks
            net = res_block_v2(net, base_channel_count, channel_multiplier=8, scope='resblock_v2_12')
            net = res_block_v2(net, base_channel_count, channel_multiplier=8, scope='resblock_v2_13')

            # Unpool, crop and concatenate the skip connection
            net = tf.image.resize_nearest_neighbor(net, [tf.shape(net)[1]*2, tf.shape(net)[2]*2])
            net = net[:, :skip3.shape[1], :skip3.shape[2], :]
            net = tf.concat([net, skip3], axis=-1)

            # 2 ResBlocks, store the output for the skip connection
            net = res_block_v2(net, base_channel_count, channel_multiplier=4, scope='resblock_v2_14')
            net = res_block_v2(net, base_channel_count, channel_multiplier=4, scope='resblock_v2_15')

            # Unpool, crop and concatenate the skip connection
            net = tf.image.resize_nearest_neighbor(net, [tf.shape(net)[1]*2, tf.shape(net)[2]*2])
            net = net[:, :skip2.shape[1], :skip2.shape[2], :]
            net = tf.concat([net, skip2], axis=-1)

            # 2 ResBlocks, store the output for the skip connection
            net = res_block_v2(net, base_channel_count, channel_multiplier=4, scope='resblock_v2_16')
            net = res_block_v2(net, base_channel_count, channel_multiplier=4, scope='resblock_v2_17')

            # Unpool, crop and concatenate the skip connection
            net = tf.image.resize_nearest_neighbor(net, [tf.shape(net)[1]*2, tf.shape(net)[2]*2])
            net = net[:, :skip1.shape[1], :skip1.shape[2], :]
            net = tf.concat([net, skip1], axis=-1)

            # 2 ResBlocks, store the output for the skip connection
            net = res_block_v2(net, base_channel_count, channel_multiplier=2, scope='resblock_v2_18')
            net = res_block_v2(net, base_channel_count, channel_multiplier=2, scope='resblock_v2_19')

            # Unpool, crop and concatenate the skip connection
            net = tf.image.resize_nearest_neighbor(net, [tf.shape(net)[1]*2, tf.shape(net)[2]*2])
            net = net[:, :skip0.shape[1], :skip0.shape[2], :]
            net = tf.concat([net, skip0], axis=-1)

            # 2 ResBlocks, store the output for the skip connection
            net = res_block_v2(net, base_channel_count, channel_multiplier=1, scope='resblock_v2_20')
            net = res_block_v2(net, base_channel_count, channel_multiplier=1, scope='resblock_v2_21')

            # Final batchnorm and relu before the prediction.
            net = slim.batch_norm(net)
            net = tf.nn.relu(net)

            return net

