#!/usr/bin/env python
import cv_bridge
import numpy as np
import rospy
import sensor_msgs.msg
import std_msgs.msg
import tensorflow as tf

from semantic_segmentation_node.msg import SemanticClassDistributions
from semantic_segmentation_node.srv import *

try:
    import tensorflow.contrib.tensorrt as trt
    tf.logging.set_verbosity(tf.logging.INFO)
except tf.errors.NotFoundError:
    trt = None
    rospy.loginfo('Could not import TensorRT, disabling it.')


def segmenter():
    rospy.init_node('semantic_segmenation')

    # Parse parameters.
    ns = rospy.get_name() + '/'
    image_topic = rospy.get_param(ns + 'image_topic', '/image')
    output_topic_rgb = rospy.get_param(ns + 'rgb_topic_out', '/output_rgb')
    output_topic_dist = rospy.get_param(ns + 'dist_topic_out', '/output_dist')
    frozen_model = rospy.get_param(ns + 'frozen_model')

    # The trt_mode parameter can either be None, FP32 or FP16
    trt_mode = rospy.get_param(ns + 'trt_mode', None)
    if trt_mode is not None and trt is None:
        rospy.loginfo('TensorRT was not found ignoring the trt_mode parameter.')
    if trt_mode is not None and trt is not None:
        if trt_mode in ['FP16', 'FP32']:
            rospy.loginfo("Using trt_mode: {}".format(trt_mode))
        else:
            rospy.logerr("Got trt_mode: {}, only 'FP16' and 'FP32' are "
                         "supported".format(trt_mode))
    rospy.loginfo('Listening for images on topic:{}'.format(image_topic))
    rospy.loginfo(
        'Publishing color images on topic:{}'.format(output_topic_rgb))
    rospy.loginfo(
        'Publishing class distributions on topic:{}'.format(output_topic_dist))
    rospy.loginfo('Restoring model from: {}'.format(frozen_model))

    # Load the complete network
    with tf.gfile.GFile(frozen_model, 'rb') as f:
        restored_graph_def = tf.GraphDef()
        restored_graph_def.ParseFromString(f.read())
    if trt_mode is not None and trt is not None:
        restored_graph_def = trt.create_inference_graph(
            input_graph_def=restored_graph_def,
            outputs=['class_colors', 'class_probabilities', 'id_to_rgb',
                     'id_to_class_names'],
            max_batch_size=1,
            # According to the docs this should be several GBs, but the smaller
            # I set it, the faster it compiles, hences it remains at 1 until an
            # actual issues arises.
            max_workspace_size_bytes=1,
            precision_mode=trt_mode)

    with tf.Graph().as_default() as graph:
        input_tensor = tf.placeholder(shape=[1, None, None, 3], dtype=tf.uint8)
        out = tf.import_graph_def(
            graph_def=restored_graph_def,
            input_map={'input': input_tensor},
            return_elements=['class_colors', 'class_probabilities',
                             'id_to_rgb', 'id_to_class_names'])
        output_color = out[0].outputs[0]
        output_dist = out[1].outputs[0]

        # The TensorRT docs suggest to set GPU options here specifying the GPU
        # memory fraction for tensorflow. In my experience nothing changed so
        # To keep the code portable we ignore this for now.
        #     config = tf.ConfigProto(gpu_options=tf.GPUOptions(
        #         per_process_gpu_memory_fraction=0.33))
        sess = tf.Session(graph=graph)

        # Recover the rgb colors and class names for the service that provides
        # this information.
        id_to_rgb, id_to_class_names = sess.run(
            [out[2].outputs[0], out[3].outputs[0]])

    # Setup a publisher for the result images and distributions.
    image_publisher = rospy.Publisher(
        output_topic_rgb, sensor_msgs.msg.Image, queue_size=1)
    dist_publisher = rospy.Publisher(
        output_topic_dist, SemanticClassDistributions, queue_size=1)

    # Configure the listener with a callback.
    bridge = cv_bridge.CvBridge()

    def callback(image):
        # Make sure somebody is subscribed otherwise we save power.
        request = []
        publish_image =  image_publisher.get_num_connections() > 0
        publish_dist = dist_publisher.get_num_connections() > 0
        if publish_image:
            request.append(output_color)
        if publish_dist:
            request.append(output_dist)

        if publish_image or publish_dist:
            # Slightly modify the shape and feed the image through the network.
            cv_image = bridge.imgmsg_to_cv2(image)
            cv_image = cv_image[None]
            result = sess.run(request, feed_dict={input_tensor: cv_image})

        # Publish the result(s)
        if publish_image:
            output_message = bridge.cv2_to_imgmsg(
                result.pop(0)[0], encoding="bgr8")
            output_message.header = image.header
            image_publisher.publish(output_message)

        # This will publish a HxWxC array with the class distribution for
        # every pixel. It is NOT tested w.r.t to the strides and due to
        # massive amount of data, it is very slow.
        if publish_dist:
            result_dist = result.pop(0)[0]
            output_message = SemanticClassDistributions()
            output_dists = std_msgs.msg.Float32MultiArray()
            output_dists.data = result_dist.flatten()
            output_dists.layout = std_msgs.msg.MultiArrayLayout()
            output_dists.layout.dim = [std_msgs.msg.MultiArrayDimension()]*3
            output_dists.layout.dim[0].label = 'height'
            output_dists.layout.dim[0].size = result_dist.shape[0]
            output_dists.layout.dim[0].stride = np.prod(result_dist.shape)
            output_dists.layout.dim[1].label = 'width'
            output_dists.layout.dim[1].size = result_dist.shape[1]
            output_dists.layout.dim[1].stride = np.prod(result_dist.shape[1:])
            output_dists.layout.dim[2].label = 'class'
            output_dists.layout.dim[2].size = result_dist.shape[2]
            output_dists.layout.dim[2].stride = result_dist.shape[2]
            output_message.dist = output_dists
            output_message.header = image.header
            dist_publisher.publish(output_message)

    # Note that the buff_size is set very high here. Apparently there are more
    # than just the queue for the subscriber and if a single message doesn't fit
    # the buffer, images are take from that queue. Having a big enough buff_size
    # though will make sure that only a single image is stored in the queue and
    # thus the image is more "up-to-date".
    rospy.Subscriber(
        image_topic, sensor_msgs.msg.Image, callback, queue_size=1,
        buff_size=100000000)

    # We additionally need to have two services to provide class information.
    def serve_class_colors(req):
        colors = GetClassColorsResponse()
        for c in id_to_rgb:
            colors.class_colors.append(
                std_msgs.msg.ColorRGBA(c[2], c[1], c[0], 255))
        return colors
    rospy.Service('get_class_colors', GetClassColors, serve_class_colors)

    def serve_class_names(req):
        names = GetClassNamesResponse()
        for n in id_to_class_names:
            names.class_names.append(std_msgs.msg.String(n))
        return names
    rospy.Service('get_class_names', GetClassNames, serve_class_names)

    # Run indefinitely
    rospy.spin()

if __name__ == '__main__':
    segmenter()
