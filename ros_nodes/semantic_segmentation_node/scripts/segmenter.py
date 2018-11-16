#!/usr/bin/env python
import cv_bridge
import rospy
import sensor_msgs.msg
import tensorflow as tf
try:
    import tensorflow.contrib.tensorrt as trt
    tf.logging.set_verbosity(tf.logging.INFO)
except ImportError:
    trt = None
    rospy.loginfo('Could not import TensorRT, disabling it.')


def segmenter():
    rospy.init_node('semantic_segmenation')

    # Parse parameters.
    ns = rospy.get_name() + '/'
    image_topic = rospy.get_param(ns + 'image_topic', '/image')
    output_topic = rospy.get_param(ns + 'output_topic', '/output')
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
    rospy.loginfo('Publishing color images on topic:{}'.format(image_topic))
    rospy.loginfo('Restoring model from: {}'.format(frozen_model))

    # Load the complete network
    with tf.gfile.GFile(frozen_model, 'rb') as f:
        restored_graph_def = tf.GraphDef()
        restored_graph_def.ParseFromString(f.read())
    if trt_mode is not None and trt is not None:
        restored_graph_def = trt.create_inference_graph(
            input_graph_def=restored_graph_def,
            outputs=['class_colors'],
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
            return_elements=['class_colors'])[0]
        output_color = out.outputs[0]

        # The TensorRT docs suggest to set GPU options here specifying the GPU
        # memory fraction for tensorflow. In my experience nothing changed so
        # To keep the code portable we ignore this for now.
        #     config = tf.ConfigProto(gpu_options=tf.GPUOptions(
        #         per_process_gpu_memory_fraction=0.33))
        sess = tf.Session(graph=graph)

    # Setup a publisher for the result images.
    image_publisher = rospy.Publisher(
        output_topic, sensor_msgs.msg.Image, queue_size=1)

    # Configure the listener with a callback.
    bridge = cv_bridge.CvBridge()

    def callback(image):
        # Make sure somebody is subscribed otherwise we save power.
        if image_publisher.get_num_connections() > 0:
            # Slightly modify the shape and feed the image through the network.
            cv_image = bridge.imgmsg_to_cv2(image)
            cv_image = cv_image[None]
            result = sess.run(output_color, feed_dict={input_tensor: cv_image})

            # Publish the result
            output_message = bridge.cv2_to_imgmsg(result[0], encoding="bgr8")
            output_message.header = image.header

            image_publisher.publish(output_message)

    # Note that the buff_size is set very high here. Apparently there are more
    # than just the queue for the subscriber and if a single message doesn't fit
    # the buffer, images are take from that queue. Having a big enough buff_size
    # though will make sure that only a single image is stored in the queue and
    # thus the image is more "up-to-date".
    rospy.Subscriber(
        image_topic, sensor_msgs.msg.Image, callback, queue_size=1,
        buff_size=100000000)

    # Run indefinitely
    rospy.spin()

if __name__ == '__main__':
    segmenter()
