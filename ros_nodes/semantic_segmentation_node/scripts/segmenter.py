#!/usr/bin/env python
import cv_bridge
import rospy
import sensor_msgs.msg
import tensorflow as tf


def segmenter():
    rospy.init_node('semantic_segmenation')

    # Parse parameters.
    ns = rospy.get_name() + '/'
    image_topic = rospy.get_param(ns + 'image_topic', '/image')
    output_topic = rospy.get_param(ns + 'output_topic', '/output')
    frozen_model = rospy.get_param(ns + 'frozen_model')

    # Load the complete network
    with tf.gfile.GFile(frozen_model, 'rb') as f:
        restored_graph_def = tf.GraphDef()
        restored_graph_def.ParseFromString(f.read())

    with tf.Graph().as_default() as graph:
        tf.import_graph_def(restored_graph_def)

        # Fetch the input and output.
        input_tensor = graph.get_tensor_by_name('import/input:0')
        output_color = graph.get_tensor_by_name('import/class_colors:0')

        # Just to document that there is a tensor for probabilities too.
        # output_probabilities = graph.get_tensor_by_name(
        #     'import/class_probabilities:0')

        sess = tf.Session(graph=graph)

    # Setup a publisher for the result images.
    image_publisher = rospy.Publisher(
        output_topic, sensor_msgs.msg.Image, queue_size=1)

    # Configure the listener with a callback.
    bridge = cv_bridge.CvBridge()

    def callback(image):
        print(image.header.seq)
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
