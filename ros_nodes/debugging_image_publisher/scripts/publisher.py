#!/usr/bin/env python
import glob

import cv2
import cv_bridge
import rospy
import sensor_msgs.msg


def publisher():
    rospy.init_node('debugging_publisher')

    # Parse parameters.
    ns = rospy.get_name() + '/'
    image_topic = rospy.get_param(ns + 'image_topic', '/image')
    image_folder = rospy.get_param(ns + 'image_folder')
    fps = float(rospy.get_param(ns + 'fps', 5.0))
    fix_width_to = rospy.get_param(ns + 'fix_width_to', 'none')
    file_extension = rospy.get_param(ns + 'file_extension', 'jpg')
    if fix_width_to == 'none':
        fix_width_to = None
    else:
        fix_width_to = int(fix_width_to)

    # Load the file list for publishing.
    file_list = sorted(glob.glob(image_folder + '/*' + file_extension))
    if len(file_list) == 0:
        print('No matching {} files fone in {}'.format(
            file_extension, image_folder))
        exit(1)

    # Init some variables needed for publishing
    image_publisher = rospy.Publisher(
        image_topic, sensor_msgs.msg.Image, queue_size=1)
    bridge = cv_bridge.CvBridge()

    # Forever keep looping over the files
    seq = 0
    while not rospy.is_shutdown():
        # Load the iamge
        image_file = file_list[seq % len(file_list)]
        image = cv2.imread(image_file)

        # Possible resize the image.
        if fix_width_to:
            h, w, _ = image.shape
            h = h * fix_width_to // w
            image = cv2.resize(image, (fix_width_to, h))

        # Convert it to the proper image format for publishing.
        image_msg = bridge.cv2_to_imgmsg(image, "bgr8")
        image_msg.header.seq = seq
        image_msg.header.stamp = rospy.Time.now()

        # Publish the image
        image_publisher.publish(image_msg)

        # Wait according to the fps. We'll just assume that loading and resizing
        # takes up to no actual time.
        rospy.sleep(1.0 / fps)
        seq += 1


if __name__ == '__main__':
    publisher()
