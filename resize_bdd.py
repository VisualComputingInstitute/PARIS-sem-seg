#!/usr/bin/env python3

from argparse import ArgumentParser
import glob
import os

import cv2
import numpy as np
try:
    from tqdm import tqdm
except ImportError:
    # In case tqdm isn't installed we don't use a progressbar.
    tqdm = lambda x, desc, smoothing, total: x
    print('Tqdm not found, not showing progress.')

import utils

parser = ArgumentParser(
    description='Downscale the Berkley Driving dataset to a provided width.')

parser.add_argument(
    '--target_width', required=True, type=utils.positive_int,
    help='Factor by which the images will be downscaled.')

parser.add_argument(
    '--bdd_root', required=True, type=utils.readable_directory,
    help='Path to the Berkley Driving dataset root.')

parser.add_argument(
    '--target_root', required=True, type=utils.writeable_directory,
    help='Location used to store the downscaled data.')

parser.add_argument(
    '--label_threshold', default=0.75, type=utils.zero_one_float,
    help='The threshold applied to the dominant label to decide which ambiguous'
         ' cases are mapped to the void label.')


def main():
    args = parser.parse_args()

    # Get filenames
    image_filenames = sorted(
        glob.glob(args.bdd_root + '/seg/images/*/*.jpg'))

    # We can miss-use the instances, since the dominant byte store the classes.
    label_filenames = sorted(
        glob.glob(args.bdd_root + '/seg/labels/*/*.png'))

    for image_filename in tqdm(image_filenames, desc='Resizing color',
            smoothing=0.01, total=len(image_filenames)):
        image = cv2.imread(image_filename)
        h, w, _ = image.shape
        h = h * args.target_width // w
        w = args.target_width
        image = cv2.resize(image, (w, h))
        target = image_filename.replace(args.bdd_root, args.target_root)
        target = target.replace('/seg/', '/')
        target_path = os.path.dirname(target)
        if not os.path.exists(target_path):
            os.makedirs(target_path)
        cv2.imwrite(target, image)

    for label_filename in tqdm(label_filenames, desc='Resizing labels',
            smoothing=0.01, total=len(image_filenames)):
        labels = cv2.imread(label_filename, cv2.IMREAD_GRAYSCALE)
        h, w = labels.shape
        h = h * args.target_width // w
        w = args.target_width
        labels = utils.soft_resize_labels(
            labels, (w, h), args.label_threshold, void_label=255)
        target = label_filename.replace(args.bdd_root, args.target_root)
        target = target.replace('/seg/', '/')
        target_path = os.path.dirname(target)
        if not os.path.exists(target_path):
            os.makedirs(target_path)
        cv2.imwrite(target, labels)


if __name__ == '__main__':
    main()