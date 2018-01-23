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
    tqdm = lambda x: x
    print('Tqdm not found, not showing progress.')

import utils

parser = ArgumentParser(
    description='Downscale the cityscapes dataset by a provided factor.')

parser.add_argument(
    '--downscale_factor', required=True, type=utils.positive_int,
    help='Factor by which the images will be downscaled.')

parser.add_argument(
    '--cityscapes_root', required=True, type=utils.readable_directory,
    help='Path to the cityscapes dataset root.')

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
    image_filenames = glob.glob(
        os.path.join(args.cityscapes_root, 'leftImg8bit') + '/*/*/*.png')

    label_filenames = glob.glob(
        os.path.join(args.cityscapes_root, 'gt') + '*/*/*/*labelIds.png')

    for image_filename, label_filename in tqdm(
            zip(image_filenames, label_filenames),
            desc='Resizing', smoothing=0.01, total=len(image_filenames)):
        # Resize the color image.
        image = cv2.imread(image_filename)
        h, w, _ = image.shape
        h = h // args.downscale_factor
        w = w // args.downscale_factor
        image = cv2.resize(image, (w, h))
        target = image_filename.replace(args.cityscapes_root, args.target_root)
        target_path = os.path.dirname(target)
        if not os.path.exists(target_path):
            os.makedirs(target_path)
        cv2.imwrite(target, image)

        # Resize the label image.
        labels = cv2.imread(label_filename, cv2.IMREAD_GRAYSCALE)
        h, w = labels.shape
        h = h // args.downscale_factor
        w = w // args.downscale_factor
        labels = utils.soft_resize_labels(
            labels, (w, h), args.label_threshold, void_label=0)
        target = label_filename.replace(args.cityscapes_root, args.target_root)
        target_path = os.path.dirname(target)
        if not os.path.exists(target_path):
            os.makedirs(target_path)
        cv2.imwrite(target, labels)


if __name__ == '__main__':
    main()