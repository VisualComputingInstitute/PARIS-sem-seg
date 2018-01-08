#!/usr/bin/env python3
import cv2
import numpy as np
import os


def check_directory(arg, access=os.W_OK, access_str="writeable"):
    """ Check for directory-type argument validity.

    Checks whether the given `arg` commandline argument is either a readable
    existing directory, or a createable/writeable directory.

    Args:
        arg (string): The commandline argument to check.
        access (constant): What access rights to the directory are requested.
        access_str (string): Used for the error message.

    Returns:
        The string passed din `arg` if the checks succeed.

    Raises:
        ArgumentTypeError if the checks fail.
    """
    path_head = arg
    while path_head:
        if os.path.exists(path_head):
            if os.access(path_head, access):
                # Seems legit, but it still doesn't guarantee a valid path.
                # We'll just go with it for now though.
                return arg
            else:
                raise ArgumentTypeError(
                    'The provided string `{0}` is not a valid {1} path '
                    'since {2} is an existing folder without {1} access.'
                    ''.format(arg, access_str, path_head))
        path_head, _ = os.path.split(path_head)

    # No part of the provided string exists and can be written on.
    raise ArgumentTypeError('The provided string `{}` is not a valid {}'
                            ' path.'.format(arg, access_str))


def writeable_directory(arg):
    """ To be used as a type for `ArgumentParser.add_argument`. """
    return check_directory(arg, os.W_OK, "writeable")


def readable_directory(arg):
    """ To be used as a type for `ArgumentParser.add_argument`. """
    return check_directory(arg, os.R_OK, "readable")


def number_between_x(arg, type_, low=None, high=None, inclusive_low=False,
                     inclusive_heigh=False):
    if inclusive_low:
        l_ = lambda v, t: v >= t
    else:
        l_ = lambda v, t: v > t

    if inclusive_heigh:
        g_ = lambda v, t: v <= t
    else:
        g_ = lambda v, t: v < t

    try:
        value = type_(arg)
    except ValueError:
        raise ArgumentTypeError('The argument "{}" is not an {}.'.format(
            arg, type_.__name__))

    if low is not None and high is not None:
        if l_(value, low) and g_(value, high):
            return value
        else:
            raise ArgumentTypeError('Found {} where an {} with {} <= value <= '
                '{} was required'.format(arg, type_.__name__, low, high))
    elif low is not None and high is None:
        if l_(value, low):
            return value
        else:
            raise ArgumentTypeError('Found {} where an {} with value >= {}'
                '{} was required'.format(arg, type_.__name__, low))
    elif low is None and high is not None:
        if g_(value, high):
            return value
        else:
            raise ArgumentTypeError('Found {} where an {} with value <= '
                '{} was required'.format(arg, type_.__name__, high))
    else:
        return value


def positive_int(arg):
    return number_between_x(arg, int, 0, None)


def zero_one_float(arg):
    return number_between_x(arg, float, 0, 1, True, True)


def soft_resize_labels(labels, new_size, valid_threshold, void_label=-1):
    """Perform a soft resizing of a label image.

    This is achieved by first creating a 3D array of size `(height, width,
    label_count)` which is then resized using OpenCV. Since all label channels
    are resized separately, no mixing of labels is performed and discrete label
    values can be retrieved afterwards.

    Args:
        label: A 2D Numpy array containing labels.
        new_size: The target size, specified as `(Width, Height)` TODO: check!
        valid_threshold: The fraction of the dominant label within a group,
            needed to be set. If it falls below this fraction, the `void_label`
            is set instead.
        void_label: The actual label value of the void label. Defaults to -1.

    Returns:
        A resized version of the label image. Using interpolation, but returning
        only valid labels.

    """
    possible_labels = set(np.unique(labels))
    if void_label in possible_labels:
        possible_labels.remove(void_label)
    possible_labels = np.asarray(list(possible_labels))

    label_vol = np.zeros(
        (labels.shape[0], labels.shape[1], len(possible_labels)))
    for i, l in enumerate(possible_labels):
        label_vol[:,:, i] = (labels == l)

    label_vol = cv2.resize(label_vol, new_size)

    # If there is only a single label, then the resize function returns a 2D
    # tensor.
    if len(label_vol.shape) == 2:
        label_vol = np.reshape(label_vol, (*label_vol.shape, 1))

    # Fin the max label using this mapping and the actual label value
    max_idx = np.argmax(label_vol, 2)
    max_val = np.max(label_vol,2)

    # Remap to original values
    max_idx = possible_labels[max_idx]
    # Set the void label according to threshold.
    max_idx[max_val < valid_threshold] = void_label

    return max_idx.astype(labels.dtype)