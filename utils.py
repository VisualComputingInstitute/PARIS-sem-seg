from argparse import ArgumentTypeError
import os
import re
import signal

import cv2
import numpy as np


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
                                    '{} was required'.format(arg,
                                                             type_.__name__,
                                                             low, high))
    elif low is not None and high is None:
        if l_(value, low):
            return value
        else:
            raise ArgumentTypeError('Found {} where an {} with value >= {}'
                                    '{} was required'.format(arg,
                                                             type_.__name__,
                                                             low))
    elif low is None and high is not None:
        if g_(value, high):
            return value
        else:
            raise ArgumentTypeError('Found {} where an {} with value <= '
                                    '{} was required'.format(arg,
                                                             type_.__name__,
                                                             high))
    else:
        return value


def positive_int(arg):
    return number_between_x(arg, int, 0, None)


def positive_float(arg):
    return number_between_x(arg, float, 0, None)


def nonnegative_int(arg):
    return number_between_x(arg, int, -1, None)


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
        new_size: The target size, specified as `(Width, Height)`
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

    if len(possible_labels) == 0:
        # This image is empty. We can simply return an image full of void labels.
        return np.full(new_size[::-1], void_label, dtype=labels.dtype)

    label_vol = np.zeros(
        (labels.shape[0], labels.shape[1], len(possible_labels)))
    for i, l in enumerate(possible_labels):
        label_vol[:, :, i] = (labels == l)

    label_vol = cv2.resize(label_vol, new_size, interpolation=cv2.INTER_LINEAR)

    # If there is only a single label, then the resize function returns a 2D
    # tensor.
    if len(label_vol.shape) == 2:
        label_vol = np.reshape(label_vol, (*label_vol.shape, 1))

    # Find the max label using this mapping and the actual label value
    max_idx = np.argmax(label_vol, 2)
    max_val = np.max(label_vol, 2)

    # Remap to original values
    max_idx = possible_labels[max_idx]
    # Set the void label according to threshold.
    max_idx[max_val < valid_threshold] = void_label

    return max_idx.astype(labels.dtype)


def select_existing_root(path, check_only_basedir=False):
    """Select a path based on multiple options included in square brackets.

    Given a path that needs to be dynamic based on the system, e.g. running on a
    machine or on a server, the different options can be included in square
    brackets, which will be expanded into several full paths which are then
    tested for existence. For example:
    /root/[location1,location2]/rest_of_path
    will result in either
    /root/location1/rest_of_path or /root/location2/rest_of_path
    depending which is available on the system.
    If none exists an error is raised, otherwise the first
    valid path will be returned. If no square brackets are found this is a noop.

    Args:
        path: A path where multiple options are included in square brackets.
        check_only_basedir: If this is set to True, only the basedir and the
            option is checked, but not the parts after the option.
            This is useful if parts of the path will be created later on.

    Raises:
        IOError if none of the paths exist.
        ValueError if more than 1 opening and closing bracket is found or no
            valid pair of brackets is available.

    Returns:
        The first valid path in given the options in the path
    """

    if '[' in path or ']' in path:
        if path.count('[') == 1 and path.count(']') == 1:
            matches = re.search('(.*)\[(.*)\](.*)', path)
            if matches is None:
                ValueError('No valid opening and closing square bracket '
                           'configuration could be found in: {}'.format(path))
            else:
                opening = matches.group(1)
                options = matches.group(2).split(',')
                closing = matches.group(3)
                for opt in options:
                    p = opening + opt
                    if not check_only_basedir:
                        p += closing
                    if os.path.exists(p):
                        if check_only_basedir:
                            p += closing
                        return p

                # No valid path was found
                raise IOError('No valid path was found given the options in '
                              '{}.'.format(path))

        else:
            raise ValueError('The path should contain exactly 1 opening and '
                             'closing square bracket or none at all. This is not'
                             ' the case for: {}'.format(path))
    else:
        return path


def load_dataset(csv_file, dataset_root, label_root=None):
    """Loads a dataset by reading image, label filenames tuples from a file.

    Args:
        csv_file: A string containing the path to the dataset csv file. Each
            line should contain an image and a label filename tuple.
        dataset_root: A string with the dataset root directory. This is
            perpended to each filename and it is verified all files exist.
            If the dataset can be located on multiple locations, a part of the
            path can be encapsulated with square brackets and all options within
            the brackets are tried, e.g. /[location1,location2]/data/images.
        label_root: An optional string with the dataset root directory in case
            the labels are stored in a different location. This is used when
            predictions are made at a lower resolution, but the evaluation
            should be performed with the original resolution. The same square
            bracket syntax as for the dataset_root applies.

    Returns:
        A list of string tuples containing image filenames and label filenames.

    Raises:
        IOError if any one file is missing.
    """
    dataset = np.genfromtxt(csv_file, delimiter=',', dtype='|U')

    file_tuples = []
    missing = []

    dataset_root = select_existing_root(dataset_root)

    if label_root is None:
        label_root = dataset_root
    else:
        label_root = select_existing_root(label_root)

    for image, labels in dataset:
        image_file = os.path.join(dataset_root, image.strip())
        if not os.path.isfile(image_file):
            missing.append(image_file)

        label_file = os.path.join(label_root, labels.strip())
        if not os.path.isfile(label_file):
            missing.append(label_file)

        file_tuples.append((image_file, label_file))

    if len(missing) > 1:
        raise IOError('Using the `{}` file and `{}` as a dataset root '
                      '{} files are missing:\n{}'.format(
            csv_file, dataset_root, len(missing),
            '\n'.join(missing)))

    return file_tuples


def mixed_dataset_generator(datasets, weights):
    """ Given a list of datasets, this function yields weighted samples forever.

    Args:
        datasets: A list of datasets, these can be single objects or tuples.
        weights: The weights for the datasets which will be used for sampling
            data points from each dataset.

    Yields:
        A tuple containing dataset samples and dataset ids representing the
        index of the dataset in the datasets list.

    Raises:
        ValueError if the number of datasets and weights do not match.
    """

    if len(datasets) != len(weights):
        raise ValueError(
            'The length of the datasets needs to correspond to the length of '
            'the weights, got {} and {}'.format(len(datasets), len(weights)))
    datasets = [np.array(d) for d in datasets]
    weights = np.array(weights) / np.sum(weights)
    dataset_ids = np.arange(len(datasets))
    index = np.zeros(len(datasets), dtype=np.uint32)

    def _shuffle_datasets(dataset_index):
        p = np.random.permutation(len(datasets[dataset_index]))
        datasets[dataset_index] = datasets[dataset_index][p]

    for i in range(len(datasets)):
        _shuffle_datasets(i)

    while True:
        # Sample which dataset to yield from.
        dataset_id = np.random.choice(size=1, a=dataset_ids, p=weights)[0]

        # Get the data.
        data = datasets[dataset_id][index[dataset_id]]

        # Move the index
        index[dataset_id] += 1

        # Possibly reshuffle
        if index[dataset_id] == len(datasets[dataset_id]):
            index[dataset_id] = 0
            _shuffle_datasets(dataset_id)

        # Add an index to the data and yield
        yield (tuple(list(data) + [dataset_id]))


def get_logging_dict(name):
    return {
        'version': 1,
        'disable_existing_loggers': False,
        'formatters': {
            'standard': {
                'format': '%(asctime)s [%(levelname)s] %(name)s: %(message)s'
            },
        },
        'handlers': {
            'stderr': {
                'class': 'logging.StreamHandler',
                'level': 'INFO',
                'formatter': 'standard',
                'stream': 'ext://sys.stderr',
            },
            'logfile': {
                'level': 'DEBUG',
                'formatter': 'standard',
                'class': 'logging.FileHandler',
                'filename': name + '.log',
                'mode': 'a',
            }
        },
        'loggers': {
            '': {
                'handlers': ['stderr', 'logfile'],
                'level': 'DEBUG',
                'propagate': True
            },

            # extra ones to shut up.
            'tensorflow': {
                'handlers': ['stderr', 'logfile'],
                'level': 'INFO',
            },
        }
    }


# This class comes from Lucas Beyer's toolbox which can be found at https://github.com/lucasb-eyer/lbtoolbox.
# It is based on an original idea from https://gist.github.com/nonZero/2907502 and heavily modified.
class Uninterrupt(object):
    """
    Use as:
    with Uninterrupt() as u:
        while not u.interrupted:
            # train
    """

    def __init__(self, sigs=(signal.SIGINT,), verbose=False):
        self.sigs = sigs
        self.verbose = verbose
        self.interrupted = False
        self.orig_handlers = None

    def __enter__(self):
        if self.orig_handlers is not None:
            raise ValueError("Can only enter `Uninterrupt` once!")

        self.interrupted = False
        self.orig_handlers = [signal.getsignal(sig) for sig in self.sigs]

        def handler(signum, frame):
            self.release()
            self.interrupted = True
            if self.verbose:
                print("Interruption scheduled...", flush=True)

        for sig in self.sigs:
            signal.signal(sig, handler)

        return self

    def __exit__(self, type_, value, tb):
        self.release()

    def release(self):
        if self.orig_handlers is not None:
            for sig, orig in zip(self.sigs, self.orig_handlers):
                signal.signal(sig, orig)
        self.orig_handlers = None
