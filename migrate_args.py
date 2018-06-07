#!/usr/bin/env python3
from argparse import ArgumentParser
import json
import os
import shutil


parser = ArgumentParser(description='Migrate old argument json files.')

parser.add_argument(
    '--args_file', required=True, type=str,
    help='Location of the old arguments json file which will be tested and '
         'migrated to the current format.')


def main():
    args = parser.parse_args()

    # Load the file and check if deprecated arguments are present.
    with open(args.args_file, 'r') as f:
        arguments = json.load(f)
    changed = False

    # A single dataset is specified instead of a list of datasets.
    if type(arguments['dataset_config']) == str:
        for k in ['dataset_config', 'dataset_root', 'train_set']:
            arguments[k] = [arguments[k]]
        changed = True

    # The old crop augmentation is used. Since this was only ever really used
    # with quarter resolution CityScapes images, I will not make this super
    # general and simply assume this is the case here.
    if 'crop_augment' in arguments and arguments['crop_augment'] > 0:
        arguments['fixed_crop_augment_height'] = 256 - arguments['crop_augment']
        arguments['fixed_crop_augment_width'] = 512 - arguments['crop_augment']
        arguments.pop('crop_augment')
        changed = True

    if changed:
        # Make a backup copy of the old file
        backup_saved = False
        for i in range(100):
            backup_name = args.args_file.replace(
                '.json', '_bu{}.json'.format(i))
            if not os.path.isfile(backup_name):
                shutil.copy(args.args_file, backup_name)
                print('Backup written to: {}'.format(backup_name))
                backup_saved = True
                break
        if not backup_saved:
            print('More than a 100 backups seem to exist, quiting, fix this '
                  'first.')
            exit(1)

        # Write the new file.
        with open(args.args_file, 'w') as f:
            json.dump(
                arguments, f, ensure_ascii=False, indent=2, sort_keys=True)
    else:
        print('No changes needed to be made.')


if __name__ == '__main__':
    main()
