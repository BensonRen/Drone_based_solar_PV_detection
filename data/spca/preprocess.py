"""

"""


# Built-in
import os
from glob import glob

# Libs
import numpy as np
from tqdm import tqdm
from natsort import natsorted

# Own modules
from data import data_utils
from mrs_utils import misc_utils


def get_images(data_dir, valid_percent=0.2):
    rgb_files = natsorted(glob(os.path.join(data_dir, '*RGB.jpg')))
    lbl_files = natsorted(glob(os.path.join(data_dir, '*GT.png')))
    assert len(rgb_files) == len(lbl_files)
    city_names = ['Fresno', 'Modesto', 'Stockton']
    city_files = {city_name: [(rgb_file, lbl_file) for (rgb_file, lbl_file) in zip(rgb_files, lbl_files)
                              if city_name in rgb_file] for city_name in city_names}
    train_files, valid_files = [], []
    for city_name, file_pairs in city_files.items():
        valid_size = int(valid_percent * len(file_pairs))
        train_files.extend(file_pairs[valid_size:])
        valid_files.extend(file_pairs[:valid_size])
    return train_files, valid_files


def create_dataset(data_dir, save_dir, patch_size, pad, overlap, valid_percent=0.2, visualize=False):
    # create folders and files
    patch_dir = os.path.join(save_dir, 'patches')
    misc_utils.make_dir_if_not_exist(patch_dir)
    record_file_train = open(os.path.join(
        save_dir, 'file_list_train.txt'), 'w+')
    record_file_valid = open(os.path.join(
        save_dir, 'file_list_valid.txt'), 'w+')
    train_files, valid_files = get_images(data_dir, valid_percent)

    for img_file, lbl_file in tqdm(train_files):
        city_name = os.path.splitext(os.path.basename(img_file))[
            0].split('_')[0]
        for rgb_patch, gt_patch, y, x in data_utils.patch_tile(img_file, lbl_file, patch_size, pad, overlap):
            if visualize:
                from mrs_utils import vis_utils
                vis_utils.compare_figures(
                    [rgb_patch, gt_patch], (1, 2), fig_size=(12, 5))
            img_patchname = '{}_y{}x{}.jpg'.format(city_name, int(y), int(x))
            lbl_patchname = '{}_y{}x{}.png'.format(city_name, int(y), int(x))
            misc_utils.save_file(os.path.join(
                patch_dir, img_patchname), rgb_patch.astype(np.uint8))
            misc_utils.save_file(os.path.join(
                patch_dir, lbl_patchname), gt_patch.astype(np.uint8))
            record_file_train.write(
                '{} {}\n'.format(img_patchname, lbl_patchname))

    for img_file, lbl_file in tqdm(valid_files):
        city_name = os.path.splitext(os.path.basename(img_file))[
            0].split('_')[0]
        for rgb_patch, gt_patch, y, x in data_utils.patch_tile(img_file, lbl_file, patch_size, pad, overlap):
            if visualize:
                from mrs_utils import vis_utils
                vis_utils.compare_figures(
                    [rgb_patch, gt_patch], (1, 2), fig_size=(12, 5))
            img_patchname = '{}_y{}x{}.jpg'.format(city_name, int(y), int(x))
            lbl_patchname = '{}_y{}x{}.png'.format(city_name, int(y), int(x))
            misc_utils.save_file(os.path.join(
                patch_dir, img_patchname), rgb_patch.astype(np.uint8))
            misc_utils.save_file(os.path.join(
                patch_dir, lbl_patchname), gt_patch.astype(np.uint8))
            record_file_valid.write(
                '{} {}\n'.format(img_patchname, lbl_patchname))


if __name__ == '__main__':
    ps = 512
    ol = 0
    pd = 0
    create_dataset(data_dir=r'/home/wh145/data/spca/Original_Tiles',
                   save_dir=r'/home/wh145/data/spca', patch_size=(ps, ps), pad=pd, overlap=ol, visualize=False)
