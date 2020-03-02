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


def get_images(data_dir):
    rgb_files = natsorted(glob(os.path.join(data_dir, 'images', '*.png')))
    lbl_files = natsorted(glob(os.path.join(data_dir, 'annotations', '*.png')))
    assert len(rgb_files) == len(lbl_files)
    while True:
        valid_idx = np.random.randint(0, len(lbl_files))
        if not '015840_se.png' in rgb_files[valid_idx]: # the urban tile must be used for training
            break
    train_files, valid_files = [], []
    for i, pair in enumerate(zip(rgb_files, lbl_files)):
        if i == valid_idx:
            valid_files.append(pair)
        else:
            train_files.append(pair)
    return train_files, valid_files


def create_dataset(data_dir, save_dir, patch_size, pad, overlap, visualize=False):
    # create folders and files
    patch_dir = os.path.join(save_dir, 'patches')
    misc_utils.make_dir_if_not_exist(patch_dir)
    record_file_train = open(os.path.join(
        save_dir, 'file_list_train.txt'), 'w+')
    record_file_valid = open(os.path.join(
        save_dir, 'file_list_valid.txt'), 'w+')
    train_files, valid_files = get_images(data_dir)

    for img_file, lbl_file in tqdm(train_files):
        prefix = os.path.splitext((os.path.basename(img_file)))[0]
        for rgb_patch, gt_patch, y, x in data_utils.patch_tile(img_file, lbl_file, patch_size, pad, overlap):
            if visualize:
                from mrs_utils import vis_utils
                vis_utils.compare_figures(
                    [rgb_patch, gt_patch], (1, 2), fig_size=(12, 5))
            img_patchname = '{}_y{}x{}.jpg'.format(prefix, int(y), int(x))
            lbl_patchname = '{}_y{}x{}.png'.format(prefix, int(y), int(x))
            misc_utils.save_file(os.path.join(
                patch_dir, img_patchname), rgb_patch.astype(np.uint8))
            misc_utils.save_file(os.path.join(
                patch_dir, lbl_patchname), gt_patch.astype(np.uint8))
            record_file_train.write(
                '{} {}\n'.format(img_patchname, lbl_patchname))

    for img_file, lbl_file in tqdm(valid_files):
        prefix = os.path.splitext((os.path.basename(img_file)))[0]
        for rgb_patch, gt_patch, y, x in data_utils.patch_tile(img_file, lbl_file, patch_size, pad, overlap):
            if visualize:
                from mrs_utils import vis_utils
                vis_utils.compare_figures(
                    [rgb_patch, gt_patch], (1, 2), fig_size=(12, 5))
            img_patchname = '{}_y{}x{}.jpg'.format(prefix, int(y), int(x))
            lbl_patchname = '{}_y{}x{}.png'.format(prefix, int(y), int(x))
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
    create_dataset(data_dir=r'/home/wh145/data/CT_downsampled',
                   save_dir=r'/home/wh145/data/CT_downsampled', patch_size=(ps, ps), pad=pd, overlap=ol, visualize=False)
