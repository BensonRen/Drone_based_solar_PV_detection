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
from mrs_utils import misc_utils, process_block

# Settings
DS_NAME = 'catalyst_sat_h3'


def get_images(data_dir, valid_percent=0.1, split=False):
    rgb_files = natsorted(glob(os.path.join(data_dir, 'images', '*.JPG')))
    lbl_files = natsorted(glob(os.path.join(data_dir, 'annotations', '*.png')))
    assert len(rgb_files) == len(lbl_files)
    train_files, valid_files = [], []
    # Ben added 2021.03.18: Get a random permutation so that the split is random
    rand_permutation = np.random.permutation(len(rgb_files))
    for i, pair in enumerate(zip(rgb_files, lbl_files)):
        if rand_permutation[i] <= int(valid_percent * len(rgb_files)):
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


def get_stats(img_dir):
    from data import data_utils
    from glob import glob
    rgb_imgs = glob(os.path.join(img_dir, '*.jpg'))
    ds_mean, ds_std = data_utils.get_ds_stats(rgb_imgs)
    return np.stack([ds_mean, ds_std], axis=0)


def get_stats_pb(img_dir):
    val = process_block.ValueComputeProcess(DS_NAME, os.path.join(os.path.dirname(__file__), '../stats/builtin'),
                                            os.path.join(os.path.dirname(__file__), '../stats/builtin/{}.npy'.format(DS_NAME)), func=get_stats).\
        run(img_dir=img_dir).val
    val_test = val
    return val, val_test


if __name__ == '__main__':
    ps = 512
    ol = 0
    pd = 0
    create_dataset(data_dir=r'/scratch/sr365/Catalyst_data/h3_satellite/',
                   save_dir=r'/scratch/sr365/Catalyst_data/h3_satellite/', patch_size=(ps, ps), pad=pd, overlap=ol, visualize=False)
