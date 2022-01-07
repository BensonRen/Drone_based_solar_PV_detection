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
DS_NAME = 'catalyst'

def get_images(data_dir, valid_percent=0.2, test_set_keyword=None):
    rgb_files = natsorted(glob(os.path.join(data_dir, 'img', '*.JPG')))
    if 'Exp1_4' in data_dir:            # The moving image cut from video are .png files
        rgb_files = natsorted(glob(os.path.join(data_dir, 'img', '*.png')))
    lbl_files = natsorted(glob(os.path.join(data_dir, 'labels', '*.png')))
    assert len(rgb_files) == len(lbl_files)
    train_files, valid_files = [], []
    # Ben added 2021.03.18: Get a random permutation so that the split is random
    rand_permutation = np.random.permutation(len(rgb_files))
    if 'test' in data_dir:
        rand_permutation = np.zeros_like(rand_permutation)
    # Ben added 2021.06.03: If the test_set_keyword is on, only take those with keyword to test set
    if test_set_keyword is not None:
        for i, pair in enumerate(zip(rgb_files, lbl_files)):
            if test_set_keyword in pair[0]:
                valid_files.append(pair)
            else:
                train_files.append(pair)
        return train_files, valid_files

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
    # ###############################
    # # Exp 1_1 Various resolution  #
    # ###############################
    for i in range(1, 5):
        ps = 512
        ol = 0
        pd = 0
        target_dir = r'../data_raw/Exp1_1_resolution_buckets/train_val/d{}'.format(i)
        create_dataset(data_dir=target_dir, save_dir=target_dir, patch_size=(ps, ps), 
                        pad=pd, overlap=ol, visualize=False)
        target_dir = r'../data_raw/Exp1_1_resolution_buckets/test/d{}'.format(i)
        create_dataset(data_dir=target_dir, save_dir=target_dir, patch_size=(ps, ps), 
                        pad=pd, overlap=ol, visualize=False)
    ###########################################
    # Exp 1_2 Satellite resolution simulation #
    ###########################################
    for res in [7.5, 15, 30, 60]:
        ps = 512
        ol = 0
        pd = 0
        target_dir = r'../data_raw/Exp1_2_sat_res/res_{}/train_val/'.format(res)
        create_dataset(data_dir=target_dir, save_dir=target_dir, patch_size=(ps, ps), 
                        pad=pd, overlap=ol, visualize=False)
                        
        target_dir = r'../data_raw/Exp1_2_sat_res/res_{}/test/'.format(res)
        create_dataset(data_dir=target_dir, save_dir=target_dir, patch_size=(ps, ps), 
                        pad=pd, overlap=ol, visualize=False)
    # ###########################################
    # # Exp 1_3 changed resolution to test set  #
    # ###########################################
    for i in range(1, 5):
        for j in range(1, 5):
            if i == j:      # skip when training res match test res
                continue
            ps = 512
            ol = 0
            pd = 0
            target_dir = r'../data_raw/Exp1_3_changed_testset/d{}_changed_to_d{}/test/'.format(i, j)
            create_dataset(data_dir=target_dir, save_dir=target_dir, patch_size=(ps, ps), 
                            pad=pd, overlap=ol, visualize=False)
    
    ###################################
    # Exp 1_4 moving images test set  #
    ###################################
    for i in range(1, 5):
        for mode in ['S','N']:
            ps = 512
            ol = 0
            pd = 0
            target_dir = r'../data_raw/Exp1_4_moving_imgs/d{}_{}_mode/test/'.format(i, mode)
            create_dataset(data_dir=target_dir, save_dir=target_dir, patch_size=(ps, ps), 
                            pad=pd, overlap=ol, visualize=False)
    
    ##########################################
    # Exp 1_5 adding aritificial motion blur #
    ##########################################
    for i in range(1, 5):
        for mb in [2, 4]:
            ps = 512
            ol = 0
            pd = 0
            target_dir = r'../data_raw/Exp1_5_artificial_motion_blur/d{}_mb_{}/test/'.format(i, mb)
            create_dataset(data_dir=target_dir, save_dir=target_dir, patch_size=(ps, ps), 
                            pad=pd, overlap=ol, visualize=False)
    