"""

"""


# Built-in
import os
import sys
import argparse

# Libs
import albumentations as A
from albumentations.pytorch import ToTensorV2
from natsort import natsorted

# Own modules
from mrs_utils import misc_utils, eval_utils
from network import network_io, network_utils


# Settings
GPU = 0
MODEL_DIR = r'/hdd/wh145/models/ct_finetune_final/ecresnet50_dcunet_dsimagenet_lre1e-03_lrd1e-02_ep30_bs7_ds5_dr0p1_crxent8p0_softiou2p0/'
DATA_DIR = r'/hdd/wh145/data/ct_examples/images'
SAVE_ROOT = r'/hdd/wh145/results/solarmapper/ct_infer'
DS_NAME = 'ct_infer'
LOAD_EPOCH = 30
PATCHS_SIZE = (512, 512)


def load_func_ct_tiles(data_dir, file_list=None, class_names=['panel',]):
    if file_list:
        with open(file_list, 'r') as f:
            rgb_files = f.read().splitlines()
    else:
        from glob import glob
        rgb_files = natsorted(glob(os.path.join(data_dir, '*.jpg')))
    lbl_files = [None] * len(rgb_files)
    assert len(rgb_files) == len(lbl_files)
    return rgb_files, lbl_files
    


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--gpu', type=int, default=GPU)
    parser.add_argument('--model_dir', type=str, default=MODEL_DIR)
    parser.add_argument('--data_dir', type=str, default=DATA_DIR)
    parser.add_argument('--save_root', type=str, default=SAVE_ROOT)
    parser.add_argument('--load_epoch', type=int, default=LOAD_EPOCH)
    parser.add_argument('--ds_name', type=str, default=DS_NAME)
    super_args = parser.parse_args()

    device, _ = misc_utils.set_gpu(super_args.gpu)

    # init model
    args = network_io.load_config(super_args.model_dir)
    model = network_io.create_model(args)
    if LOAD_EPOCH:
        args['trainer']['epochs'] = super_args.load_epoch
    ckpt_dir = os.path.join(
        super_args.model_dir, 'epoch-{}.pth.tar'.format(args['trainer']['epochs']))
    network_utils.load(model, ckpt_dir)
    print('Loaded from {}'.format(ckpt_dir))
    model.to(device)
    model.eval()

    # eval on dataset
    mean = (0.485, 0.456, 0.406)
    std = (0.229, 0.224, 0.225)
    tsfm_valid = A.Compose([
        A.Normalize(mean=mean, std=std),
        ToTensorV2(),
    ])
    save_dir = os.path.join(super_args.save_root, os.path.basename(
        network_utils.unique_model_name(args)))
    evaluator = eval_utils.Evaluator(super_args.ds_name, super_args.data_dir, tsfm_valid, device, load_func=load_func_ct_tiles)
    # evaluator.evaluate(model, PATCHS_SIZE, 2*model.lbl_margin,
    #                    pred_dir=save_dir, report_dir=save_dir)
    evaluator.infer(model=model, patch_size=PATCHS_SIZE, overlap=2*model.lbl_margin,
                       pred_dir=save_dir, save_conf=True)


if __name__ == '__main__':
    main()
