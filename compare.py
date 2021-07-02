import os
from glob import glob

from skimage import io
import matplotlib.pyplot as plt

#ct_dir = '/scratch/sr365/Catalyst_data/2021_03_21_15_C_90/save_root/H3_img_H3_model'
#ct_dir = '/scratch/sr365/RTI_data/positive_class/save_root/h3_pretrained'
#ct_dir = '/scratch/sr365/RTI_data/positive_class/save_root/ecresnet50_dcunet_dsSDhist_lre1e-02_lrd1e-02_ep180_bs5_ds30_100_150_dr0p1_crxent0p7_softiou0p3'
#ct_dir = '/scratch/sr365/Catalyst_data/2021_02_17_10_B_90/save_root/ecresnet50_dcunet_dsct_new_non_random_3_splits_lre1e-03_lrd1e-02_ep180_bs7_ds30_dr0p1_crxent7p0_softiou3p0'
ct_dir = '/scratch/sr365/Catalyst_data/moving_imgs/labelled/img/save_root/h2_model'
#sd_dir = '/scratch/sr365/RTI_data/positive_class/save_root/ecresnet50_dcunet_dsct_new_non_random_3_splits_lre1e-03_lrd1e-02_ep180_bs7_ds30_dr0p1_crxent7p0_softiou3p0'
#sd_dir = '/scratch/sr365/RTI_data/positive_class/save_root/RTI_h3_mixed_training_best_hyper'
sd_dir = '/scratch/sr365/Catalyst_data/moving_imgs/labelled/img/save_root/h3_model'
#data_dir = '/scratch/sr365/RTI_data/positive_class'
data_dir = '/scratch/sr365/Catalyst_data/moving_imgs/labelled/img'                             # The dir holding rgb images 
gt_dir =  '/scratch/sr365/Catalyst_data/moving_imgs/labelled/cvs'
# Usually gt_dir and data_dir are the same
#gt_dir = data_dir
compare_dir = os.path.join(data_dir, 'compare_moving')

image_names = [n.split('.')[0] for n in os.listdir(data_dir)]

# Make the compare directory
if not os.path.isdir(compare_dir):
    os.mkdir(compare_dir)

for file in os.listdir(ct_dir):
    if '.txt' in file or os.path.isdir(os.path.join(data_dir, file)):
        continue;
    name = file.split('.')[0].split('_conf')[0]
    print(name)
    fig, ax = plt.subplots(2, 2, figsize=(10, 10))
    ct = io.imread(glob(os.path.join(ct_dir, name+'*.png'))[0])
    sd = io.imread(glob(os.path.join(sd_dir, name+'*.png'))[0])
    # For normal case the rgb is .JPG
    #rgb = io.imread(glob(os.path.join(data_dir, name+'*.JPG'))[0])
    # For moving imgs the rgb is .png
    rgb = io.imread(glob(os.path.join(data_dir, name+'*.png'))[0])
    gt = io.imread(glob(os.path.join(gt_dir, name+'*.png'))[0])

    for i, (title, img) in enumerate(zip(['h2_model','h3_model', name, 'gt' ], [ct, sd, rgb, gt])):
        #ax_current = ax[i]
        ax_current = ax[i%2][int(i/2)]
        ax_current.imshow(img)
        ax_current.axis('off')
        ax_current.set_title(title)
    
    plt.tight_layout()
    plt.savefig(os.path.join(compare_dir, '{}.png'.format(name)), dpi=300)
    #plt.show()
