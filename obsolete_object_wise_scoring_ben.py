import sys
from mrs_utils import misc_utils, vis_utils
from mrs_utils import eval_utils
import os
from skimage import io
import numpy as np
import matplotlib.pyplot as plt


# Creat object scorer class
osc = eval_utils.ObjectScorer(min_th=0.5, link_r=20, eps=2)

# Define the source
data_dir = '/scratch/sr365/Catalyst_data/2021_03_21_15_C_90/H3_raw'
conf_dir = '/scratch/sr365/Catalyst_data/2021_03_21_15_C_90/save_root/H3_img_H2_model' 

save_name = 'H3_img_H2_model'

def get_conf_true_from_img(lbl_file, conf_file):
    """
    The function to get the p r curve (object-wise) from a labelled photo and the 
    """
    lbl_img, conf_img = misc_utils.load_file(lbl_file)[:,:,0]/255, misc_utils.load_file(conf_file)

    # Group objects
    lbl_groups = osc.get_object_groups(lbl_img)
    conf_groups = osc.get_object_groups(conf_img)
    lbl_group_img = eval_utils.display_group(lbl_groups, lbl_img.shape[:2], need_return=True)
    conf_group_img = eval_utils.display_group(conf_groups, conf_img.shape[:2], need_return=True)


    # Score the conf map
    conf_list, true_list = eval_utils.score(conf_img, lbl_img, min_th=0.5, link_r=10, iou_th=0.5)
    return conf_list, true_list

def plot_PR_curve(conf_list, true_list, save_name='PR_curve'):       
    """
    The function to plot PR curve from a list of confidence and true list
    """
    ap, p, r, _ = eval_utils.get_precision_recall(conf_list, true_list)
    plt.plot(r[1:], p[1:])
    plt.xlim([0, 1])
    plt.ylim([0, 1])
    plt.xlabel('recall')
    plt.ylabel('precision')
    plt.title('AP={:.2f}'.format(ap))
    plt.tight_layout()
    plt.savefig('../PR_curves/' + save_name + '.png')


if __name__ == '__main__':
    large_conf_list, large_true_list = [], []
    for file in os.listdir(conf_dir):
        print("processing file: ", file)
        if not file.endswith('_conf.png'):
            continue
        # get the file names
        conf_file = os.path.join(conf_dir, file)
        lbl_file = os.path.join(data_dir, file.replace('_conf',''))
        # get the conf_list and true list
        conf_list, true_list = get_conf_true_from_img(lbl_file, conf_file)

        if len(conf_list) == 0  or len(true_list) == 0:
            print("Either you don't have a true file or a ground truth", file)
            continue

        print("conf_list shape:", np.shape(conf_list))
        print("true_list shape:", np.shape(true_list))
        print("large conf list shape:", np.shape(large_conf_list))
        print("large true list shape:", np.shape(large_true_list))
        if len(large_conf_list) == 0:
            large_conf_list = conf_list
            large_true_list = true_list
        else:
            large_conf_list = np.concatenate((large_conf_list, conf_list), axis=0)
            large_true_list = np.concatenate((large_true_list, true_list), axis=0)
    
    np.save('../PR_curves/conf_list.npy', large_conf_list)
    np.save('../PR_curves/true_list.npy', large_true_list)
    plot_PR_curve(np.reshape(large_conf_list, [-1,]), np.reshape(large_true_list, [-1,]), save_name = save_name)
