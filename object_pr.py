import os
from glob import glob
from tqdm import tqdm
import matplotlib.pyplot as plt
import numpy as np
from skimage import io, measure
from mrs_utils import eval_utils
from sklearn import metrics
from multiprocessing import Pool

def plain_post_proc(conf, min_conf, min_area):
    tmp = conf > min_conf
    label = measure.label(tmp)
    props = measure.regionprops(label, conf)

    dummy = np.zeros(conf.shape)
    for p in props:
        if p.area > min_area:
            for x, y in p.coords:
                dummy[x, y] = 1
    return dummy

################################################################################################################################################
def get_area_covered(tile_name, tile_size):
    """
    The function that calculates the area covered in a image
    param: tile_name: The name of the tile, from which we get the height
    param: tile_size: The size of the tile, from which we calculate the total area
    """
    def get_current_res(height):
        return height * 6.17 / 4000 / 4.3
    def get_height(img_name):
        #print('current image name is: ', img_name)
        return int(img_name.split('height_')[-1].split('m')[0])
    height = get_height(tile_name)
    res = get_current_res(height)
    number_pixel = tile_size[0] * tile_size[1]
    area = number_pixel * res * res
    return area
################################################################################################################################################


def plot_PR_curve(min_region, dilation_size, link_r, min_th, iou_th, conf_dir_list, tile_name_list, gt_dict, save_title, output_dir, calculate_area=True):
    """
    The funciton to plot the PR curve
    :param min_region: The minimal number of pixels to count
    :param dilation_size: The number of pixels to dialate in image processing
    :param link_r: 
    :param min_th: The minimal threshold to consider as positive prediction
    """
    # Loop through a list of confidence maps
    for conf_dir in tqdm(conf_dir_list):
        plt.figure(figsize=(8, 8))

        # Get the confidence map dictionary where the key is the tile_name and the value is the actual images read from skimage.io
        conf_dict = dict(
            zip(
                tile_name_list,
                [io.imread(os.path.join(conf_dir, f+'_conf.png'))
                for f in tile_name_list]
            )
        )
        # Place holder for confidence list and gt list
        conf_list, true_list = [], []
        area_list = []                                  # Getting the area for the normalized ROC curve
        # Loop over each tiles
        for tile in tqdm(tile_name_list, desc='Tiles'):
            if len(np.shape(gt_dict[tile])) == 3:
                conf_img, lbl_img = conf_dict[tile]/255, gt_dict[tile][:, :, 0]                                 # Get the confidence image and the label image
            else:
                conf_img, lbl_img = conf_dict[tile]/255, gt_dict[tile]  
            # save_confusion_map = conf_dir.split('Catalyst_data')[-1].split('image')[0].replace('/','_') + tile    # THis is for 
            save_confusion_map = conf_dir.split('images/')[-1].replace('/','_') + tile
            # print('the save confusion plot name is :', save_confusion_map)
            conf_tile, true_tile = eval_utils.score(                                                        # Call a function in utils.score to score this
                conf_img, lbl_img, min_region=min_region, min_th=min_th/255, 
                dilation_size=dilation_size, link_r=link_r, iou_th=iou_th, 
                tile_name=os.path.join('post_process_understanding', tile))#, save_confusion_map=save_confusion_map)    
            conf_list.extend(conf_tile)
            true_list.extend(true_tile)
            if calculate_area:              # For RTI data this  is off
                area = get_area_covered(tile, np.shape(gt_dict[tile])[:2])
                area_list.append(area)                      # Getting the area for the normalized ROC curve
        #print('true_list = ', true_list)
        print('number of objects in ground truth = {}'.format(np.sum(true_list)))
        # Plotting the PR curve
        ap, p, r, _ = eval_utils.get_precision_recall(conf_list, true_list) 
        # print('len p = {}, len r = {}, ap = {}'.format(len(p), len(r), ap))
        f1  = 2 * (p * r) / (p + r + 0.000001)
        best_f1_idx = np.argmax(f1[1:]) + 1
        print('best_f1_idx = {}, p = {}, r = {}'.format(best_f1_idx, p[best_f1_idx], r[best_f1_idx]))

        #plt.plot(r[1:], p[1:], label='AP: {:.2f}; Dilation radius: {:.2f}'.format(
        #    ap, dilation_size))

        plt.plot(r[1:], p[1:], label='AP: {:.2f}; Dilation radius: {:.2f}'.format(
            ap, dilation_size))

        plt.plot(r[best_f1_idx], p[best_f1_idx], 'ro')
        plt.annotate(
            'Best F1 point (F1={:.2f})\nPrecision={:.2f}\nRecall={:.2f}'.format(
                f1[best_f1_idx],
                p[best_f1_idx], 
                r[best_f1_idx]
            ),
            (r[best_f1_idx] - 0, p[best_f1_idx] - 0)
        )

        plt.xlim([0, 1])
        plt.ylim([0, 1])
        plt.xlabel('Recall')
        plt.ylabel('Precision')
        plt.title('num_obj_{}'.format(np.sum(true_list))+save_title)
        plt.legend()
        plt.tight_layout()
        save_path = os.path.join(output_dir, save_title + '.png')
        plt.savefig(save_path, dpi=300)
        plt.close()
        # Save the pr values for future plotting
        print('shape of r', np.shape(r))
        print('shape of p', np.shape(p))
        # Saving the pr values
        pr = np.concatenate([np.reshape(r, [-1, 1]), np.reshape(p, [-1, 1])], axis=1)
        print('shape of pr', np.shape(pr))
        np.savetxt(save_path.replace('.png','.txt'), pr)
        # Saving the conf and label list values
        conf_label_pair = np.concatenate([np.reshape(conf_list, [-1, 1]), np.reshape(true_list, [-1, 1])], axis=1)
        np.savetxt(save_path.replace('.png','_conf_label_pair.txt'), conf_label_pair)

        # Plotting the ROC curve
        fpr, tpr, thresholds = metrics.roc_curve(true_list, conf_list, pos_label=1)
        auroc = metrics.auc(fpr, tpr)
        f = plt.figure(figsize=(8,8))
        plt.plot(fpr, tpr,label='AUROC={:.2f}'.format(auroc))
        plt.xlabel('fpr')
        plt.ylabel('tpr')
        plt.legend()
        plt.xlim([0, 1])
        plt.ylim([0, 1])
        save_path = os.path.join(output_dir, save_title + 'ROC.png')
        plt.title('ROC_'+save_title)
        plt.savefig(save_path, dpi=300)

        if calculate_area:
            # Plotting the normalized ROC curve
            negative_class_num = np.sum(np.equal(true_list, 0))
            normalizing_factor = negative_class_num / np.sum(area_list)
            normalized_fpr = fpr * normalizing_factor           # The normalized fpr value
            f = plt.figure(figsize=(8,8))
            plt.plot(normalized_fpr, tpr)
            plt.xlabel('normalized_fpr, #/m^2')
            plt.ylabel('tpr')
            plt.ylim([0, 1])
            save_path = os.path.join(output_dir, save_title + 'normalized_ROC.png')
            plt.title('normalized_ROC_'+save_title)
            plt.savefig(save_path, dpi=300)
            nfpr_tpr_pair = np.concatenate([np.reshape(normalized_fpr, [-1, 1]), np.reshape(tpr, [-1, 1]), np.reshape(normalizing_factor * np.ones_like(tpr), [-1, 1])], axis=1)
            np.savetxt(save_path.replace('.png','_nfpr_tpr_pair.txt'), nfpr_tpr_pair)


def take_pair_wise_object_pr(i, j, min_region, dilation_size, min_th, iou_th, trail=0):
    ###############################
    # The 20m ensemble test group #
    ###############################
    prefix = 'model_d{}_test_d{}'.format(j, i)
    gt_dir = '/scratch/sr365/Catalyst_data/every_20m/d{}/annotations'.format(i)  
    output_dir = '/scratch/sr365/PR_curves/dx_dx_test_set_best_model/iou_th_{}_min_th_{}_dila_{}'.format(iou_th, min_th, dilation_size)
    conf_dir = '/scratch/sr365/Catalyst_data/every_20m/d{}/images/catalyst_from_ct_d{}/best_model/'.format(i, j)

    save_title = prefix + '_dial_{}_mireg_{}_mith_{}_iou_th_{}'.format(dilation_size, min_region, min_th, iou_th)
    conf_dir_list = [conf_dir]
    # creat folder if not exist
    if not os.path.exists(output_dir):
        try:
            os.makedirs(output_dir)
        except:
            print('there is exception in os.makedirs step')
    
    # Get tile name list from conf_dir_list
    tile_name_list = ['_'.join(f.split('_')[:-1]) for f in os.listdir(conf_dir)]
    print(tile_name_list)
    # Get the list of ground truth
    gt_list = [io.imread(os.path.join(gt_dir, f+'.png')) for f in tile_name_list]
    gt_dict = dict(zip(tile_name_list, gt_list))
    plot_PR_curve(min_region=min_region, dilation_size=dilation_size, link_r=0, min_th=min_th, iou_th=iou_th, 
                conf_dir_list=conf_dir_list, tile_name_list=tile_name_list, gt_dict=gt_dict, save_title=save_title, output_dir=output_dir)



def take_object_pr_RTI(output_dir, conf_dir, gt_dir, prefix,  min_region, dilation_size, min_th, iou_th):
    save_title = prefix + '_dial_{}_mireg_{}_mith_{}_iou_th_{}'.format(dilation_size, min_region, min_th, iou_th)
    conf_dir_list = [conf_dir]
    # creat folder if not exist
    if not os.path.exists(output_dir):
        try:
            os.makedirs(output_dir)
        except:
            print('there is exception in os.makedirs step')
    # Get tile name list from conf_dir_list
    tile_name_list = ['_'.join(f.split('_')[:-1]) for f in os.listdir(conf_dir)]
    # Get the list of ground truth
    gt_list = [io.imread(os.path.join(gt_dir, f+'.png')) for f in tile_name_list]
    gt_dict = dict(zip(tile_name_list, gt_list))
    plot_PR_curve(min_region=min_region, dilation_size=dilation_size, link_r=0, min_th=min_th, iou_th=iou_th, 
                conf_dir_list=conf_dir_list, tile_name_list=tile_name_list, gt_dict=gt_dict, 
                save_title=save_title, output_dir=output_dir, calculate_area=False)

if __name__ == '__main__':
    # The minimal number of pixels in the group
    min_region = 30
    # The dilation size for post-processing
    dilation_size = 5
    # The minimal threshold for the first step to eliminate noise
    min_th = 255*0.5
    # The threshold in IoU for counting object matching
    iou_th = 0.2
    # !!! Change this !!! The confidence map directory that was output from infer.py
    conf_dir_list = [r'data_raw/Exp1_1_resolution_buckets/test/d1/images/save_root/model1']
    # Make the tile list from confidence list
    tile_name_list = ['_'.join(f.split('_')[:-1]) for f in os.listdir(conf_dir_list[0])]
    # Make the grounth truth list
    gt_list = [io.imread(os.path.join(gt_dir, f+'.png')) for f in tile_name_list]
    # combine the ground truth list into a pair 
    gt_dict = dict(zip(tile_name_list, gt_list))
    # !!! Change this !!! The title of the saving PR curve
    save_title = '_dial_{}_mireg_{}_mith_{}_iou_th_{}'.format(dilation_size, min_region, min_th, iou_th)
    # !! Change this !!! The output directory
    output_dir = '../PR_curves/'
    plot_PR_curve(min_region=min_region, dilation_size=dilation_size, link_r=0, min_th=min_th, iou_th=iou_th, 
                conf_dir_list=conf_dir_list, tile_name_list=tile_name_list, gt_dict=gt_dict, 
                save_title=save_title, output_dir=output_dir)