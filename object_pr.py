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


def plot_PR_curve(min_region, dilation_size, link_r, min_th, iou_th, conf_dir_list, tile_name_list, gt_dict, save_title, output_dir):
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
            conf_img, lbl_img = conf_dict[tile]/255, gt_dict[tile][:, :, 0]                                 # Get the confidence image and the label image
            conf_tile, true_tile = eval_utils.score(                                                        # Call a function in utils.score to score this
                conf_img, lbl_img, min_region=min_region, min_th=min_th/255, 
                dilation_size=dilation_size, link_r=link_r, iou_th=iou_th)    
            conf_list.extend(conf_tile)
            true_list.extend(true_tile)
            area = get_area_covered(tile, np.shape(gt_dict[tile])[:2])
            area_list.append(area)                      # Getting the area for the normalized ROC curve
        #print('true_list = ', true_list)
        print('number of objects in ground truth = {}'.format(np.sum(true_list)))
        # Plotting the PR curve
        ap, p, r, _ = eval_utils.get_precision_recall(conf_list, true_list) 
        print('len p = {}, len r = {}, ap = {}'.format(len(p), len(r), ap))
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
                np.max(f1),
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


def bulk_object_pr():
    # Some parameters to swipe
    #gt_dir = '/scratch/sr365/RTI_data/positive_class'
    #gt_dir = '/scratch/sr365/Catalyst_data/2021_03_21_15_C_90_test_set'
    #gt_dir = '/scratch/sr365/Catalyst_data/moving_imgs/labelled/cvs'
    gt_dir = '/scratch/sr365/Catalyst_data/d1/images'
    #gt_dir = '/scratch/sr365/Catalyst_data/d2/images'
    #gt_dir = '/scratch/sr365/Catalyst_data/d3/images'
    #gt_dir = '/scratch/sr365/Catalyst_data/d4/images'
    gt_dir_list = ['/scratch/sr365/Catalyst_data/every_10m/{}m/annotations'.format(i) for i in range(5, 13)]
    prefix = 'd1_model_test_d1'
    #model_img_pair = 'ecresnet50_dcunet_dsct_new_non_random_3_splits_lre1e-03_lrd1e-02_ep180_bs7_ds30_dr0p1_crxent7p0_softiou3p0'
    #model_img_pair = 'ecresnet50_dcunet_dsSDhist_lre1e-02_lrd1e-02_ep180_bs5_ds30_100_150_dr0p1_crxent0p7_softiou0p3'
    #model_img_pair='h3_moving'
    #model_img_pair = ''
    #model_img_pair = 'RTI_h3_mixed_training_150_class_weight'
    #model_img_pair = 'RTI_h2_mixed_training_best_hyper'
    # min_region_list = [10]
    # dilation_size_list = [10]
    # min_th_list = [200]
    output_dir = '/scratch/sr365/PR_curves/'
    #conf_dir_list = [
        # Catalyst dx part
    #    os.path.join(gt_dir, 'save_root', 'ecresnet50_dcdlinknet_dscatalyst_d1_lre1e-03_lrd1e-02_ep80_bs16_ds50_75_dr0p1_crxent1p0_softiou0p5')
        #os.path.join(gt_dir, 'save_root', 'ecresnet50_dcdlinknet_dscatalyst_d2_lre1e-03_lrd1e-02_ep80_bs16_ds50_75_dr0p1_crxent1p0_softiou0p5')
        #os.path.join(gt_dir, 'save_root', 'ecresnet50_dcdlinknet_dscatalyst_d3_lre1e-03_lrd1e-02_ep80_bs16_ds50_75_dr0p1_crxent1p0_softiou0p5')
        #os.path.join(gt_dir, 'save_root', 'ecresnet50_dcdlinknet_dscatalyst_d4_lre1e-03_lrd1e-02_ep80_bs16_ds50_75_dr0p1_crxent1p0_softiou0p5')
        
        # Motion part 
        #'/scratch/sr365/Catalyst_data/moving_imgs/labelled/img/save_root/h2_model_h2_img'
        #'/scratch/sr365/Catalyst_data/moving_imgs/labelled/img/save_root/h3_model_h3_img'
        #'/scratch/sr365/Catalyst_data/moving_imgs/labelled/img/save_root/h2',
        #'/scratch/sr365/Catalyst_data/moving_imgs/labelled/img/save_root/h3'
        # Motion N mode
        #'/scratch/sr365/Catalyst_data/moving_imgs/labelled/img/save_root/h3_model_h3_img_N_mode',
        #'/scratch/sr365/Catalyst_data/moving_imgs/labelled/img/save_root/h2_model_h2_img_N_mode'
        # Motion S mode
        #'/scratch/sr365/Catalyst_data/moving_imgs/labelled/img/save_root/h3_model_h3_img_S_mode',
        #'/scratch/sr365/Catalyst_data/moving_imgs/labelled/img/save_root/h2_model_h2_img_S_mode'
        # stationary part
        #'/scratch/sr365/Catalyst_data/2021_03_21_15_C_90_test_set/save_root/H2_img_H2_model'
        #'/scratch/sr365/Catalyst_data/2021_03_21_15_C_90_test_set/save_root/H3_img_H3_model'
        #'/scratch/sr365/Catalyst_data/2021_03_21_15_C_90_test_set/save_root/test'
        #'/scratch/sr365/Catalyst_data/2021_03_21_15_C_90_test_set/H3_raw_mb_{}/save_root/ecresnet50_dcdlinknet_dscatalyst_h3_lre1e-03_lrd1e-02_ep80_bs16_ds50_100_dr0p1_crxent1p0_softiou0p5'.format(i) for i in range(1, 6)
        #'/scratch/sr365/Catalyst_data/2021_03_21_15_C_90_test_set/H2_raw_mb_1/save_root/ecresnet50_dcdlinknet_dscatalyst_h2_lre1e-03_lrd1e-02_ep80_bs16_ds50_50_dr0p1_crxent1p0_softiou0p5'.format(i) for i in range(1, 6)
        #'/scratch/sr365/RTI_data/positive_class/save_root/' + model_img_pair
    #]
    #save_name_list = ['H3_raw_mb_{}'.format(i) for i in range(1, 6)]
    #save_name_list = ['H2_raw_mb_{}'.format(i) for i in range(1, 6)]
    link_r = 10
    min_region = 30
    dilation_size = 5
    min_th = 122
    iou_th = 0.6
    # Looping
    # for min_region in min_region_list:
    #     for dilation_size in dilation_size_list:
    #         for min_th in min_th_list:
    #             for conf_dir in conf_dir_list:
                    #save_title = model_img_pair + '_dialation_{}_min_region_{}_min_th{}'.format(dilation_size, min_region, min_th)
   
    # for i in range(5, 13):      # The image folder name
    #     for j in range(5, 13):  # The model folder name
    # for i in range(1, 5):      # The image folder name
    #      for j in range(1, 5):  # The model folder name
    
    ################################################
    # 2021.06.10 added for understanding PR process#
    ################################################
    # gt_list = [io.imread('/scratch/sr365/Catalyst_data/tmp_PR_test/2021_05_11_10_C_90DJI_0452_height_121m.png')]
    # tile_name_list = ['/scratch/sr365/Catalyst_data/tmp_PR_test/img/2021_05_11_10_C_90DJI_0452_height_121m']
    # conf_dir = '/scratch/sr365/Catalyst_data/tmp_PR_test/img'
    # gt_dict = dict(zip(tile_name_list, gt_list))
    # conf_dir_list = [conf_dir]
    # prefix = 'test_PR'
    # save_title = 'single_image'
    # plot_PR_curve(min_region, dilation_size, link_r, min_th)

def take_pair_wise_object_pr(i, j, min_region, dilation_size, min_th, iou_th):
    # # Every 10 meters
    # output_dir = '/scratch/sr365/PR_curves/every_10_meter_train/iou_th_{}_min_th_{}_dila_{}'.format(iou_th, min_th, dilation_size)
    # # output_dir = '/scratch/sr365/PR_curves/every_10_meter_test/iou_th_{}_min_th_{}_dila_{}'.format(iou_th, min_th, dilation_size)
    # gt_dir = '/scratch/sr365/Catalyst_data/every_10m/{}0m/annotations'.format(i)
    # conf_dir = '/scratch/sr365/Catalyst_data/every_10m/{}0m/images/save_root/{}0m_model'.format(i, j)
    # # conf_dir = '/scratch/sr365/Catalyst_data/every_10m/{}0m/images/test_set_BW_save_root/{}0m_model'.format(i, j)
    # #prefix = 'BW_test_model_{}0m_test_{}0m'.format(j, i)
    # prefix = 'train_model_{}0m_test_{}0m'.format(j, i)

    # Every 20 meters
    prefix = 'model_d{}_test_d{}'.format(j, i)
    gt_dir = '/scratch/sr365/Catalyst_data/d{}/annotations'.format(i)
    # output_dir = '/scratch/sr365/PR_curves/dx_dx_test_set/iou_th_{}_min_th_{}_dila_{}'.format(iou_th, min_th, dilation_size)
    output_dir = '/scratch/sr365/PR_curves/dx_dx_test_set_ensemble/iou_th_{}_min_th_{}_dila_{}'.format(iou_th, min_th, dilation_size)
    # output_dir = '/scratch/sr365/PR_curves/dx_train_trail_1/iou_th_{}_min_th_{}_dila_{}'.format(iou_th, min_th, dilation_size)
    # conf_dir = '/scratch/sr365/Catalyst_data/d{}/images/train_set/ecresnet50_dcdlinknet_dscatalyst_d{}_lre1e-03_lrd1e-02_ep120_bs16_ds50_75_dr0p1_crxent1p0_softiou0p5'.format(i, j)
    conf_dir = '/scratch/sr365/Catalyst_data/d{}/images/test_domain_ensembled_img_d{}_model_d{}'.format(i, i, j)
    # conf_dir = '/scratch/sr365/Catalyst_data/d{}/images/train_save_root_trail_1/ecresnet50_dcdlinknet_dscatalyst_d{}_lre1e-03_lrd1e-02_ep80_bs16_ds50_75_dr0p1_crxent1p0_softiou0p5'.format(i, j)
    #output_dir = '/scratch/sr365/PR_curves/dx_dx_test_set/iou_th_{}_min_th_{}_dila_{}'.format(iou_th, min_th, dilation_size)
    #conf_dir = '/scratch/sr365/Catalyst_data/d{}/images/train_set/ecresnet50_dcdlinknet_dscatalyst_d{}_lre1e-03_lrd1e-02_ep120_bs16_ds50_75_dr0p1_crxent1p0_softiou0p5'.format(i, j)
    
    save_title = prefix + '_dial_{}_mireg_{}_mith_{}_iou_th_{}'.format(dilation_size, min_region, min_th, iou_th)
    conf_dir_list = [conf_dir]
    #for conf_dir, save_name in zip(conf_dir_list, save_name_list):
    #    save_title = prefix + model_img_pair + '{}_dialation_{}_min_region_{}_min_th{}'.format(save_name, dilation_size, min_region, min_th)
        
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
                conf_dir_list=conf_dir_list, tile_name_list=tile_name_list, gt_dict=gt_dict, save_title=save_title, output_dir=output_dir)

if __name__ == '__main__':
    num_cpu = 64
    try: 
        pool = Pool(num_cpu)
        min_region = 30
        dilation_size = 5
        min_th = 2
        iou_th = 0.2
        args_list = []
        min_th_list = np.array([0.3, 0.5, 0.7])
        min_th_list = min_th_list * 255   
        print(min_th_list)
        for min_th in min_th_list:
            for iou_th in [0.2, 0.4, 0.6]:
            #for iou_th in [0.4]:
                min_th = int(min_th)        # Make sure it is a integer
                # Every 10 meters
                # for i in range(5, 13):
                #     for j in range(5, 13):
                # Every 20 meters
                for i in range(1, 5):
                    for j in range(1, 5):
                        args_list.append((i, j, min_region, dilation_size, min_th, iou_th))
        print(args_list)
        pool.starmap(take_pair_wise_object_pr, args_list)
    finally:
        pool.close()
        pool.join()
    

    # temporary for visualizatin purpose
    # take_pair_wise_object_pr(5, 5, 30, 5, 127, 0.3)
