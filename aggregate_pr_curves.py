# This function aims to aggregate multiple PR curves together, plotted by the object.py curve

import matplotlib.pyplot as plt
from numpy.lib.npyio import save
import pandas as pd
import numpy as np
import os

from sklearn.utils.extmath import _incremental_mean_and_var
from adjustText import adjust_text
from seaborn.matrix import heatmap
from sklearn import metrics
import seaborn as sns; sns.set_theme()
from multiprocessing import Pool
from skimage import io
FPR_THRESHOLD = 3e-4
PR_curve_dir = ''
# The list of pr curves to aggreaget
"""
# H3 
aggregate_list = [ 'stationaryH3_img_H3_model_dialation_10_min_region_10_min_th125.txt',
                    'movingh3_model_h3_img_N_mode_dialation_10_min_region_10_min_th125.txt',
                    'movingh3_model_h3_img_S_mode_dialation_10_min_region_10_min_th125.txt' ]
                    #'moving_h3_model_h3_img_dialation_10_min_region_10_min_th125.txt',
legend_list = [ 'stationary','moving_N_mode','moving_S_mode' ]#'moving_all'


# H2
aggregate_list = [ 'stationaryH2_img_H2_model_dialation_10_min_region_10_min_th125.txt',
                    'movingh2_model_h2_img_N_mode_dialation_10_min_region_10_min_th125.txt',
                    'movingh2_model_h2_img_S_mode_dialation_10_min_region_10_min_th125.txt' ]
                    #'moving_h2_model_h2_img_dialation_10_min_region_10_min_th125.txt']
legend_list = ['stationary','moving_N_mode','moving_S_mode' ]#'moving_all']
"""

# Artificial blur
"""
aggregate_list = ['H2_raw_mb_5_dialation_10_min_region_10_min_th125.txt', 
                    'H2_raw_mb_4_dialation_10_min_region_10_min_th125.txt',
                    'H2_raw_mb_3_dialation_10_min_region_10_min_th125.txt',
                    'H2_raw_mb_2_dialation_10_min_region_10_min_th125.txt',
                    'H2_raw_mb_1_dialation_10_min_region_10_min_th125.txt']

# Artificial blur
aggregate_list = ['H3_raw_mb_5_dialation_10_min_region_10_min_th125.txt', 
                    'H3_raw_mb_4_dialation_10_min_region_10_min_th125.txt',
                    'H3_raw_mb_3_dialation_10_min_region_10_min_th125.txt',
                    'H3_raw_mb_2_dialation_10_min_region_10_min_th125.txt',
                    'H3_raw_mb_1_dialation_10_min_region_10_min_th125.txt']
legend_list = ['blur_{}'.format(i) for i in range(5, 0, -1)]#'moving_all']
"""

def draw_on_one_plot():
    # Cross-resolution evaluation
    i = 4
    aggregate_list = ['d{}_model_test_d{}ecresnet50_dcdlinknet_dscatalyst_d{}_lre1e-03_lrd1e-02_ep80_bs16_ds50_75_dr0p1_crxent1p0_softiou0p5_dialation_10_min_region_10_min_th125.txt'
                .format(i, j, i) for j in range(1,5) ]
    legend_list = ['test_on_d{}'.format(j) for j in range(1, 5)]

    title = 'cross_validation_model_{}'.format(i)
    save_name = os.path.join(PR_curve_dir, title+'.png')
    plt.figure(figsize=(8, 8))
    assert len(aggregate_list) == len(legend_list), 'The length of the PR curve file and legend should be the same, check!'
    texts=[]
    # Loop over the list
    for pr_curve_file, label in zip(aggregate_list, legend_list):
        pr = pd.read_csv(os.path.join(PR_curve_dir, pr_curve_file), header=None, sep=' ').values
        r, p = pr[:, 0], pr[:, 1]
        plt.plot(r[1:], p[1:], label=label)
        ind_maxr = np.argmax(r[1:])+1
        plt.plot(r[ind_maxr], p[ind_maxr], 'ro')
        pos_random = 0.5*np.random.uniform(size=1)
        # Add text box for max recall point
        #texts.append(plt.text(r[ind_maxr]-0.5*pos_random, p[ind_maxr]-0.5*pos_random, 'P={:.2f}\n R={:.2f}'.format(p[ind_maxr], r[ind_maxr]),fontsize=8))
        texts.append(plt.annotate('P={:.2f}\n R={:.2f}'.format(p[ind_maxr], r[ind_maxr]),
        xy=(r[ind_maxr], p[ind_maxr]), xytext=(r[ind_maxr]-0.5*np.random.uniform(size=1), p[ind_maxr]-0.5*np.random.uniform(size=1)), fontsize=8, arrowprops={'arrowstyle':'->'}))
    print(texts)
    #adjust_text(texts)
    plt.legend()
    plt.title(title)
    plt.xlim([0, 1])
    plt.ylim([0, 1])
    plt.xlabel('recall')
    plt.ylabel('precision')
    plt.savefig(save_name)

def get_num_rows_index_start_from_dir_name(PR_curve_dir):
    """
    This function gives the number of rows and starting index by looking at the directory name
    :dx means this is a 20m interval, 10_meter means this is a 10m interval
    """   
    if 'dx' in PR_curve_dir:
        # The 20m group
        num_rows = 4        # Number of rows in the plot
        index_start = 1     # The start of the index
        return 4, 1
    elif '10_meter' in PR_curve_dir:
        # The 10m group
        num_rows = 8        # Number of rows in the plot
        index_start = 5     # The start of the index
        return 8, 5
    else:
        print('In function get_num_rows_index_start_from_dir_name, you do not have correct format, folder name={}'.format(PR_curve_dir))


def draw_pairwise_PR_curves(PR_curve_dir, mode='PR', save_name='pair_wise_PR.png', 
                dial=10, minreg=30, mith=2, iou_th=0.8, plot_pairwise=False, fpr_threshold=FPR_THRESHOLD):
    """
    The function that does the pairwise plot of PR curve
    :param: mode= One of the rests:
    AUPR: The PR curve pairwise plus the Average Precision, which is the AUC of PR curve
    AUROC: The ROC curve pairwise plus the AUROC, which is AUC of ROC curve
    F1PR: PR + best F1
    max_recall_recall_PR: PR + maximum recal 
    max_recall_precision_PR: PR + precision at the maximum recal
    :param fpr_threshold: The fpr curoff threshold for setting the normalized AUROC curve
    """
    f = plt.figure(figsize=(32, 32))
    ax = plt.Axes(f, [0., 0., 1., 1.])
    ax.set_axis_off()
    f.add_axes(ax)
    z = 1   # counter for subplot index
    num_rows, index_start = get_num_rows_index_start_from_dir_name(PR_curve_dir)            # Get the number of rows and staring index
    aggreagate_stat_table = np.zeros([num_rows, num_rows])      # allocate a table for recording the aggregate statistics
    for i in range(index_start, index_start + num_rows, 1):
        for j in range(index_start, index_start + num_rows, 1):
            #name = 'd{}_model_test_d{}ecresnet50_dcdlinknet_dscatalyst_d{}_lre1e-03_lrd1e-02_ep80_bs16_ds50_75_dr0p1_crxent1p0_softiou0p5_dialation_10_min_region_10_min_th125.txt'.format(i, j, i) 
            if num_rows == 4:
                # The 20m groups
                name = 'model_d{}_test_d{}_dial_{}_mireg_{}_mith_{}_iou_th_{}.txt'.format(i, j, dial, minreg, mith, iou_th)
            elif num_rows == 8:
                # The 10m groups
                # name = 'model_{}0m_test_{}0m_dial_{}_mireg_{}_mith_{}_iou_th_{}.txt'.format(i, j, dial, minreg, mith, iou_th)
                name = 'BW_test_model_{}0m_test_{}0m_dial_{}_mireg_{}_mith_{}_iou_th_{}.txt'.format(i, j, dial, minreg, mith, iou_th)
            if 'ROC' in mode:
                # name = 'model_{}0m_test_{}0m_dial_10_mireg_10_mith5_conf_label_pair.txt'.format(i, j)
                name = name.replace('.txt', '_conf_label_pair.txt')
            value = pd.read_csv(os.path.join(PR_curve_dir, name), header=None, sep=' ').values
            if plot_pairwise:           # Save time 
                ax = plt.subplot(num_rows, num_rows,z)
            ###########
            # PR part #
            ###########
            if 'PR' in mode:
                r, p = value[:, 0], value[:, 1]       # Get P, R values
                if plot_pairwise:           # Save time 
                    plt.plot(r[1:], p[1:])          # Plot the P, R curve
                if 'AU' in mode:
                    ap = metrics.auc(r[1:], p[1:])    # Aera under PR curve, which is the Average Precision
                    aggreagate_stat_table[i-index_start, j-index_start] = ap
                elif 'max_recall_recall' in mode:
                    aggreagate_stat_table[i-index_start, j-index_start] = r[1]
                elif 'max_recall_precision' in mode:
                    aggreagate_stat_table[i-index_start, j-index_start] = p[1]
                elif 'F1' in mode:
                    f1  = 2 * (p * r) / (p + r + 0.000001)
                    best_f1 = np.max(f1[1:])
                    aggreagate_stat_table[i-index_start, j-index_start] = best_f1
            ############
            # ROC part #
            ############
            elif 'ROC' in mode:
                conf_list, true_list = value[:, 0], value[:, 1].astype('int')
                fpr, tpr, thresholds = metrics.roc_curve(true_list, conf_list, pos_label=1)
                if plot_pairwise:           # Save time 
                    plt.plot(fpr, tpr)
                if 'AU' in mode:
                    auroc = metrics.auc(fpr, tpr)
                    aggreagate_stat_table[i-index_start, j-index_start] = auroc
                if 'normalized_fpr' in mode:
                    normalized_name = name.replace('_conf_label_pair', 'normalized_ROC_nfpr_tpr_pair')
                    value = pd.read_csv(os.path.join(PR_curve_dir, normalized_name), header=None, sep=' ').values
                    # print(normalized_name, np.shape(value))
                    valid_index = value[:, 0] < fpr_threshold
                    normalized_fpr, tpr = value[valid_index, 0] / value[0, 2] , value[valid_index, 1]
                    AUnROC = metrics.auc(normalized_fpr, tpr) / normalized_fpr[-1]          # Divide by the last value of fpr
                    # print(max_fpr)
                    aggreagate_stat_table[i-index_start, j-index_start] = AUnROC
            if plot_pairwise:           # Save time 
                z += 1      # Add the counter
                plt.xlim([0, 1])
                plt.ylim([0, 1])
                plt.title('model{}0m->test{}0m'.format(i, j))
    if plot_pairwise:           # Save time 
        plt.savefig(os.path.join(PR_curve_dir, os.path.basename(PR_curve_dir) + '_' + save_name))

    # Plot the aggreagate value in a heatmap
    f = plt.figure()
    ax = plt.Axes(f, [0., 0., 1., 1.])
    ax.set_axis_off()
    f.add_axes(ax)
    # plt.xticks(ticks=np.arange(num_rows), labels=['{}0m'.format(i) for i in range(index_start, index_start + num_rows)])
    # plt.yticks(ticks=np.arange(num_rows), labels=['{}0m'.format(i) for i in range(index_start, index_start + num_rows)])
    # hm = plt.imshow(aggreagate_stat_table, cmap='Blues',interpolation="nearest")
    # plt.colorbar(hm)
    if num_rows < 5 and 'normalized' not in mode:
        sns.set(font_scale=2)
    elif 'normalized_fpr' in mode and num_rows > 5:
        sns.set(font_scale=1)
    else: 
        sns.set(font_scale=1)
    ax = sns.heatmap(aggreagate_stat_table,annot=True, fmt=".0%", cmap="YlGnBu")
    if 'normalized_fpr' not in mode:
        plt.title(mode + '{}_iou_th_{}_min_th'.format(iou_th, mith))
    else:
        plt.title('number of false positives per km^2 {}_iou_th_{}_min_th'.format(iou_th, mith))
    heatmap_name = os.path.join(PR_curve_dir, os.path.basename(PR_curve_dir) + '_' + mode)
    if 'normalized_fpr' in mode:                        # Change the name for the fpr cutoff threshold
        heatmap_name += '_fpr_thres_{}'.format(fpr_threshold)
    plt.savefig(heatmap_name + '.jpg')
    np.savetxt(heatmap_name +'.txt', aggreagate_stat_table)
    # Close all the figures
    plt.close('all')

def draw_a_lot_of_curves(PR_curve_dir='/scratch/sr365/PR_curves/', dial=None, minreg=None, mith=None, iou_th=None, fpr_threshold=FPR_THRESHOLD):
    # ROC oriented
    draw_pairwise_PR_curves(PR_curve_dir,mode='AUROC', save_name='pair_wise_ROC.png', dial=dial, minreg=minreg, mith=mith, iou_th=iou_th, plot_pairwise=True,  fpr_threshold=fpr_threshold)
    draw_pairwise_PR_curves(PR_curve_dir, mode='normalized_fpr_ROC', save_name='pair_wise_ROC.png', dial=dial, minreg=minreg, mith=mith, iou_th=iou_th, plot_pairwise=False,  fpr_threshold=fpr_threshold)
    # PC oriented
    draw_pairwise_PR_curves(PR_curve_dir, mode='AUPR', save_name='pair_wise_PR.png', dial=dial, minreg=minreg, mith=mith, iou_th=iou_th, plot_pairwise=True,  fpr_threshold=fpr_threshold)
    draw_pairwise_PR_curves(PR_curve_dir, mode='F1PR', save_name='pair_wise_PR.png', dial=dial, minreg=minreg, mith=mith, iou_th=iou_th, plot_pairwise=False,  fpr_threshold=fpr_threshold)
    draw_pairwise_PR_curves(PR_curve_dir, mode='max_recall_recall_PR', save_name='pair_wise_PR.png', dial=dial, minreg=minreg, mith=mith, iou_th=iou_th, plot_pairwise=False,  fpr_threshold=fpr_threshold)
    draw_pairwise_PR_curves(PR_curve_dir, mode='max_recall_precision_PR', save_name='pair_wise_PR.png', dial=dial, minreg=minreg, mith=mith, iou_th=iou_th, plot_pairwise=False,  fpr_threshold=fpr_threshold)

def get_iou_th_min_th_dila_from_name(folder, mode='str'):
    """
    This function gets the iou_th, min_th and dila values from the folder name
    :param: folder: The folder name of interest for extraction
    :param: mode: The format of extraction, currently support either 'str' or 'num'
    """
    iou_th = folder[folder.find('iou_th_')+7:].split('_')[0]
    min_th = folder[folder.find('min_th_')+7:].split('_')[0]
    dila = folder[folder.find('dila_')+5:].split('_')[0]
    if mode is not 'num' and mode is not 'str':
        exit('Your mode in get_iout_min_th_dila_from_name should be either str or num!! reset the argument pls!')
    if mode is 'num':
        return float(iou_th), int(min_th), int(dila)
    return iou_th, min_th, dila


def get_hyper_param_axis_unique(mother_folder, draw_type, x_axis, y_axis):
    """
    Get the list of hyper-parameter pairs from a mother folder
    """
    hyper_param_pairs_list = []
    for folder in os.listdir(mother_folder):            # Loop over the directory
        if not folder.startswith('iou_th_') or not os.path.isdir(os.path.join(mother_folder, folder)):            # skip the non folder
            continue
        iou_th, min_th, dila = get_iou_th_min_th_dila_from_name(folder, mode='num')         # Get the numerical value
        pair = [eval(x_axis), eval(y_axis)]
        hyper_param_pairs_list.append(pair)
    print(hyper_param_pairs_list)
    hyper_param_table = np.array(hyper_param_pairs_list)                            # Form a numpy array from list
    print('shape of the hyper_param_table is', np.shape(hyper_param_table))
    x_axis_unique = np.sort(np.unique(hyper_param_table[:, 0]))                              # Get the unique values
    y_axis_unique = np.sort(np.unique(hyper_param_table[:, 1]))                              # Get the unique values
    print('unique values in x axis', x_axis_unique)
    print('unique values in y axis', y_axis_unique)
    return x_axis_unique, y_axis_unique

def draw_things_into_one_plot(mother_folder, draw_type='AUPR', x_axis = 'iou_th', y_axis = 'min_th', 
                            save_name='agg_plot', fpr_threshold=FPR_THRESHOLD, dila=5):
    """
    This function draws a number of plots into a single plot so that I do not have to put them one-by-one into a damn ppt slide
    Note that the third variable has to stay constant in this process, otherwise the program would break
    :param: x_axis: The horizontal axis changing variable, the default is iou_th
    :param: y_axis: The vertical axis changing variable, the default is min_th
    """
    x_axis_unique, y_axis_unique = get_hyper_param_axis_unique(mother_folder, draw_type, x_axis, y_axis)
    x_len = len(x_axis_unique)
    y_len = len(y_axis_unique)
    # Start the plotting process
    f , axs = plt.subplots(y_len, x_len, figsize=[30, 30])
    #ax = plt.gca()
    # ax = f.add_subplot(111)
    # plt.sca(ax)
    # plt.xlabel(x_axis, fontsize=40)
    # plt.ylabel(y_axis, fontsize=40)
    # ax.grid(False)
    # ax.set_xticks([])
    # ax.set_yticks([])
    # For loop for plotting
    for ind_x, x_axis_value in enumerate(x_axis_unique):
        for ind_y, y_axis_value in enumerate(y_axis_unique):
            img_name = os.path.join(mother_folder, 'iou_th_{0}_min_th_{1}_dila_{2}/iou_th_{0}_min_th_{1}_dila_{2}_{3}.jpg'.format(x_axis_value, int(y_axis_value), dila, draw_type))
            if 'normalized' in draw_type:
                img_name = img_name.replace('.jpg', '_fpr_thres_{}.jpg'.format(fpr_threshold))
            img = io.imread(img_name)
            current_ax = axs[ind_y, ind_x]
            plt.sca(current_ax)
            # current_ax.axis('off')
            current_ax.grid(False)
            current_ax.set_xticks([])
            current_ax.set_yticks([])
            plt.xlabel(x_axis_value, fontsize=40)
            plt.ylabel(y_axis_value, fontsize=40)
            # plt.tight_layout()
            current_ax.imshow(img)
    # ax.text(0.5, 0.04, 'common xlabel', ha='center', va='center')
    # ax.text(0.06, 0.5, 'common ylabel', ha='center', va='center', rotation='vertical')
    # for ax in axs.flat:
    #     ax.set(xlabel='x-label', ylabel='y-label')
    for ax in axs.flat:
        ax.label_outer()
    plt.tight_layout()
    if 'normalized' in draw_type:
        save_name = os.path.join(mother_folder, save_name + draw_type + 'nfpr_thres_{}'.format(fpr_threshold)+ '.jpg')
    else:
        save_name = os.path.join(mother_folder, save_name + draw_type + '.jpg')
    plt.savefig(save_name)


def plot_mean_variance_from_aggregate(mother_dir_list, save_dir, draw_type='AUPR', x_axis = 'iou_th', y_axis = 'min_th', dila=5):
    """
    This function plots the mean and vairance of certain aggregate values, such as AUPR, nROC, etc...
    :param mother_dir_list: The mother dir list that this function operates at, which are the same quantity plotted but trained from different models
    :param mode: The mode of operation
    :param save_dir: The diectory to save the mean and variance plot to
    """
    if not os.path.isdir(save_dir):         # Make sure the folder exists
        os.makedirs(save_dir)
    # This operation assumes exactly the same folder structure as well as the iou naming structure
    x_axis_unique, y_axis_unique = get_hyper_param_axis_unique(mother_dir_list[0], draw_type, x_axis, y_axis)
    x_len, y_len = len(x_axis_unique), len(y_axis_unique)                                   # Get the length
    num_rows, index_start = get_num_rows_index_start_from_dir_name(mother_dir_list[0])            # Get the number of rows and staring index
    # Create the large master matrix which is 5 dimension, contains #models, #hyper_param_x_axis, #hyper_param_y_axis, #num_rows, #num_rows values
    the_master_matrix = np.zeros([len(mother_dir_list), len(x_axis_unique), len(y_axis_unique), num_rows, num_rows])
    for ind_dir, mother_dir in enumerate(mother_dir_list):
        for ind_x, x_axis_value in enumerate(x_axis_unique):
            for ind_y, y_axis_value in enumerate(y_axis_unique):
                # Get the performance compare matrix
                mat_name = os.path.join(mother_dir, 'iou_th_{0}_min_th_{1}_dila_{2}/iou_th_{0}_min_th_{1}_dila_{2}_{3}.txt'.format(x_axis_value, int(y_axis_value), dila, draw_type))
                mat = pd.read_csv(mat_name, header=None, sep=' ').values
                assert np.shape(mat) == np.shape(the_master_matrix)[-2:], 'The shape of your mat is {}, which is not correct!'.format(np.shape(mat))
                # Assign the matrix to the master matrix
                the_master_matrix[ind_dir, ind_x, ind_y, :, :] = mat
    # After getting the master matrix, do the mean and variance
    mean_mat = np.mean(the_master_matrix, axis=0)
    range_mat = np.max(the_master_matrix, axis=0) - np.min(the_master_matrix, axis=0)
    def plot_heat_map(x_len, y_len, x_axis_unique, y_axis_unique, value_table, save_name):
        # Plot the new mean and variance value
        f , axs = plt.subplots(y_len, x_len, figsize=[30, 30])
        for ind_x, x_axis_value in enumerate(x_axis_unique):
            for ind_y, y_axis_value in enumerate(y_axis_unique):
                current_ax = axs[ind_y, ind_x]
                plt.sca(current_ax)
                if num_rows < 5 and 'normalized' not in draw_type:
                    sns.set(font_scale=2)
                elif 'normalized_fpr' in draw_type and num_rows > 5:
                    sns.set(font_scale=1)
                else: 
                    sns.set(font_scale=1)
                ax = sns.heatmap(value_table[ind_x, ind_y],annot=True, fmt=".0%", cmap="YlGnBu")
                current_ax.grid(False)
                current_ax.set_xticks([])
                current_ax.set_yticks([])
                plt.xlabel(x_axis_value, fontsize=40)
                plt.ylabel(y_axis_value, fontsize=40)
                plt.title('iou_th_{0}_min_th_{1}_dila_{2}'.format(x_axis_value, int(y_axis_value), dila))
        for ax in axs.flat:
            ax.label_outer()
        plt.tight_layout()  
        plt.savefig(os.path.join(save_dir, draw_type + save_name))
    plot_heat_map(x_len, y_len, x_axis_unique, y_axis_unique, mean_mat, save_name='mean_mat.jpg')
    plot_heat_map(x_len, y_len, x_axis_unique, y_axis_unique, range_mat, save_name='range_mat.jpg')
    
def get_RTI_unique_iou_min_list(folder):
    """
    This function gets an input folder and get all the unique values  (sorted) of iou_th and min_th
    """
    iou_th_list = []
    min_th_list = []
    print('the master folder working at ', folder)
    for file in os.listdir(folder):
        # print('working on file ', file)
        if 'iou_th' not in file or 'mith' not in file:
            print('there is not keywords, skipping')
            continue
        iouth = float(os.path.splitext(file)[0].split('iou_th_')[-1].split('_')[0].split('ROC')[0])
        # print('iou th extracted: ', iouth)
        minth = int(os.path.splitext(file)[0].split('mith_')[-1].split('_')[0])
        # print('iou th extracted: ', minth)
        iou_th_list.append(iouth)
        min_th_list.append(minth)
    # print(iou_th_list)
    iou_unique_list, min_th_unique_list = np.sort(np.unique(np.array(iou_th_list))),  np.sort(np.unique(np.array(min_th_list)))
    dial = os.path.splitext(file)[0]
    dial = dial.split('dial_')[-1]
    dial = int(dial.split('_')[0])
    minreg = int(os.path.splitext(file)[0].split('mireg_')[-1].split('_')[0])
    # print(dial, minreg)
    return iou_unique_list, min_th_unique_list, dial, minreg

def plot_RTI_aggregate(mother_folder, mode='PR'):
    """
    This function aggregates the RTI heatmap plots together
    """
    print('mother folder working at ', mother_folder)
    for folder in os.listdir(mother_folder):
        if not os.path.isdir(os.path.join(mother_folder, folder)):
            continue
        # Get into each subfolder
        cur_folder = os.path.join(mother_folder, folder)
        # print('cur_folder', cur_folder)
        PR_curve_dir = cur_folder
        # Get unique values of iou_th and mith
        iou_th_list, min_th_list, dial, minreg = get_RTI_unique_iou_min_list(cur_folder)
        # print('unique values of iou th and min th are', iou_th_list, min_th_list)
        if 'PR' in mode:
            ap_table = np.zeros([len(iou_th_list), len(min_th_list)])
            max_recall_table =np.zeros([len(iou_th_list), len(min_th_list)])
            precision_at_max_recall_table = np.zeros([len(iou_th_list), len(min_th_list)])
            best_f1_table = np.zeros([len(iou_th_list), len(min_th_list)])
        elif 'ROC' in mode:
            AUROC_table = np.zeros([len(iou_th_list), len(min_th_list)])
        for ind_x, iou_th in enumerate(iou_th_list):
            for ind_y, mith in enumerate(min_th_list):
                name = '_dial_{}_mireg_{}_mith_{}_iou_th_{}.txt'.format(dial, minreg, mith, iou_th)
                if 'ROC' in mode:
                    name = name.replace('.txt', '_conf_label_pair.txt')
                value = pd.read_csv(os.path.join(PR_curve_dir, name), header=None, sep=' ').values
                if 'PR' in mode:
                    r, p = value[:, 0], value[:, 1]       # Get P, R values
                    # ap = metrics.auc(r[1:], p[1:])    # Aera under PR curve, which is the Average Precision
                    ap = metrics.auc(r, p)
                    max_recall = r[1]
                    precision_at_max_recall = p[1]
                    f1  = 2 * (p * r) / (p + r + 0.000001)
                    best_f1 = np.max(f1[1:])
                    ap_table[ind_x, ind_y] = ap
                    max_recall_table[ind_x, ind_y] = max_recall
                    precision_at_max_recall_table[ind_x, ind_y] = precision_at_max_recall
                    best_f1_table[ind_x, ind_y] = best_f1
                elif 'ROC' in mode:
                    conf_list, true_list = value[:, 0], value[:, 1].astype('int')
                    fpr, tpr, thresholds = metrics.roc_curve(true_list, conf_list, pos_label=1)
                    auroc = metrics.auc(fpr, tpr)
                    AUROC_table[ind_x, ind_y] = auroc
        if 'PR' in mode:
            table_list = ['ap_table','max_recall_table','precision_at_max_recall_table','best_f1_table']
        elif 'ROC' in mode:
            table_list = ['AUROC_table']    
        for table in table_list:
            f = plt.figure()
            ax = plt.gca()
            # ax = plt.Axes(f, [0., 0., 1., 1.])
            # ax.set_axis_off()
            f.add_axes(ax)
            sns.set(font_scale=2)
            # print('plotting ', table)
            # print('shape of PR table ', np.shape(ap_table))
            # print('it has shape ', np.shape(eval(table)))
            # ax = sns.heatmap(eval(table),annot=True, fmt=".0%", cmap="YlGnBu", xticklabels=iou_th_list, yticklabels=min_th_list)
            ax = sns.heatmap(eval(table),annot=True, fmt=".0%", cmap="YlGnBu", xticklabels=min_th_list, yticklabels=iou_th_list)
            plt.title(table)
            plt.ylabel('iou_th')
            plt.xlabel('min_th')
            heatmap_name = os.path.join('/', *PR_curve_dir.split('/')[:-1], os.path.basename(PR_curve_dir) + '_' + mode + table)
            plt.savefig(heatmap_name + '.jpg')
            np.savetxt(heatmap_name +'.txt', eval(table))
            plt.close('all')
        
if __name__ == '__main__':
    # The function to plot all the curves on one plot
    #draw_on_one_plot()
    # compare_dir_list = ['/scratch/sr365/PR_curves/dx_test_trail_0',
    #                     '/scratch/sr365/PR_curves/dx_test_trail_1',
    #                     '/scratch/sr365/PR_curves/dx_test_trail_2']
    
    # # Comparing various PR curve values
    compare_dir_list = ['/scratch/sr365/PR_curves/dx_train_trail_{}'.format(i) for i in range(4)]
    # compare_dir_list = ['/scratch/sr365/PR_curves/dx_train_trail_0',
    #                     '/scratch/sr365/PR_curves/dx_train_trail_1',
    #                     '/scratch/sr365/PR_curves/dx_train_trail_2']
    fpr_threshold_list = [1e-4, 2e-4, 3e-4, 4e-4, 5e-4]

    draw_type_list = ['AUPR', 'F1PR','max_recall_recall_PR','max_recall_precision_PR']
    # for fpr_thres in fpr_threshold_list:
    #     draw_things_into_one_plot('/scratch/sr365/PR_curves/dx_dx_test_set_ensemble', 
    #                             draw_type='normalized_fpr_ROC', fpr_threshold=fpr_thres)
    #     draw_things_into_one_plot('/scratch/sr365/PR_curves/dx_dx_train_set_ensemble',
    #                             draw_type='normalized_fpr_ROC', fpr_threshold=fpr_thres)
    for draw_type in draw_type_list:
        # draw_things_into_one_plot('/scratch/sr365/PR_curves/dx_dx_test_set_ensemble', draw_type=draw_type)
        # draw_things_into_one_plot('/scratch/sr365/PR_curves/dx_dx_train_set_ensemble', draw_type=draw_type)
        plot_mean_variance_from_aggregate(compare_dir_list, save_dir='/scratch/sr365/PR_curves/compare_plot_dx_train/',
                                            draw_type=draw_type)
    
    # mother_dir = '/scratch/sr365/PR_curves/dx_dx_test_set'
    # mother_dir = '/scratch/sr365/PR_curves/every_10_meter_test'
    # mother_dir = '/scratch/sr365/PR_curves/dx_test_trail_0'
    # mother_dir_list = ['/scratch/sr365/PR_curves/dx_test_trail_0',
    #                     '/scratch/sr365/PR_curves/dx_test_trail_1',
    #                     '/scratch/sr365/PR_curves/dx_test_trail_2',
    #                     '/scratch/sr365/PR_curves/dx_train_trail_0',
    #                     '/scratch/sr365/PR_curves/dx_train_trail_1',
    #                     '/scratch/sr365/PR_curves/dx_train_trail_2']
    #################
    # Gaia specific #
    #################
    mother_dir_list = ['/scratch/sr365/PR_curves/dx_dx_test_set_ensemble']#,
    #                   '/scratch/sr365/PR_curves/dx_dx_train_set_ensemble']
    num_cpu = 64

    # # Trying different iou_threshold and min threshold for confidence intensity
    # if num_cpu > 1:
    #     try: 
    #         pool = Pool(num_cpu)
    #         # The value agnostic version where directory is provided
    #         args_list = []
    #         for mother_dir in mother_dir_list:
    #             for folder in os.listdir(mother_dir):
    #                 if 'iou_th' not in folder:      # Make sure this is a iou_th and hyper-param sweeping folder
    #                         continue
    #                 for fpr_threshold in fpr_threshold_list:
    #                     iou_th, min_th, dila = get_iou_th_min_th_dila_from_name(folder)
    #                     args_list.append((os.path.join(mother_dir, folder), dila, 30, min_th, iou_th, fpr_threshold))         # 30 is the min region parameter
    #         pool.starmap(draw_a_lot_of_curves, args_list)
    #     finally:
    #         pool.close()
    #         pool.join()
    # else:
    #     for mother_dir in mother_dir_list:
    #         for folder in os.listdir(mother_dir):
    #             if 'iou_th' not in folder:      # Make sure this is a iou_th and hyper-param sweeping folder
    #                 continue
    #             for fpr_threshold in fpr_threshold_list:
    #                 iou_th, min_th, dila = get_iou_th_min_th_dila_from_name(folder)
    #                 draw_a_lot_of_curves(os.path.join(mother_dir, folder), dila, 30, min_th, iou_th, fpr_threshold)          # 30 is the min region parameter

    
    #################
    # quad specific #
    #################
    """
    #mother_dir_list = ['/scratch/sr365/PR_curves/dx_dx_train_set_ensemble']
    # fpr_threshold_list = [1e-4, 2e-4, 3e-4, 4e-4, 5e-4]
    
    # RTI Rwanda dataset
    num_cpu = 4

    # Trying different iou_threshold and min threshold for confidence intensity
    if num_cpu > 0:
        try: 
            pool = Pool(num_cpu)
            # The value agnostic version where directory is provided
            args_list = []
            for mode in ['PR', 'ROC']:
                for mother_list in ['/home/sr365/Gaia/PR_curves/test_object_only/', '/home/sr365/Gaia/PR_curves/train_object_only/']:
                    args_list.append((mother_list, mode))         # 30 is the min region parameter
            pool.starmap(plot_RTI_aggregate, args_list)
        finally:
            pool.close()
            pool.join()
    """
    # draw_type_list = ['AUPR', 'F1PR','max_recall_recall_PR','max_recall_precision_PR']
    # try: 
    #     pool = Pool(num_cpu)
    #     # The value agnostic version where directory is provided
    #     args_list = []
    #     for mother_dir in mother_dir_list:
    #         # Various draw type
    #         for draw_type in draw_type_list:
    #             args_list.append((mother_dir, draw_type))         # 30 is the min region parameter
    #         for fpr_threshold in fpr_threshold_list:
    #             args_list.append((mother_dir, 'normalized_fpr_ROC', fpr_threshold))
    #     pool.starmap(draw_a_lot_of_curves, args_list)
    # finally:
    #     pool.close()
    #     pool.join()
