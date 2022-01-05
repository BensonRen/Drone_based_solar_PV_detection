# This function counts the number of panels in the ground truth label images and then gives the estimates
from numpy.core.defchararray import array, mod
from numpy.core.numeric import _moveaxis_dispatcher
from skimage import io
import os
import numpy as np
import pandas as pd
import imagesize
from mrs_utils.eval_utils import ObjectScorer, dummyfy, get_stats_from_group
from skimage.transform import resize
from shutil import copyfile

mother_dir = '/home/sr365/Gaia/labels/'

def count_panels(mother_dir, size_warning=3000):
    obj_scorer = ObjectScorer(min_region=5, min_th=0.5, dilation_size=1, link_r=0, eps=0)
    def get_num_solar_panels(lbl_name, obj_scorer):
        lbl = io.imread(lbl_name)[:, :, 0]
        reg_props = obj_scorer.get_object_groups(lbl)
        del lbl
        #print('This image has {} solar panels'.format(len(reg_props)))
        return [ a.area for a in reg_props]

    ################################################
    # Counting individual villages solar panel num #
    ################################################
    # Read in the ground truth labels
    # for folder in os.listdir(mother_dir):
    #     cur_folder = os.path.join(mother_dir, folder)
    #     # Go into all subfolders
    #     if not os.path.isdir(cur_folder):
    #         continue
    #     for phase_folder in os.listdir(cur_folder):
    #         cur_phase = os.path.join(cur_folder, phase_folder)
    #         if not os.path.isdir(cur_phase):
    #             continue
    #         PV_count = 0
    #         for files in os.listdir(cur_phase):
    #             if not files.endswith('.csv'):
    #                 continue
    #             data = pd.read_csv(os.path.join(cur_phase, files))
    #             data = data['Object'].values
    #             if len(data) > 0:
    #                 PV_count += data[-1]
    #         #     if not files.endswith('.png'):
    #         #         continue
    #         #     cur_file = os.path.join(cur_phase, files)
    #         #     # This is a lbl file
    #         #     num_panels = get_num_solar_panels(cur_file, obj_scorer)
    #         #     PV_count += num_panels
    #         print('Folder {} has {} solar panels'.format(cur_phase, PV_count))

    ##########################################################
    # Counting the whole annotation folder in aggregate form #
    ##########################################################
    size_list = []
    size_warn_list = []
    for file in os.listdir(mother_dir):
        # Skip if not 
        if not file.endswith('.png'):
            continue
        current_panel_list = get_num_solar_panels(os.path.join(mother_dir, file), obj_scorer)
        size_list.extend(current_panel_list)
        # Check for oversized solar panels and make sure they are correct
        if len(current_panel_list) > 0 and np.sum(np.array(current_panel_list) > size_warning) > 0:
            size_warn_list.append(file)
            print('Check!!! There is panel larger than {} pixels in : {}'.format(size_warning, file))
        print('finished counting {}, currently there are {} solar panels'.format(file, len(size_list)))
    print(size_list)
    #np.savetxt('/home/sr365/Gaia/Rwanda_RTI/panel_size_count.txt', size_list)

    print('printing the oversized solar panel files')
    for file_warn in size_warn_list:
        print(file_warn)
    print('Finished! Largest panel {} pixels, smallest panel {} pixels'.format(np.max(size_list), np.min(size_list)))


def check_labels_complete():
    # Checking for each label if there is a corresponding image
    folder_dir = '/home/sr365/Gaia/labels'
    img_folder = '/home/sr365/Gaia/rti_rwanda_cut_tiles_ps_8000'
    for folder in os.listdir(folder_dir):
        cur_folder = os.path.join(folder_dir)
        if not os.path.isdir(cur_folder):
            continue
        #print('entering folder', cur_folder)
        for subfolder in os.listdir(cur_folder):
            cur_subfolder = os.path.join(cur_folder, subfolder)
            if not os.path.isdir(cur_subfolder):
                continue
            #print('entering folder', cur_subfolder)
            for file in os.listdir(cur_subfolder):
                if '.png' not in file:
                    continue
                # Check for the same name in another folder
                new_name = os.path.join(img_folder, folder + '_' + subfolder, file)
                if not os.path.exists(new_name):
                    print('Thiere is only label but not image!! {}'.format(file))
                # # Check for .png
                # if '.png' in file:
                #     # Check whether the image is there
                #     if not os.path.exists(os.path.join(folder_dir, file.replace('.png', '.JPG'))):
                #         print('Thiere is only label but not image!! {}'.format(file))
                # elif '.JPG' in file:
                #     # Check whether the label is there
                #     if not os.path.exists(os.path.join(folder_dir, file.replace('.JPG', '.png'))):
                #         print('Thiere is only image but not label!! {}'.format(file))
                print(new_name)
    print('Finished, this is alright!')

    # Check whether each image has a label
    for folder in os.listdir(img_folder):
        cur_folder = os.path.join(img_folder, folder)
        if not os.path.isdir(cur_folder):
            continue
        #print('entering folder', cur_folder)
        for file in os.listdir(cur_folder):
            # Only take the .png one
            if not file.endswith('.png'):
                continue
            new_name = os.path.join(folder_dir, folder.replace('ed_Ph', 'ed/Ph'), file)
            #print(new_name)
            if not os.path.exists(new_name):
                print('Thiere is only image but not label!! {}'.format(file))

def check_RTI_image_label_size_match():
    """
    Due to cutting of RTI imagery, there are some of the images that does not have 8000x8000 patch size. 
    Therefore we are checking them here and adjusting the labels of them accordingly
    """
    check_folder = '/home/sr365/Gaia/Rwanda_RTI/RTI_data_set/geo_test'
    img_folder = os.path.join(check_folder, 'images')
    lbl_folder = os.path.join(check_folder, 'annotations')
    for file in os.listdir(img_folder):
        # Only check for the .png files
        if not file.endswith('.png'): 
            continue
        # Get the width and height
        width_img, height_img = imagesize.get(os.path.join(img_folder, file))
        width_lbl, height_lbl = imagesize.get(os.path.join(lbl_folder, file))
        if width_img == width_lbl and height_lbl == height_img:
            continue
        lbl_name = os.path.join(lbl_folder, file)
        # Until here, this means that the label size is mis-matched
        lbl_img = io.imread(lbl_name)
        # Just to make sure this is a blank label image, the size difference is simply due to the fact that it was blamk image
        assert np.sum(lbl_img) == 0, 'This is not an empty label image in the size checking function!'
        lbl_resize = resize(lbl_img, (width_img, height_img))
        io.imsave(lbl_name, lbl_resize)
        print('resizing {} to shape {}'.format(lbl_name, (width_img, height_img)))

def get_label_pixel_intensity_from_1_to_255(folder):
    """
    This function takes the pixel intensity of 1 to 255 for folder of labels
    """
    for file in os.listdir(folder):
        if not file.endswith('.png'): # Only work with .png files,  which is the only possible postfix for label files
            continue
        file_name = os.path.join(folder, file)
        print('processing image ', file_name)
        lbl = io.imread(file_name)
        if np.max(lbl) == 1:
            # This means that this is a label file and the max pixel intensity is 1
            lbl *= 255
            io.imsave(file_name, lbl)


def get_label_pixel_intensity_from_255_to_1(folder):
    """
    This function takes the pixel intensity of 255 to 1 for folder of labels
    """
    for file in os.listdir(folder):
        if not file.endswith('.png'): # Only work with .png files,  which is the only possible postfix for label files
            continue
        file_name = os.path.join(folder, file)
        print('processing image ', file_name)
        lbl = io.imread(file_name)
        if len(np.unique(np.reshape(lbl, [-1, 1]))) and np.max(lbl) == 255:
            # This means that this is a label file and the max pixel intensity is 1
            lbl[lbl == 255] = 1
            io.imsave(file_name, lbl)

def get_rid_of_low_information_images(size_limit=14*1024, mode='image'):
    """
    Since the RTI rwanda imagery is a rotated one to adjust for the geography coordinates, 
    there are potentially lots of images that have little content.
    Therefore, we would like to get rid of the ones that does not actually contain enough content for learning
    This is not a necessary step, but it helps with the overall training time

    This function works in the patches folder and directly deletes the ones that does not have enough content by file size
    """
    # patch_folder = '/home/sr365/Gaia/Rwanda_RTI/RTI_data_set/new_all/patches'
    patch_folder = '/home/sr365/Gaia/Rwanda_RTI/RTI_data_set/all_conatains_object/patches'
    #patch_folder = '/home/sr365/Gaia/Rwanda_RTI/RTI_data_set/train/patches'
    file_size_list = []
    for file in os.listdir(patch_folder):
        # Only work on .jpg files that are the images or labels
        if mode == 'image' and not file.endswith('.jpg'):
            continue
        if mode == 'label' and not file.endswith('.png'):
            continue
        filename = os.path.join(patch_folder, file)
        # Check the file size and see
        filesize = os.path.getsize(filename)
        if filesize < size_limit:
            if mode == 'image':
                os.remove(filename.replace('.jpg', '.png'))         # Delete its label
            else:
                os.remove(filename.replace('.png', '.jpg'))
            os.remove(filename)                                 # Delete the image itself
    #     file_size_list.append(filesize)
    # np.savetxt('file_size_list.txt', file_size_list)


def make_file_list_for_RTI_Rwanda():
    """
    This function makes the training and testing file list for RTI Rwanda imagery
    
    """
    # folder = '/home/sr365/Gaia/Rwanda_RTI/RTI_data_set/train'
    # folder = '/home/sr365/Gaia/Rwanda_RTI/RTI_data_set/all'
    # folder = '/home/sr365/Gaia/Rwanda_RTI/RTI_data_set/all_conatains_object'
    folder = '/home/sr365/Gaia/Rwanda_RTI/RTI_data_set/all_train_5_percent/train_patches'
    if 'test' in folder:
        save_file = os.path.join(folder, 'file_list_test.txt')
    elif 'train' in folder:
        save_file = os.path.join(folder, 'file_list_train.txt')
    else:
        print("Your make_file_for_RTI_dataset does not have train or test in your folder name")
        save_file = os.path.join(folder, 'file_list_raw.txt')
    with open(save_file, 'a') as f:
        for file in os.listdir(os.path.join(folder, 'patches')):
            if not file.endswith('.jpg'):
                continue
            f.write(file)
            f.write(' ')
            f.write(file.replace('.jpg', '.png'))
            f.write('\n')


def sub_sample_randomly_image_label_pair(sample_size=0.005, mode='mv'):
    """
    This function subsamples a random portion of the image and label pair for the RTI dataset
    """
    source_folder = '/home/sr365/Gaia/Rwanda_RTI/RTI_data_set/geo_test/patches/'
    dest_folder = '/home/sr365/Gaia/Rwanda_RTI/RTI_data_set/geo_test_sample_10_percent/patches/'
    for file in os.listdir(source_folder):
        if not file.endswith('.png'):
            continue
        if np.random.uniform(size=1) < sample_size:
            if mode == 'mv':
                os.rename(os.path.join(source_folder, file), os.path.join(dest_folder, file))
                os.rename(os.path.join(source_folder, file.replace('.png', '.jpg')), os.path.join(dest_folder, file.replace('.png','.jpg')))
            elif mode == 'cp':
                copyfile(os.path.join(source_folder, file), os.path.join(dest_folder, file))
                copyfile(os.path.join(source_folder, file.replace('.png', '.jpg')), os.path.join(dest_folder, file.replace('.png','.jpg')))
            

def rename_infered_folder():
    """
    Cuz the inference structure, usually underneath the image/test_domain_trail there is a long and useless 
    model name like ecresnet50_dcdlinknet_dscatalyst_d1_lre1e-03_lrd1e-02_ep80_bs16_ds50_75_dr0p1_crxent1p0_softiou0p5
    This function aims to remove that folder name and move everything upwards
    """
    # Define the list of folders to lift upward (delete the model name and get everything up)
    # folder_to_move_upward_list = []
    # for i in range(1, 5):
    #     # Loop over the d{} images
    #     for j in range(1, 6):
    #         # Loop over the trails
    #         folder_to_move_upward_list.append('/scratch/sr365/models/catalyst_from_ct_d{}/d{}_trail_{}'.format(i, i, j))
    
    # RTI ones
    mother_folder_list = ['/home/sr365/Gaia/models/rwanda_rti_from_ct', '/home/sr365/Gaia/models/rwanda_rti_from_catalyst']
    folder_to_move_upward_list = []
    for mother_folder in mother_folder_list:
        for folder in os.listdir(mother_folder):
            folder_to_move_upward_list.append(os.path.join(mother_folder, folder))

    # Start the upward movement
    for folder in folder_to_move_upward_list:
        print('currently working on folder ', folder)
        if len(os.listdir(folder)) > 1:
            print('Folder {}, There is more than 1 model folder under your renaming folder, check again'.format(folder))
            continue
        model_folder = os.path.join(folder, os.listdir(folder)[0])
        # move out all the files
        for files in os.listdir(model_folder):
            os.rename(os.path.join(model_folder, files), os.path.join(folder, files))
        # make sure that the folder dir is empty now
        assert len(os.listdir(model_folder)) == 0, 'The model folder should be empty, which is not?'
        # Delete the original model folder
        os.rmdir(model_folder)
        
def remove_long_model_middle_folder(mother_folder):
    """
    This function removes the long middle name of a model
    """
    for folder in os.listdir(mother_folder):
        cur_folder = os.path.join(mother_folder, folder)
        print('currently working in folder', cur_folder)
        assert len(os.listdir(cur_folder)) == 1, 'There are more than 1 file in your folder, check again!'
        middle_name = os.listdir(cur_folder)[0]
        for files in os.listdir(os.path.join(cur_folder, middle_name)):
            # Move everything out
            os.rename(os.path.join(cur_folder, middle_name, files), os.path.join(cur_folder, files))


def copy_according_to_file_list(folder, file_list_path, dest_folder):
    """
    The funciton to copy files according to a file list to destination folder
    """
    # Creat the folder if not exist
    if not os.path.isdir(os.path.join(folder, dest_folder)):
        os.makedirs(os.path.join(folder, dest_folder))
    # Get the file list into list
    with open(os.path.join(folder, file_list_path), "r") as file:
        file_list = file.readlines()
    # Loop over the file list and 
    for line in file_list:
        img, lbl = line.split(' ')[0], line.split(' ')[1]
        lbl = lbl.replace("\n", '')
        print('moving {} and {} to new folder {}'.format(img, lbl, dest_folder))
        copyfile(os.path.join(folder, 'patches', img), os.path.join(folder, dest_folder, img))
        copyfile(os.path.join(folder, 'patches', lbl), os.path.join(folder, dest_folder, lbl))
    

def rename_PR_curves(mother_folder):
    """
    Renaming all the PR curves file so that they can be put into the same folder and downloaded
    """
    for folder in os.listdir(mother_folder):
        cur_folder = os.path.join(mother_folder, folder)
        # Skip if this is not a folder
        if not os.path.isdir(cur_folder):
            continue
        for file in os.listdir(cur_folder):
            cur_file = os.path.join(cur_folder, file)
            # Skip if this is not a regular file (should be only .txt, .pngs)
            if not os.path.isfile(cur_file):
                continue
            os.rename(cur_file, os.path.join(mother_folder, folder + file))


def post_processing(mother_folder, min_region=10, min_th=0.5, 
                    dilation_size=5, link_r=0, eps=2, 
                    operating_object_confidence_thres=0.9):
    """
    For the Rwanda imagery predicted confidence map, do the post-processing just like the evaluation does
    """
    # Loop over all the prediction maps
    for conf_map_name in os.listdir(mother_folder):
        # Skip and only work on the conf_map
        if 'conf.png' not in conf_map_name:
            continue
        print('post processing {}'.format(conf_map_name))
        # Read this conf map
        conf_map = io.imread(os.path.join(mother_folder, conf_map_name))
        # Rescale to 0, 1 interval
        conf_map  = conf_map / 255
        # Create the object scorer to do the post processing
        obj_scorer = ObjectScorer(min_region, min_th, dilation_size, link_r, eps)
        # Get the object groups after post processing
        group_conf = obj_scorer.get_object_groups(conf_map)
        # If no groups are around, continue
        if len(group_conf) == 0:
            continue
        # Loop over each individual groups
        for g_pred in group_conf:
            # Assigning flags for whether to keep this group
            g_pred.keep = True
            # Calculate the average confidence value (to be thresholded)
            _, conf = get_stats_from_group(g_pred, conf_map)
            # IF this object is smaller than what it should be
            if conf < operating_object_confidence_thres:
                # Throw this group away
                g_pred.keep = False
        # Threshold the objects by their confidence values
        group_conf = [a for a in group_conf if a.keep == True]
        # dummyfy the result into a binary plot
        conf_dummy = dummyfy(conf_map, group_conf)
        io.imsave(os.path.join(mother_folder, conf_map_name.replace('conf', 'conf_post_processed')), conf_dummy)

def check_over_sized_RTI_panel():
    """
    Check the oversized RTI panels check from the panel size function above
    """
    image_list = ['rti_rwanda_crop_type_raw_Rwakigarati_Processed_Phase3_y0x5920.png',
                    'rti_rwanda_crop_type_raw_Cyampirita_Processed_Phase2_y18177x0.png',
                    'rti_rwanda_crop_type_raw_Kabarama_Processed_Phase1_y21099x20755.png',
                    'rti_rwanda_crop_type_raw_Cyampirita_Processed_Phase3_y13510x7989.png',
                    'rti_rwanda_crop_type_raw_Kabarama_Processed_Phase1_y21099x13837.png',
                    'rti_rwanda_crop_type_raw_Cyampirita_Processed_Phase3_y20266x0.png',
                    'rti_rwanda_crop_type_raw_Kabarama_Processed_Phase1_y28132x20755.png',
                    'rti_rwanda_crop_type_raw_Kabarama_Processed_Phase1_y7033x20755.png',
                    'rti_rwanda_crop_type_raw_Cyampirita_Processed_Phase2_y12118x0.png',
                    'rti_rwanda_crop_type_raw_Cyampirita_Processed_Phase1_y19272x6668.png',
                    'rti_rwanda_crop_type_raw_Cyampirita_Processed_Phase1_y12848x6668.png',
                    'rti_rwanda_crop_type_raw_Cyampirita_Processed_Phase1_y12848x0.png',
                    'rti_rwanda_crop_type_raw_Cyampirita_Processed_Phase3_y13510x0.png',
                    'rti_rwanda_crop_type_raw_Kabarama_Processed_Phase1_y28132x27674.png',
                    'rti_rwanda_crop_type_raw_Rwakigarati_Processed_Phase2_y15095x10788.png',
                    'rti_rwanda_crop_type_raw_Cyampirita_Processed_Phase1_y19272x0.png',
                    'rti_rwanda_crop_type_raw_Kabarama_Processed_Phase1_y35165x6918.png',
                    'rti_rwanda_crop_type_raw_Kabarama_Processed_Phase1_y42199x20755.png',
                    'rti_rwanda_crop_type_raw_Kabarama_Processed_Phase3_y21343x17480.png',
                    'rti_rwanda_crop_type_raw_Cyampirita_Processed_Phase2_y12118x7888.png']
    image_dir = '/home/sr365/Gaia/Rwanda_RTI/RTI_data_set/all/'
    image_dest = '/home/sr365/Gaia/Rwanda_RTI/RTI_data_set/all/strange_large_panel'
    for image in image_list:
        for mode in ['images','annotations']:
            # Copy the file to a new directory for further investigation
            copyfile(os.path.join(image_dir, mode, image), os.path.join(image_dest, mode, image))
    




if __name__ == '__main__':
    count_panels('/scratch/sr365/Catalyst_data/every_20m/d1/annotations')
    #count_panels('/home/sr365/Gaia/Rwanda_RTI/RTI_data_set/all/strange_large_panel/annotations')
    # check_labels_complete()
    #check_RTI_image_label_size_match()
    #get_rid_of_low_information_images()
    #make_file_list_for_RTI_Rwanda()

    # Getting rid of the test files that does not contains any solar panels
    # An complete dark label has 334 Byte of information
    # get_rid_of_low_information_images(335, mode='label')

    # sub_sample_randomly_image_label_pair(mode='cp')

    # remove the model folder from the inference
    #rename_infered_folder()

    # Change the maximum pixel intensity of files
    # get_label_pixel_intensity_from_1_to_255('/home/sr365/Gaia/Rwanda_RTI/RTI_data_set/all/patches')
    # get_label_pixel_intensity_from_1_to_255('/home/sr365/Gaia/Rwanda_RTI/RTI_data_set/geo_train/patches')
    # get_label_pixel_intensity_from_1_to_255('/home/sr365/Gaia/Rwanda_RTI/RTI_data_set/geo_test/patches')
    # get_label_pixel_intensity_from_1_to_255('/home/sr365/Gaia/Rwanda_RTI/RTI_data_set/all_backup/patches')
    
    #get_label_pixel_intensity_from_1_to_255('/home/sr365/Gaia/Rwanda_RTI/RTI_data_set/all/annotations')
    #get_label_pixel_intensity_from_1_to_255('/home/sr365/Gaia/Rwanda_RTI/RTI_data_set/geo_train/annotations')
    #get_label_pixel_intensity_from_1_to_255('/home/sr365/Gaia/Rwanda_RTI/RTI_data_set/geo_test/annotations')
    #get_label_pixel_intensity_from_1_to_255('/home/sr365/Gaia/Rwanda_RTI/RTI_data_set/all_backup/annotations')
    # get_label_pixel_intensity_from_1_to_255('/home/sr365/Gaia/Rwanda_RTI/RTI_data_set/all_conatains_object/patches')

    # get_label_pixel_intensity_from_255_to_1('/home/sr365/Gaia/Rwanda_RTI/RTI_data_set/all_train_5_percent/patches')

    # Copying the train and test files to get
    # copy_according_to_file_list(folder='/home/sr365/Gaia/Rwanda_RTI/RTI_data_set/all_train_10_percent_half',
    #                             file_list_path='file_list_test.txt', dest_folder='test_patches')
    # copy_according_to_file_list(folder='/home/sr365/Gaia/Rwanda_RTI/RTI_data_set/all_train_10_percent_half',
    #                             file_list_path='file_list_train.txt', dest_folder='train_patches')
    
    #remove_long_model_middle_folder('/home/sr365/Gaia/models/rwanda_rti_from_catalyst/random_10_percent_split/')


    #rename_PR_curves('/home/sr365/Gaia/PR_curves/Catalyst_E4_train')

    # post_processing('/home/sr365/Gaia/Rwanda_RTI/RTI_data_set/geo_train/images/save_root/ecresnet50_dcdlinknet_dsrwanda_rti_lre1e-03_lrd1e-02_ep80_bs16_ds50_75_dr0p1_crxent1p0_softiou0p1')

    # Check oversized
    #check_over_sized_RTI_panel()
