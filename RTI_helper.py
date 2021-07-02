# This function counts the number of panels in the ground truth label images and then gives the estimates
from numpy.core.numeric import _moveaxis_dispatcher
from skimage import io
import os
import numpy as np
import pandas as pd
import imagesize
from skimage.transform import resize
mother_dir = '/home/sr365/Gaia/labels/'

def count_panels(mother_dir):
    def get_num_solar_panels(lbl_name, obj_scorer):
        lbl = io.imread(lbl_name)[:, :, 0]
        reg_props = obj_scorer.get_object_groups(lbl)
        #print('This image has {} solar panels'.format(len(reg_props)))
        return len(reg_props)

    # Read in the ground truth labels
    for folder in os.listdir(mother_dir):
        cur_folder = os.path.join(mother_dir, folder)
        if not os.path.isdir(cur_folder):
            continue
        for phase_folder in os.listdir(cur_folder):
            cur_phase = os.path.join(cur_folder, phase_folder)
            if not os.path.isdir(cur_phase):
                continue
            PV_count = 0
            for files in os.listdir(cur_phase):
                if not files.endswith('.csv'):
                    continue
                data = pd.read_csv(os.path.join(cur_phase, files))
                data = data['Object'].values
                if len(data) > 0:
                    PV_count += data[-1]
            #     if not files.endswith('.png'):
            #         continue
            #     cur_file = os.path.join(cur_phase, files)
            #     # This is a lbl file
            #     num_panels = get_num_solar_panels(cur_file, obj_scorer)
            #     PV_count += num_panels

            print('Folder {} has {} solar panels'.format(cur_phase, PV_count))

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

def have_img_lbl_pair_in_same_folder():
    """
    During the moving of the image and labels, there are odd numbers of patches, which is really strange
    This function is to check whether there are pairs of image and label
    """
    check_folder = '/home/sr365/Gaia/Rwanda_RTI/RTI_data_set/train/patches'


def check_RTI_image_label_size_match():
    """
    Due to cutting of RTI imagery, there are some of the images that does not have 8000x8000 patch size. 
    Therefore we are checking them here and adjusting the labels of them accordingly
    """
    check_folder = '/home/sr365/Gaia/Rwanda_RTI/RTI_data_set/test'
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
        lbl_resize = resize(lbl_img, (width_img, height_img))
        io.imsave(lbl_name, lbl_resize)
        print('resizing {} to shape {}'.format(lbl_name, (width_img, height_img)))


def get_rid_of_low_information_images(size_limit=14*1024, mode='image'):
    """
    Since the RTI rwanda imagery is a rotated one to adjust for the geography coordinates, 
    there are potentially lots of images that have little content.
    Therefore, we would like to get rid of the ones that does not actually contain enough content for learning
    This is not a necessary step, but it helps with the overall training time

    This function works in the patches folder and directly deletes the ones that does not have enough content by file size
    """
    # patch_folder = '/home/sr365/Gaia/Rwanda_RTI/RTI_data_set/new_all/patches'
    patch_folder = '/home/sr365/Gaia/Rwanda_RTI/RTI_data_set/new_all/test_full_patches'
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
    # folder = '/home/sr365/Gaia/Rwanda_RTI/RTI_data_set/new_all'
    folder = '/home/sr365/Gaia/Rwanda_RTI/RTI_data_set/test_object_only'
    if 'test' in folder:
        save_file = os.path.join(folder, 'file_list_test.txt')
    elif 'train' in folder:
        save_file = os.path.join(folder, 'file_list_train.txt')
    else:
        print("Your make_file_for_RTI_dataset does not have train or test in your folder name")
    with open(save_file, 'a') as f:
        for file in os.listdir(os.path.join(folder, 'patches')):
            if not file.endswith('.jpg'):
                continue
            f.write(file)
            f.write(' ')
            f.write(file.replace('.jpg', '.png'))
            f.write('\n')


def sub_sample_randomly_image_label_pair(sample_size=0.1):
    """
    This function subsamples a random portion of the image and label pair for the RTI dataset
    """
    source_folder = '/home/sr365/Gaia/Rwanda_RTI/RTI_data_set/train/patches'
    dest_folder = '/home/sr365/Gaia/Rwanda_RTI/RTI_data_set/new_all/patches'
    for file in os.listdir(source_folder):
        if not file.endswith('.png'):
            continue
        if np.random.uniform(size=1) < sample_size:
            os.rename(os.path.join(source_folder, file), os.path.join(dest_folder, file))
            os.rename(os.path.join(source_folder, file.replace('.png', '.jpg')), os.path.join(dest_folder, file.replace('.png','.jpg')))

if __name__ == '__main__':
    # count_panels(mother_dir)
    # check_labels_complete()
    # check_RTI_image_label_size_match()
    #get_rid_of_low_information_images()
    make_file_list_for_RTI_Rwanda()

    # Getting rid of the test files that does not contains any solar panels
    # An complete dark label has 334 Byte of information
    # get_rid_of_low_information_images(335, mode='label')

    # sub_sample_randomly_image_label_pair()
