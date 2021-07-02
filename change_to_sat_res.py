# This script changes the resolution of images to satellite resolution!

from posixpath import basename
import cv2
import numpy as np
import os
from skimage import io
import matplotlib.pyplot as plt
from skimage.util import pad

# Define folders
sat_res = 0.3  # Unit in meters per pixel

# The sat dir is where to save the imges
sat_dir = '/scratch/sr365/Catalyst_data/h2_satellite/'

# We assume one-to-one correspondenceon the img folder and the label folder
img_folder_big = '/scratch/sr365/Catalyst_data/h2/' 



# Helper functions
def get_current_res(height):
    return height * 6.17 / 4000 / 4.3

def get_height(img_name):
    print('current image name is: ', img_name)
    return int(img_name.split('height_')[-1].split('m')[0])

def change_resolution(img_name, target_res=sat_res):
    img = io.imread(img_name)
    height = get_height(img_name)                                       # Get height
    current_res = get_current_res(height)                               # Get current resolution
    scale = current_res / target_res                                       # Calculate scale 
    new_shape = (int(scale*img.shape[1]), int(scale*img.shape[0]))      # prepare new shape
    resized = cv2.resize(img, new_shape)                                # resize
    return resized

def mirroring_to_get_larger_size(img):
    """
    Since the MRS only support input of 512x512 patches, and the patches produced here
    with resolution adjustments are extraoridinary small, we need to mirror the patches bigger
    """
    pad_to_size = 512
    l, w = np.shape(img)[:2]
    if l < pad_to_size or w < pad_to_size:
        print('your image is smaller than {}, padding now to that size'.format(pad_to_size))
        img = pad(img, ((max(int(np.floor((pad_to_size - l)/2)), 0), max(int(np.ceil((pad_to_size-l)/2)), 0)), 
                        (max(int(np.floor((pad_to_size - w)/2)), 0), max(int(np.ceil((pad_to_size-w)/2)), 0)),
                        (0,0)), mode='symmetric')
    return img


def change_height_to_new_height(master_folder, target_height, dest_folder):
    """
    The function that changes the resolution of image folder (both images and annotations) to another resolution
    """
    # Get the folder names
    image_folder = os.path.join(master_folder, 'images')
    annotation_folder = os.path.join(master_folder, 'annotations')
    dest_image_folder = os.path.join(dest_folder, 'images')
    dest_annotation_folder = os.path.join(dest_folder, 'annotations')
    
    # If the folder does not exist, create this folder
    if not os.path.exists(dest_image_folder):
        os.makedirs(dest_image_folder)
    if not os.path.exists(dest_annotation_folder):
        os.makedirs(dest_annotation_folder)
    
    # Loop inside the folders to change
    for file in os.listdir(image_folder):
        # Make sure it is a image
        if not (file.endswith('.png') or file.endswith('.jpg') or file.endswith('.JPG')):
            continue
        # Change the resolution
        resized_image = change_resolution(img_name=os.path.join(image_folder, file), 
                                            target_res=get_current_res(target_height))
        # Save the image
        io.imsave(os.path.join(dest_image_folder, file), resized_image)
    
    # Loop inside the folders to change
    for file in os.listdir(annotation_folder):
        # Make sure it is a image
        if not (file.endswith('.png') or file.endswith('.jpg') or file.endswith('.JPG')):
            continue
        # Change the resolution
        resized_image = change_resolution(img_name=os.path.join(annotation_folder, file), 
                                            target_res=get_current_res(target_height))
        # Save the image
        io.imsave(os.path.join(dest_annotation_folder, file), resized_image)
    

def cut_img_into_even_pixel_number(img_folder_list):
    for img_folder in img_folder_list:
        for file in os.listdir(img_folder):
            # Make sure this is a image
            if not (file.endswith('.jpg') or file.endswith('.JPG') or file.endswith('.png')):
                continue
            # Read in the image and cut if it is not an even
            img_name = os.path.join(img_folder, file)
            print('reshapeing img {}'.format(img_name))
            img = io.imread(img_name)
            height, width, channel = np.shape(img)
            if height % 2 == 0 and width % 2 == 0:
                continue
            print('original size = {}x{}'.format(height, width))
            img = img[:height - height%2, :width - width%2, :]
            print('current size = {}'.format(np.shape(img)))
            io.imsave(img_name, img)

if __name__ == '__main__':
    #####################################################
    # Change the resolution to the satellite resolution #
    #####################################################
    # for folder in ['images','annotations']:
    #     # Get the image folder and save folder name
    #     img_dir = os.path.join(img_folder_big, folder) 
    #     save_dir = os.path.join(sat_dir, folder)

    #     # Make the save dir if not exist
    #     if not os.path.isdir(save_dir):
    #         os.makedirs(save_dir)
        
    #     # Loop over the images
    #     for file in os.listdir(img_dir):
    #         if not file.endswith('.JPG') and not file.endswith('.png'):
    #             continue
    #         img_full_name = os.path.join(img_dir, file)
    #         resized_img = change_resolution(img_full_name)
    #         resized_img = mirroring_to_get_larger_size(resized_img)
    #         io.imsave(os.path.join(save_dir, 'sat_' + file), resized_img)

    ####################################
    # Change resolution to the various #
    ####################################
    # for i in range(5, 13):          # The original resolution
    #     for j in range(5, 13):      # The target resolution
    #         if i == j:              # The same resolution therefore no need to proceed
    #             continue
    #         # The main function to change the resolution
    #         change_height_to_new_height(master_folder='/scratch/sr365/Catalyst_data/every_10m/{}0m/'.format(i), 
    #                                     target_height=10*j, 
    #                                     dest_folder='/scratch/sr365/Catalyst_data/every_10m_change_res/{}0m_resample_to_{}0m'.format(i, j))
    

    #################################
    # Change pixel into even number #
    #################################
    #folder_list = ['/scratch/sr365/Catalyst_data/every_10m_change_res/50m_resample_to_{}0m/images'.format(i) for i in range(6, 13)]
    folder_list = []
    for i in range(5, 13):
        for j in range(5, 13):
            if i == j:
                continue
            folder_list.append('/scratch/sr365/Catalyst_data/every_10m_change_res/{}0m_resample_to_{}0m/images'.format(i, j))
            folder_list.append('/scratch/sr365/Catalyst_data/every_10m_change_res/{}0m_resample_to_{}0m/annotations'.format(i, j))

    cut_img_into_even_pixel_number(folder_list)