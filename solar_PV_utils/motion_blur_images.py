# This function applies the motion blur to images and creates the artificial motion blurred images

import numpy as np
from skimage import io
import os
import shutil
import cv2

def apply_motion_blur_and_save(src_img_dir, dest_img_dir, motion_blur_kernel_size, src_img_postfix='.JPG', save_label_postfix='.png'):
    """
    This function applies motion blur to image and save them. The motion blur part is inspired by: https://www.geeksforgeeks.org/opencv-motion-blur-in-python/
    :param src_img_dir, dest_img_dir: The source image directory to get the images and destination directory to save the imagery
    :param motion_blur_kernel: The size of the kernel of convolution, the larger the larger the motion is (-1 is the motion blur pixel)
    :param src_img_postfix: The postfix of the images to be blurred
    :param save_label_postfix: The postfix of the label images to save, if None it means that don't copy the label images there (default is .png and yes copy them as well without touching them)
    """
    # Make sure destination exists
    if not os.path.isdir(dest_img_dir):
        os.makedirs(dest_img_dir)
    
    # Set up the kernels for blurred
    kernel = np.zeros((motion_blur_kernel_size, motion_blur_kernel_size))
    # Fill the middle row with ones.
    kernel[:, int((motion_blur_kernel_size - 1)/2)] = np.ones(motion_blur_kernel_size)
    # Normalize.
    kernel /= motion_blur_kernel_size

    for img_name in os.listdir(src_img_dir):
        # Only process the ones that has the pre-defined postfix
        if not img_name.endswith(src_img_postfix):
            continue
        cur_img_name = os.path.join(src_img_dir, img_name)
        # Add motion blur to this image
        img = cv2.imread(cur_img_name)
        img_mb = cv2.filter2D(img, -1, kernel)

        # Save this image
        cv2.imwrite(os.path.join(dest_img_dir, img_name), img_mb)

        # If the label exist and we want to copy it as well, copy it to new destination
        if save_label_postfix is not None:
            label_file = cur_img_name.replace(src_img_postfix, save_label_postfix)
            if os.path.exists(label_file):
                shutil.copyfile(label_file, label_file.replace(src_img_dir, dest_img_dir))
            

def copy_files_in_list(scr_dir, dest_dir, file_list):
    """
    Copy all files/dir in scr_dir to dest_dir if name in the file_list
    """
    for file_or_dir in os.listdir(scr_dir):
        if file_or_dir in file_list:
            shutil.copytree(os.path.join(scr_dir, file_or_dir), os.path.join(dest_dir, file_or_dir))


if __name__ == '__main__':
    
    #src_dir_list = ['/scratch/sr365/Catalyst_data/2021_03_21_15_C_90_test_set/H2_raw', '/scratch/sr365/Catalyst_data/2021_03_21_15_C_90_test_set/H3_raw']
    src_dir_list = ['/scratch/sr365/Catalyst_data/every_20m/d{}'.format(i) for i in range(1, 5)]
    
    motion_blur_list = [3, 5]#, 7, 9, 11, 21]

    for src_dir in src_dir_list:
        for motion_blur in motion_blur_list:
            # The test cases in 2021_03_21_15_C_90
            #dest_dir = src_dir + '_mb_{}'.format(motion_blur - 1)
            #apply_motion_blur_and_save(src_dir, dest_dir, motion_blur_kernel_size=motion_blur)
            
            # The training cases in h2, h3
            dest_dir = src_dir + '_mb_{}'.format(motion_blur - 1)
            apply_motion_blur_and_save( os.path.join(src_dir, 'images'), 
                                        os.path.join(dest_dir, 'images'), 
                                        motion_blur_kernel_size=motion_blur)
            # copy the rest of the files/dir to the dest dir
            file_list = ['annotations']
            copy_files_in_list(src_dir, dest_dir, file_list)