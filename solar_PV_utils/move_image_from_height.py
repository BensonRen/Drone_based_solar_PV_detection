# This function is to make it easier to move the photos by their heights

# H1:  < 45m 
# H2: 45m < H2 <85m
# H3: >85m

import numpy as np
import os
import shutil

move_or_copy = 'copy'

# Folder information
#source_folder = '/scratch/sr365/Catalyst_data/moving_imgs/labelled/img/save_root/h2'

height = 120
dest_folder = '/scratch/sr365/Catalyst_data/every_10m/{}m'.format(height)

source_folder_list = ['/scratch/sr365/Catalyst_data/d{}/images/'.format(i) for i in range(1, 5)]
"""
'/scratch/sr365/Catalyst_data/0112_Moreen',
'/scratch/sr365/Catalyst_data/2021_02_17_10_B_90',
'/scratch/sr365/Catalyst_data/2021_02_24_13_C_90',
'/scratch/sr365/Catalyst_data/2021_03_10_10_D_90',
'/scratch/sr365/Catalyst_data/2021_03_21_15_C_90',
'/scratch/sr365/Catalyst_data/2021_04_03_10_BW_90',
'/scratch/sr365/Catalyst_data/2021_04_12_10_BW_90',
'/scratch/sr365/Catalyst_data/2021_05_11_10_C_90']
"""
# H2 bound
#height_lower_bound = 45
#height_higher_bound = 85

# H3 bound
#height_lower_bound = 85
#height_higher_bound = 120

# D1 bound
# height_lower_bound = 45
# height_higher_bound = 65

# D2 bound
# height_lower_bound = 65
# height_higher_bound = 85

# D3 bound
# height_lower_bound = 85
# height_higher_bound = 105

# D4 bound
#height_lower_bound = 105
#height_higher_bound = 130

# temp bound
height_lower_bound = height-5
height_higher_bound = height+5

def get_height(img_name):
    """
    return the int height of a img
    """
    if 'height' in img_name:
        return int(img_name.split('height_')[-1].split('m')[0])
    else:
        print('Your file {} does not have height information in its name, aborting!'.format(img_name))
        return None

def move_img_from_height(source_folder, dest_folder, height_lower_bound, height_higher_bound, move_or_copy, motion_mode=None):
    """
    Move the image to the destination folder if the height of the image labelled is within the bound
    :param: source_/dest_folder: To move the images from and to folder
    :param: height_lower/higher_bound: The bounds of height to move those images
    :param: move_or_copy: A flag tht signifies either to move or copy the image [move, copy]
    :param: motion_mode: Default None, activated to [S(ports), N(ormal)] to move only the motion images of sports or normal mode
    """
    for file in os.listdir(source_folder):
        # ignore non image format
        if not file.endswith('.JPG') and not file.endswith('.png') and not file.endswith('jpg'):
            continue;
        
        
        # The motion mode
        if motion_mode is 'S' and 'S0' not in file:
            continue
        elif motion_mode is 'N' and 'N0' not in file:
            continue

        # Get height information
        height = get_height(file)
        
        # ignore if no height information or this is not in the interest group
        if height is None or height < height_lower_bound or height > height_higher_bound:
            continue;
        
        # Create the destination folder if not exist
        if not os.path.isdir(dest_folder):
            os.makedirs(dest_folder)

        # Copy
        if move_or_copy == 'copy':
            shutil.copyfile(os.path.join(source_folder, file), os.path.join(dest_folder, os.path.basename(source_folder) + file))
        elif move_or_copy == 'move':
            os.rename(os.path.join(source_folder, file), os.path.join(dest_folder, os.path.basename(source_folder) + file))
        else:
            print('Your move_or_copy option has to be either move or copy!!! aborting!')
            quit()


if __name__ == '__main__':
    # Move the images
    #move_img_from_height(source_folder, dest_folder, height_lower_bound, height_higher_bound, move_or_copy)

    # Move the images from videos (motion)
    # move_img_from_height(source_folder, dest_folder, height_lower_bound, height_higher_bound, move_or_copy, motion_mode='N')

    # Aggregate version of moving massive amount of images to new location
    for source_folder in source_folder_list:
        move_img_from_height(source_folder, dest_folder, height_lower_bound, height_higher_bound, move_or_copy='copy')
