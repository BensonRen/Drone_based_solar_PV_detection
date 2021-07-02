# This function serves as a image cutter for RTI images to smaller pieces so that the labelling process is easier

import numpy as np
import pandas as pd
import os
import rasterio
from data.data_utils import patch_tile_single
from mrs_utils import misc_utils
import gc
from multiprocessing import Pool

patch_size = 8000
test_file = '/scratch/sr365/RTI_Rwanda_full/rti_rwanda_crop_type_raw/rti_rwanda_crop_type_raw_Kinyaga_Processed_Phase3/image.tif'
dest_dir = '/home/sr365/Gaia/rti_rwanda_cut_tiles' + '_ps_{}'.format(patch_size)
# test_file = '/scratch/sr365/RTI_Rwanda_full/rti_rwanda_crop_type_raw/rti_rwanda_crop_type_raw_Kinyaga_Processed_Phase3/image.tif'
# dest_dir = '/scratch/sr365/RTI_Rwanda_full/cut_tiles' + '_ps_{}'.format(patch_size)


def read_tiff(tiff_file):
    """
    Read the tiff file into a big numpy array, meanwhile printing the overall information about this tile
    :param: tiff_file: The file name
    """
    # Open the tiff file
    dataset = rasterio.open(tiff_file)
    # output some basic information
    print('The tiff file opened is :{}\n There are {} channels in this file, width = {}, height = {} pixels\n'.format(tiff_file, len(dataset.indexes), 
           dataset.width, dataset.height))
    # Get the number of channels and initialize the big array
    num_channels = len(dataset.indexes) - 1     # The last layer is just the valid map, which should be discarded
    img_big = np.zeros([dataset.height, dataset.width, num_channels])
    # Loop over the number of channels
    for i in range(num_channels):
        print('reading channel {}'.format(i))
        img = dataset.read(i+1)
        img_big[:, :, i] = img
    # for real-application, no subsampling of image
    # img_subsampled = img_big[::sample_rate, ::sample_rate, :]
    
    # Delete the variable so that memory consumption is kept low
    del img
    gc.collect()        # Call the garbage collecter
    
    return img_big


def cut_img_big_into_tiles(tiff_file, dest_dir, ps, pad=0, overlap=0):
    """
    This function cut the big image RTI tiles that read from the read_tiff() into small tiles
    :param: tiff_file: The file name
    :param patch_size: The size of the output patch [ps, ps, 3]
    :param dest_dir: The destination directory to save the cut image tiles
    :param pad, overlap: The padding on the peripheral of the image, overlap between patches on the original image
    :return None: (It saves the tiles in the dest_dir)
    """
    patch_size = (ps, ps)
    #########################
    # Step 1: read the tiff #
    #########################
    img_big = read_tiff(tiff_file)
        
    ##########################################
    # Step 2: Cut the images into small tiles#
    ##########################################
    # Set destination folder name first
    RTI_img_label = tiff_file.split('/')[-2]
    patch_dir = os.path.join(dest_dir, RTI_img_label)
    print('saving your patch to patch_dir {}'.format(patch_dir))
    
    # Create the directory if that does not exist
    if not os.path.isdir(patch_dir):
        os.makedirs(patch_dir)
    
    # Loop over the patches cutted
    for rgb_patch, y, x in patch_tile_single(img_big, patch_size, pad, overlap):
        print('saving patch {} {} now'.format(x, y))
        img_patchname = '{}_y{}x{}.png'.format(RTI_img_label, int(y), int(x))
        misc_utils.save_file(os.path.join(
            patch_dir, img_patchname), rgb_patch.astype(np.uint8))
     

def cut_all(data_dir):
    """
    The master function that cuts all the RTI image data
    :param data_dir: The master diectory that contains the RTI imagery
    """
    # Loop over all the sub-directories
    for folder in os.listdir(data_dir):
        cur_folder = os.path.join(data_dir, folder)
        img_tif = os.path.join(cur_folder, 'image.tif')
        # Skip if either this is not a folder or the image.tif does not exist
        if not os.path.isdir(cur_folder) or not os.path.exists(img_tif):
            continue
        try:
            # Start cutting
            cut_img_big_into_tiles(img_tif, ps=patch_size, dest_dir=dest_dir, pad=0, overlap=0)
        except:
            print('The cutting for {} folder failed!!! Continuing now'.format(img_tif))
            continue
        

if __name__ == '__main__':
    #read_tiff(test_file)
    #cut_img_big_into_tiles(test_file, ps=8000, dest_dir=dest_dir, pad=0, overlap=0)
    cut_all('/home/sr365/Gaia/rti_rwanda_crop_type_raw/Cyampirita')
    #cut_all('/scratch/sr365/RTI_Rwanda_full/rti_rwanda_crop_type_raw')
    
