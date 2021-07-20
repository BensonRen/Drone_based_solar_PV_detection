# This function/script is to get the color histogram of a set of images so that to compare the domain difference between 2 domains
import cv2
import numpy as np
import os
from matplotlib import pyplot as plt


def get_color_histogram(folder, save_name, save_dir, post_fix='.jpg' ,mask_option=False, normalize=True):
    """
    The main function to plot the comparison of color histogram of 2 folders
    :param folder: The folder containing the images
    :param save_dir: The directory to save the comparison plot of the color histogram
    :param save_name: The name of the color histogram to save
    :param post_fix: The postfix of the images you wish to color histogram for
    :param mask: Default False. The mask of the color histogram, if activated 
    :param normalize: The histogram should be normalized or not
    """
    print('your mask option is: ', mask_option)
    # Get the image name list
    img_name_list = [os.path.join(folder, a) for a in os.listdir(folder) if  a.endswith(post_fix)]
    # Initialize the image list and mask list, which are all Nones
    img_list, mask_list = [None]*len(img_name_list), [None]*len(img_name_list)
    # Put the images into the image list
    for ind, img_name in enumerate(img_name_list):
        img = cv2.imread(img_name)
        img_list[ind] = img
        # If the mask mode is on, which means we only calculate the solar panels color histogram
        if mask_option:
            # print('reading label file')
            mask_name = img_name.replace(post_fix, '.png')
            mask = cv2.imread(mask_name)[:, :, 0]
            # print('shape of the label mask = ', np.shape(mask))
            if np.max(mask) > 1:    #  Normalize the mask
                print('normalizing mask')
                mask = mask / np.max(mask)
            elif np.max(mask) == 1:
                print('the mask max is 1')
        else:
            # print('getting only the content pixels')
            # If we calculate all of them, we want to get rid of the total black pixles which is artifact from 
            mask = np.ones_like(img)[:, :, 0]       # Initialize the mask for pixels that has content
            mask[np.sum(img, axis=2) == 0] = 0                              # Get the total black pixels to 0
        # mask_list[ind] = np.expand_dims(mask.astype("uint8"), axis=2)
        mask_list[ind] = mask.astype("uint8")
    print('type of mask_list', type(mask_list))
    print('shape of mask_list', np.shape(mask_list))
    print('type of img_list', type(img_list))
    print('shape of img_list', np.shape(img_list))

    print('shape of mask_list[0]', np.shape(mask_list[0]))
    print('shape of img_list[0]', np.shape(img_list[0]))

    assert np.sum(mask_list) > 0, 'All your mask is empty! check again and redo, sum of mask list =  {}'.format(np.sum(mask_list))
    # Plotting part of the histogram
    color = ('b','g','r')
    # Initialize the histogram_list
    histr_list = [None] * 3
    for i,col in enumerate(color):      # Loop over color to get the histograms
        for j in range(len(img_list)):
            # histr_list[i] = cv2.calcHist(images=img_list,channels=[i], mask=np.array(mask_list),  histSize=[256], ranges=[0,256])
            histr = cv2.calcHist(images=[img_list[j]], channels=[i], mask=np.array(mask_list[j]),  histSize=[256], ranges=[0,256])
            # histr_list[i] = cv2.calcHist(images=img_list,channels=[i], mask=None,  histSize=[256], ranges=[0,256])
            if histr_list[i] is None:
                histr_list[i] = histr
            else:
                histr_list[i] += histr
    if normalize:
        print('normalizing the histogram of the rgb by the same amount')
        #print(histr_list)
        num_pix = np.sum(histr_list)
        print('number of pixels: ', num_pix)
        histr_list = histr_list / num_pix
    
    plt.figure()
    for i,col in enumerate(color):      # Loop over color to plot
        plt.plot(histr_list[i],color = col)
        plt.xlim([0,256])
    plt.xlabel('pixel intensity')
    plt.ylabel('normalized frequency')
    plt.title(save_name + ' color histogram')
    plt.savefig(os.path.join(save_dir, save_name + '_color_hist.png'))


if __name__ == '__main__':
    # Individual process
    # get_color_histogram('/home/sr365/Gaia/Rwanda_RTI/RTI_data_set/geo_test_sample_10_percent/patches', 'geo_test_sample_10%', 
    #                     save_dir='/home/sr365/Gaia/color_hist', mask_option=False)
    # get_color_histogram('/home/sr365/Gaia/Rwanda_RTI/RTI_data_set/geo_test_sample_10_percent/patches', 'geo_test_sample_10%_object', 
    #                     save_dir='/home/sr365/Gaia/color_hist', mask_option=True)
    

    # Batch process
    folder_list = ['/home/sr365/Gaia/Rwanda_RTI/RTI_data_set/geo_train/patches',
                    '/home/sr365/Gaia/Rwanda_RTI/RTI_data_set/geo_train/patches',
                    '/home/sr365/Gaia/Rwanda_RTI/RTI_data_set/geo_test/patches',
                    '/home/sr365/Gaia/Rwanda_RTI/RTI_data_set/geo_test/patches']
                    
    save_name_list = ['geo_train', 'geo_train_object', 'geo_test','geo_test_object']
    for folder, save_name in zip(folder_list, save_name_list):
        # If the name has object, then it is using the mask
        if 'object' in save_name:
            mask_option = True
        else:
            mask_option = False
        get_color_histogram(folder, save_name, save_dir='/home/sr365/Gaia/color_hist', mask_option=mask_option)