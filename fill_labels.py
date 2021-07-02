# This funcion has 2 functions:
# 1. fills in the labels that are not defined in the folder 
# 2. identify the ones that does not have a label and prompt for delete (that were mis-put, classified into positive class)
# This is due to the sparsity of the target class in the dataset, particularly in the RTI_Rwanda test image setting

import os
import numpy as np
from PIL import Image
from skimage import io
import shutil

def generate_labels_for_negative_images(img_path):
    """
    This function takes the path of a image and returns a .png that is completely black with postfix of .png and in the same folder
    :param img_path: the path of the image that is either .jpg or .JPG
    """
    if not img_path.endswith('.JPG') and not img_path.endswith('.jpg'):
        print("This image path does not end with .jpg or .JPG, aborting check again!")
        quit()
    img = io.imread(img_path)
    if np.shape(img)[0] == 2:
        lbl = np.zeros(img[0].shape[:-1])
    else:
        lbl = np.zeros(img.shape[:-1])
    print('The shape of the label file is', np.shape(lbl))
    lbl_im= Image.fromarray(lbl).convert('RGB')
    save_name = img_path.replace('jpg','png').replace('JPG','png')
    lbl_im.save(save_name)


def generate_label_negative_folder(negative_folder):
    """
    Calls the generate_labels_for_negative_images function for the whole directory
    """
    for file in os.listdir(negative_folder):
        if not file.endswith('.JPG') and not file.endswith('.jpg'):
            continue
        generate_labels_for_negative_images(os.path.join(negative_folder, file))


def check_and_delete_the_unlabelled_ones(positive_img_dir, mode='move',move_dest='check_not_label/'):
    """
    This function checks whether a folder is fully labelled and print and possibly delete or move those files (to move_dest if option is move)
    Fully labelled: only takes in the folder that contains all the positive images, which is defined by there exist an target object in the image

    :param: positive_img_dir: The directory where only the positive images show exist and to check whether they are fully labelled
    :param: mode: The mode option: there are :
            'None': which only print out those files
            'move': which moves those ones to the move_dest
            'delete': which deletes those files
    :param: move_dest: the folder to move the not-labelled ones
    """
    for file in os.listdir(positive_img_dir):
        # only check those files that ends with .jpg or .JPG
        if not file.endswith('.jpg') and not file.endswith('.JPG'):
            continue
        corresponding_label_file_name = os.path.join(positive_img_dir, file.replace('.jpg','.png').replace('.JPG', '.png'))
        # Check if this file exist, if so then we are good!
        if os.path.isfile(corresponding_label_file_name):
            continue
        # If not, we need to do something depending on the option of mode
        if mode is None:
            # only print out the name of those files
            print('Image {} does not have a label with it, check it'.format(file))
        elif mode == 'move':
            # move it instead to the move_dest
            if not os.path.isdir(move_dest):
                os.makedirs(move_dest)
            move_file_name = os.path.join(move_dest, file)
            print('Image {} does not have a label with it, moving it to {}'.format(file, move_file_name))
            shutil.move(os.path.join(positive_img_dir, file), move_file_name)
        elif mode == 'delete':
            # Delete this file! Cautious with this
            print('!!! Deleting image here!!! Image {} does not have a label with it, deleting it'.format(file))
            os.remove(os.path.join(positive_img_dir, file))


def copy_some_image_randomly(img_dir, dest_dir, random_ratio=0.1):
    """
    This function copies some images randomly from a folder along with the labels.
    :param img_dir: the source dir to move the images from
    :param dest_dir: the destination dir to move the images to
    :param radom_ratio: The random ratio to move those images
    """
    # Create the destination folder
    if not os.path.isdir(dest_dir):
        os.makedirs(dest_dir)
    # Loop over the imgs
    for img in os.listdir(img_dir):
        if not img.endswith('.jpg') and not img.endswith('.JPG'):
            continue
        lbl_file = os.path.join(img_dir, img.replace('.jpg','.png').replace('.JPG', '.png'))
        img_file = os.path.join(img_dir, img)
        lbl_dest = os.path.join(dest_dir, img.replace('.jpg','.png').replace('.JPG', '.png'))
        img_dest = os.path.join(dest_dir, img)
        if not os.path.isfile(lbl_file):
            print('Your img file does not have the label : ', file)
            quit()
        # Use uniform distribution to get the randomness
        if np.random.uniform(0,1,1) < random_ratio:
            shutil.copyfile(lbl_file, lbl_dest)
            shutil.copyfile(img_file, img_dest)
        
            

if __name__ == '__main__':
    #check_and_delete_the_unlabelled_ones('positive_class')
    #generate_label_negative_folder('negative_class')
    copy_some_image_randomly(img_dir='negative_class', dest_dir='subsampled_negative_10percent')

