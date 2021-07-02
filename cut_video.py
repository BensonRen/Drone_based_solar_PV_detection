# This script / function calls ffmpeg in the linux system to cut frames

import os
import sys
import numpy as np
import shutil

save_img_big_dir = '/scratch/sr365/Catalyst_data/video_cut'
video_big_dir = '/scratch/sr365/Catalyst_data'
post_fix = '.mp4'

# Function that gets the list of videos
def get_video_list(video_big_dir, post_fix):
    """
    This script get in all the folders in the input folder and get the list of FULL PATH of all the videos
    :param: video_big_dir: The directory to inquire one folder by another to look for videos
    """
    video_list = []
    for folder in os.listdir(video_big_dir):
        sub_dir = os.path.join(video_big_dir, folder)
        # Only go through folder
        if not os.path.isdir(sub_dir):
            continue
        for file in os.listdir(sub_dir):
            current_file = os.path.join(sub_dir, file)
            # Only go through the videos
            if not current_file.endswith(post_fix):
                continue
            video_list.append(current_file)
    return video_list




def cut_video_to_dest(video_list, save_img_big_dir, video_big_dir):
    """
    The function to save the cut images from the video list to save_img_big_dir
    """
    for video in video_list:
        # Get the video name
        video_name = video.split(video_big_dir)[-1].split(post_fix)[0]
        
        # Strip the leading '/'
        if video_name.startswith('/'):
            video_name = video_name[1:]
        video_name = video_name.replace('/','_')
        print('cutting :', video_name)

        # Create the save_dir if not exist
        save_dir = os.path.join(save_img_big_dir,video_name)

        # ONLY ffmpeg if this folder does not exist, which means this video has not been cut before
        if not os.path.isdir(save_dir):
            # This means the video was never cut, make dir and cut here!
            os.makedirs(save_dir)
        else:
            # This means the video has been cut, ignore and continue!!
            continue
        
        # prepare the ffmpeg command and execute!
        command = 'ffmpeg -i {} {}/%04d.png -r 24 -hide_banner'.format(video, save_dir )
        os.system(command)
        
#  ffmpeg -i ../2021_03_10_10_D_90/DJI_0009__height_50m_N.mp4 DJI_0009_height_50m_N%04d.jpg -hide_banner


def label_imgs_with_folder_name(mother_dir):
    """
    This function labels the images cut from this function with the folder name concatenated in front
    example: /video_cut/2021_03_10_10_D_90_DJI_0052_DJI_0051_height_100m_S/001.jpg
    change to 2021_03_10_10_D_90_DJI_0052_DJI_0051_height_100m_S_001.jpg in the same folder
    """
    for folders in os.listdir(mother_dir):
        cur_folders = os.path.join(mother_dir, folders)
        if not os.path.isdir(cur_folders):
            continue
        for img in os.listdir(cur_folders):
            cur_img = os.path.join(cur_folders, img)
            new_name = os.path.join(cur_folders, os.path.basename(cur_folders) + img)
            print('original name {}, new name {}'.format(cur_img, new_name))
            os.rename(cur_img, new_name)


def sample_from_video_cuts(mother_dir, save_dir, exclude_pre=0.1, exclude_post=0.2, sample_num=5):
    """
    This function samples a subset of the video cuts to form a dataset 
    :param moether_dir: The source dir with all the video cuts inside, each one is a folder with all the images inside
    :param exclude_pre/post: The portion of images to exclude in front / end
    :param sample_num: The number of samples drawn from each of the videos
    :param save_dir: The directory to save the video
    """
    total_samples_got = 0
    for folders in os.listdir(mother_dir):
        cur_folders = os.path.join(mother_dir, folders)
        # check if this is a folder
        if not os.path.isdir(cur_folders):
            continue
        img_list = os.listdir(cur_folders)      
        img_list.sort()                     # Get the sorted list of file names
        
        num_img = len(img_list)
        # Get the sampled indexs
        sample_index_list = np.random.permutation(int(num_img*(exclude_post+exclude_pre)))[:sample_num] + int(num_img*exclude_pre)
        #print('pre lim {}, {}, post lim {}'.format(int(num_img*exclude_pre), sample_index_list, int(num_img*(1-exclude_post))))

        # Copy the images into the big dir
        for sample_index in sample_index_list:
            shutil.copyfile(os.path.join(cur_folders, img_list[sample_index]), os.path.join(save_dir, img_list[sample_index]))
        
        # Testing purposes
        #quit()
    
    print('out of {} folders, we got {} samples and saving them in {}'.format(len(os.listdir(mother_dir)), total_samples_got, save_dir))


if __name__ == '__main__':
    # The first step of cutting them
    #video_list = get_video_list(video_big_dir, post_fix)
    #cut_video_to_dest(video_list, save_img_big_dir, video_big_dir)
    
    # The second step of relabelling them
    #label_imgs_with_folder_name(save_img_big_dir)

    # The third step of sampling a subset of them
    sample_from_video_cuts(save_img_big_dir, save_dir='/scratch/sr365/Catalyst_data/moving_imgs')
        

