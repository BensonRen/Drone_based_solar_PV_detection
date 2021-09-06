# Ben 2021.02.27 
# Function to make a list of FULL PATH of files of JPGs

import os

#data_dir = '/scratch/sr365/Catalyst_data/2021_02_24_13_C_90'
#data_dir = '/scratch/sr365/Catalyst_data/2021_02_17_10_B_90'
#data_dir = '/scratch/sr365/RTI_data'

def make_file_list(data_dir, postfix='jpg', must_have=None, must_not_have=None, pairs=False):
    """
    Generate a file list of FULL PATH of files of JPGs
    :param: must_have: The file must have this this string to be made into the list
    :param: pairs: If true, output pairs of image name and label name (If patches, default true)
    """
    # If "patches" in the data_dir name, then default we are using pairs
    if 'patches' in data_dir:
        pairs = True

    save_file = os.path.join(data_dir, 'file_list_raw.txt')
    # Clear the previous file
    if os.path.isfile(save_file):
        os.remove(save_file)
    if must_have == 'BW' and pairs:
        save_file = save_file.replace('raw', 'valid')
    elif must_not_have == 'BW' and pairs:
        save_file = save_file.replace('raw', 'train')

    with open(save_file, 'a') as f:
        for files in os.listdir(data_dir):
            if must_have is not None and must_have not in files:
                print('{} does not have must_have component {}, skipping'.format(files, must_have))
                continue
            if must_not_have is not None and must_not_have in files:
                print('{} does not have must_not_have component {}, skipping'.format(files, must_have))
                continue
            if postfix is 'jpg':
                if files.endswith('.JPG') or files.endswith('.jpg'):
                    if pairs:
                        f.write(files)
                        f.write(' ')
                        f.write(files.replace('.jpg', '.png'))
                    else:
                        f.write(os.path.join(data_dir, files))
                    f.write('\n')
            elif postfix is 'png':
                if files.endswith('.PNG') or files.endswith('.png'):
                    f.write(os.path.join(data_dir, files))
                    f.write('\n')

def group_make_file_list(dir_group, postfix='jpg', must_have=None, must_not_have=None, pairs=False):
    """
    Make file list for a group of folders
    """
    for data_dir in dir_group:
        print('Making file list in {}'.format(data_dir))
        make_file_list(data_dir, postfix=postfix, must_have=must_have, must_not_have=must_not_have, pairs=pairs)

if __name__ == '__main__':
    # data_dir = '/scratch/sr365/RTI_data'
    # Make file list for individual folder
    #make_file_list(data_dir)
    
    # Make file list for a group of folders
    dir_group = ['/scratch/sr365/Catalyst_data/every_20m/d{}/images'.format(i) for i in range(1, 5)]

    
    # The cross validation
    #dir_group = []
    # for i in range(5, 13):
    #     for j in range(5, 13):
    #for i in range(1, 5):
    #    for j in range(1, 5):
    #         if i == j:
    #            continue
    #        dir_group.append('/scratch/sr365/Catalyst_data/every_20m/d{}_change_res_to_d{}/images'.format(i,j ))
            #dir_group.append('/scratch/sr365/Catalyst_data/every_10m_change_res/{}0m_resample_to_{}0m/images'.format(i, j))
    
    ##########################
    # Gaia specific          #
    # Exp 1.1+ expand dx     #
    ##########################
    #dir_group = []
    #height_list = [210, 420, 840] 
    #for height in height_list:
    #    dir_group.append('/scratch/sr365/Catalyst_data/simulated_satellite_height_{}/images'.format(height))
        
    ##########################
    # Gaia specific          #
    # Exp 1.2 expand dx      #
    ##########################
    #dir_group = []
    #for i in range(1, 5):
    #    for j in range(1, 5):
    #        # Skip the same ones
    #        if i == j:
    #            continue
    #        dir_group.append('/scratch/sr365/Catalyst_data/every_20m_change_res/d{}_change_res_to_d{}/images'.format(i, j))

    #for height in height_list:
    #    dir_group.append('/scratch/sr365/Catalyst_data/simulated_satellite_height_{}/patches'.format(height))

    ###################
    # Gaia specific   #
    # Exp 2 dx speed  #
    ###################
    # for i in range(1, 5):      # Model index
    #     for mode in ['Normal','Sports']:  # Test index
    #         dir_group.append('/scratch/sr365/Catalyst_data/test_moving_imgs/{}/d{}/images'.format(mode, i))


    #####################################
    # Gaia specific                     #
    # Exp 2 artificial motion blur speed#
    #####################################
    #dir_group = []
    #motion_blur_dir = '/scratch/sr365/Catalyst_data/every_20m/motion_blur'
    #for folder in os.listdir(motion_blur_dir):
    #    # make sure this is a folder and have mb in it
    #    cur_folder = os.path.join(motion_blur_dir, folder)
     #   if not 'mb' in folder or not os.path.isdir(cur_folder):
     #       continue
     #   # Add the folder in
     #   dir_group.append(os.path.join(cur_folder, 'images'))
    
    # Test set
    group_make_file_list(dir_group, must_have='BW', must_not_have=None, postfix='jpg')
    # Train set
    # group_make_file_list(dir_group, must_have=None, must_not_have='BW')
