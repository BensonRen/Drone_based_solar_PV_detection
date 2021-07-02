# The function to change the name of a list of folders
# 2021.06.07 Ben wants to change a list of folder names that is too long for plotting

import numpy as np
import os
import shutil


name_change_dir_list = ['/scratch/sr365/Catalyst_data/every_10m/{}0m/images/save_root'.format(i) for i in range(5, 13)]

def change_folder_name(name_change_dir_list):
    for name_change_dir in name_change_dir_list:
        for folders in os.listdir(name_change_dir):
            # Change the name 
            new_name = folders.split('catalyst')[-1].split('lr')[0].split('_')[1]+'_model'
            print('old name is {}, change to {}'.format(folders, new_name))
            os.rename(os.path.join(name_change_dir, folders), os.path.join(name_change_dir, new_name))

def append_name(mother_folder, name_starts_with='agg'):
    """
    This function appends the folder name to the start of the individual file names
    Typically this is for the 
    """
    for folder in os.listdir(mother_folder):
        cur_folder = os.path.join(mother_folder, folder)
        # Skip if this is not a folder
        if not os.path.isdir(cur_folder): 
            continue
        # For each subfolder, change the names of the files inside them
        for file in os.listdir(cur_folder):
            # If it does not start from NAME_STARTS_WITH, skip
            if not file.startswith(name_starts_with):
                continue
            os.rename(os.path.join(cur_folder, file), os.path.join(cur_folder, folder + file))


if __name__ == '__main__':
    append_name('/scratch/sr365/PR_curves/')