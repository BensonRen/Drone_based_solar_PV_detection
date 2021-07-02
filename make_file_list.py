# Ben 2021.02.27 
# Function to make a list of FULL PATH of files of JPGs

import os

#data_dir = '/scratch/sr365/Catalyst_data/2021_02_24_13_C_90'
#data_dir = '/scratch/sr365/Catalyst_data/2021_02_17_10_B_90'
#data_dir = '/scratch/sr365/RTI_data'

def make_file_list(data_dir, postfix='jpg', must_have=None, must_not_have=None):
    """
    Generate a file list of FULL PATH of files of JPGs
    :param: must_have: The file must have this this string to be made into the list
    """
    save_file = os.path.join(data_dir, 'file_list_raw.txt')
    # Clear the previous file
    if os.path.isfile(save_file):
        os.remove(save_file)
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
                    print('writing to file_list_raw.txt')
                    f.write(os.path.join(data_dir, files))
                    f.write('\n')
            elif postfix is 'png':
                if files.endswith('.PNG') or files.endswith('.png'):
                    f.write(os.path.join(data_dir, files))
                    f.write('\n')

def group_make_file_list(dir_group, postfix='jpg', must_have=None, must_not_have=None):
    """
    Make file list for a group of folders
    """
    for data_dir in dir_group:
        print('Making file list in {}'.format(data_dir))
        make_file_list(data_dir, postfix=postfix, must_have=must_have, must_not_have=must_not_have)

if __name__ == '__main__':
    # data_dir = '/scratch/sr365/RTI_data'
    # Make file list for individual folder
    #make_file_list(data_dir)
    
    # Make file list for a group of folders
    #dir_group = ['/scratch/sr365/Catalyst_data/d{}/images'.format(i) for i in range(1, 5)]
    
    # The cross validation
    # dir_group = []
    # for i in range(5, 13):
    #     for j in range(5, 13):
    #         if i == j:
    #             continue
    #         dir_group.append('/scratch/sr365/Catalyst_data/every_10m_change_res/{}0m_resample_to_{}0m/images'.format(i, j))

    # Test set
    #group_make_file_list(dir_group, must_have='BW', must_not_have=None)
    # Train set
    #group_make_file_list(dir_group, must_have=None, must_not_have='BW')