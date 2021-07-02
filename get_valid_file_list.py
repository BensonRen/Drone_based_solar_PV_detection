# The function that gets validation file list 
import numpy as np
import pandas as pd
import os

def get_valid_file_list(folder):
    # Set some folder name and prefixs
    valid_file = pd.read_csv(os.path.join(folder, 'file_list_valid.txt'), sep=' ', header=None).values
    #print(np.shape(valid_file))

    #print(valid_file[0,:])
    # Loop over all the pairs of imager and labels
    for i in range(len(valid_file)):
        valid_file[i, 0] = ''.join([os.path.join(folder, 'images/'), valid_file[i, 0].split('_y')[0], '.JPG'])
    
    # get file list
    file_list = np.unique(valid_file[:, 0])
    print(list(file_list))
    
    # Save the file list
    with open(os.path.join(folder, 'images', 'file_list_raw.txt'), 'w') as output:
        for image_name in list(file_list):
            output.write(image_name)
            output.write('\n')


if __name__ == '__main__':

    #folder = '/scratch/sr365/Catalyst_data/d1'
    folder_list = ['/scratch/sr365/Catalyst_data/d1',
                    '/scratch/sr365/Catalyst_data/d2',
                    '/scratch/sr365/Catalyst_data/d3',
                    '/scratch/sr365/Catalyst_data/d4']
    for folders in folder_list:
        get_valid_file_list(folders)