import os 
import pandas as pd
import numpy as np

# Moving labels from the RTI_Rwanda imagery to the individual folders
# Checking for each label if there is a corresponding image
folder_dir = '/home/sr365/Gaia/Rwanda_RTI/rti_rwanda_cut_tiles_ps_8000/labels/'
img_folder = '/home/sr365/Gaia/Rwanda_RTI/rti_rwanda_cut_tiles_ps_8000'

for folder in os.listdir(img_folder):
    cur_folder = os.path.join(img_folder, folder)
    if not os.path.isdir(cur_folder) or 'label' in cur_folder:
        continue
    annotate_folder = os.path.join(cur_folder, 'annotations')
    img_fol = os.path.join(cur_folder, 'images')
    for file in os.listdir(cur_folder):
        if not file.endswith('.png'):
            continue
        cur_file = os.path.join(cur_folder, file)
        new_name = os.path.join(img_fol, file)
        os.rename(cur_file, new_name)
    # if not os.path.isdir(annotate_folder):
    #     os.makedirs(annotate_folder)
    # if not os.path.isdir(img_fol):
    #     os.makedirs(img_fol)

# for folder in os.listdir(folder_dir):
#     cur_folder = os.path.join(folder_dir, folder)
#     if not os.path.isdir(cur_folder):
#         continue
#     print('entering folder', cur_folder)
#     for subfolder in os.listdir(cur_folder):
#         cur_subfolder = os.path.join(cur_folder, subfolder)
#         if not os.path.isdir(cur_subfolder):
#             continue
#         print('entering subfolder', cur_subfolder)
#         for file in os.listdir(cur_subfolder):
#             if '.png' not in file:
#                 continue
#             cur_name = os.path.join(cur_subfolder, file)
#             # Check for the same name in another folder
#             new_name = os.path.join(img_folder, folder + '_' + subfolder, 'annotations', file)
#             print('renaming {} to {}'.format(cur_name, new_name ))
#             os.rename(cur_name, new_name)