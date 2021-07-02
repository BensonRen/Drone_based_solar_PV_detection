# This script is to plot the histogram of height 

import numpy as np
import os
from skimage import io
import matplotlib.pyplot as plt

folder = 'data'
folder_list = ['images','annotations']
# h1 = < 45m
# 45m < h2 < 85m
# h3 >= 85m

#height_list = []

for fol in folder_list:
    img_folder = os.path.join(folder, fol)
    for img in os.listdir(img_folder):
        #print(img)
        # get the height
        height = int(eval(img.split('.')[0].split('_')[-1][:-1]))
        #print("height is", height)
        
        # Get them in the list
        #height_list.append(height)
        
        if height < 45:
            os.rename(os.path.join(img_folder, img), os.path.join('h1', fol, img))
        elif height < 85:
            os.rename(os.path.join(img_folder, img), os.path.join('h2', fol, img))
        else:
            os.rename(os.path.join(img_folder, img), os.path.join('h3', fol, img))

    
    
# Plotting a histogram
#f = plt.figure()
#plt.hist(height_list)
#plt.savefig('hist of height.png')

# 
