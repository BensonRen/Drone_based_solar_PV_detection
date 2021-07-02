# This folder gets the images of the RTI top 530 data that contains a solar panel into a sub folder
# Note that the process of going through these photos and determine whether they contain a solar panel is done by human (Ben lol)

import os
import numpy as np
import matplotlib.pyplot as plt
from shutil import copyfile, move


str_photos_indexs='42，43，80，82，120,121,162-167,193-198,200-207,217-223,238-241,325-326,332-334,\
                        361-364,375-379,399-404,419-425,439-443,447，478,453-456,481-482,491-495,519-526'

def parse_str_to_list(str_photo):
    """
    This parse the string list of photo, which compose of all strange stuff like /n or /t, all type of comma into a sorted list
    :param str_photo: The large string of photo list
    """
    output = []
    # Strip the illegal ones
    str_photo = str_photo.replace('/n',',')
    str_photo = str_photo.replace('，',',')
    # Parse it into a list
    parsed_list = str_photo.split(',')
    for s in parsed_list:
        if '-' in s:
            output.extend(get_range_to_list(s))
            continue
        output.append(int(s))
    output = sorted(output)
    return output


def get_range_to_list(range_str):
    """
    Takes a range string (e.g. 123-125) and return the list
    """
    start = int(range_str.split('-')[0])
    end = int(range_str.split('-')[1])
    if start > end:
        print("Your range string is wrong, the start is larger than the end!", range_str)
    return range(start, end+1)


def move_solar_panel_containing_img_out(img_dir, dest_dir='positive_class',mode='move'):
    """
    This function moves the images that are in the list that contains the solar panels out
    basically separating the ones containing solar panel and those that does not contain
    """
    # make sure dest exist
    if not os.path.isdir(dest_dir):
        os.makedirs(dest_dir)
    # Get the list to move
    move_list = parse_str_to_list(str_photos_indexs)
    for img_index in move_list:
        img_name = 'DJI_{}.JPG'.format(str(img_index).zfill(4))
        if mode == 'move':
            move(os.path.join(img_dir, img_name), os.path.join(dest_dir, img_name))
        elif mode == 'copy':
            copyfile(os.path.join(img_dir, img_name), os.path.join(dest_dir, img_name))
        else:
            print("Your mode of operation is neither copy nor move! aborting!")
            quit()

            




if __name__ == '__main__':
    #output=parse_str_to_list(str_photos_indexs)
    #print(len(output))
    #print(get_range_to_list('128-130'))
    move_solar_panel_containing_img_out('raw_images') 
    

