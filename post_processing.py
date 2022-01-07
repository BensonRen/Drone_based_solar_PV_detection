from numpy.core.defchararray import array, mod
from numpy.core.numeric import _moveaxis_dispatcher
from skimage import io
import os
import numpy as np
import pandas as pd
import imagesize
from mrs_utils.eval_utils import ObjectScorer, dummyfy, get_stats_from_group

def post_processing(mother_folder, min_region=10, min_th=0.5, 
                    dilation_size=5, link_r=0, eps=2, 
                    operating_object_confidence_thres=0.9):
    """
    From confidence map, do the post-processing just like the evaluation does
    :param: operating_object_confidence_thres: The operating confidence threshold that decide where you are 
        on the PR curve during acutal operation
    """
    # Loop over all the prediction maps
    for conf_map_name in os.listdir(mother_folder):
        # Skip and only work on the conf_map
        if 'conf.png' not in conf_map_name:
            continue
        print('post processing {}'.format(conf_map_name))
        # Read this conf map
        conf_map = io.imread(os.path.join(mother_folder, conf_map_name))
        # Rescale to 0, 1 interval
        conf_map  = conf_map / 255
        # Create the object scorer to do the post processing
        obj_scorer = ObjectScorer(min_region, min_th, dilation_size, link_r, eps)
        # Get the object groups after post processing
        group_conf = obj_scorer.get_object_groups(conf_map)
        # If no groups are around, continue
        if len(group_conf) == 0:
            continue
        # Loop over each individual groups
        for g_pred in group_conf:
            # Assigning flags for whether to keep this group
            g_pred.keep = True
            # Calculate the average confidence value (to be thresholded)
            _, conf = get_stats_from_group(g_pred, conf_map)
            # IF this object is smaller than what it should be
            if conf < operating_object_confidence_thres:
                # Throw this group away
                g_pred.keep = False
        # Threshold the objects by their confidence values
        group_conf = [a for a in group_conf if a.keep == True]
        # dummyfy the result into a binary plot
        conf_dummy = dummyfy(conf_map, group_conf)
        io.imsave(os.path.join(mother_folder, conf_map_name.replace('conf', 'conf_post_processed')), conf_dummy)

if __name__ == '__main__':
    confidence_folder = r'data_raw/Exp1_1_resolution_buckets/test/d1/images/save_root/model1'
    post_processing(mother_folder=confidence_folder)