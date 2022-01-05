# This function is for plotting the sample images of the post-processing chart
import os
from skimage import io
import numpy as np
import pandas as pd
from skimage import io, measure
from mrs_utils.eval_utils import score
from sklearn import metrics

# RTI model
# conf_map_folder = '/scratch/sr365/Catalyst_data/example_plot_for_flow_chart/save_root/ecresnet50_dcdlinknet_dsrwanda_rti_lre1e-03_lrd1e-02_ep80_bs16_ds50_75_dr0p1_crxent1p0_softiou0p5'

# H3 model
conf_map_folder ='/scratch/sr365/Catalyst_data/example_plot_for_flow_chart/save_root/ecresnet50_dcdlinknet_dscatalyst_h3_lre1e-03_lrd1e-02_ep80_bs16_ds50_100_dr0p1_crxent1p0_softiou0p5'

for conf_map in os.listdir(conf_map_folder):
    cur_conf = os.path.join(conf_map_folder, conf_map)
    conf = io.imread(cur_conf)
    print(np.shape(conf))
    
    conf_tile, true_tile = score(conf/255, conf, min_region=100, min_th=0.8, 
              dilation_size=5, link_r=5, iou_th=0.5, 
              tile_name=os.path.join('post_process_understanding', conf_map))