# This scipt serves as an ensemlber of the confidence map that is produced by the inference of the models
import os
import numpy as np
from skimage import io
from multiprocessing import Pool

def ensemble_maps(input_folder_list, output_dir):
    """
    The function that produces the ensembled map from a list of folders that contains the same confidence maps
    :param: input_folder_list: The list of input folders that contains the ensembled map
    :param: output_dir: The output directory to which we output the ensembled confidence map
    :param: model_index: The model index of d{}
    """
    # make sure the destination folder exist
    if not os.path.isdir(output_dir):
        os.makedirs(output_dir)

    print('the ensembled list of folder is {}'.format(input_folder_list))
    # Loop over all the images in the folder
    for conf_map_name in os.listdir(input_folder_list[0]):
        sum_conf_map = None
        # Loop over the input folder list
        for input_folder in input_folder_list:
            cur_conf_map_name = os.path.join(input_folder, conf_map_name)
            cur_conf_map = io.imread(cur_conf_map_name).astype('float')
            # Sum the conf maps
            if sum_conf_map is None:
                sum_conf_map = np.copy(cur_conf_map)
            else:
                sum_conf_map += np.copy(cur_conf_map)
                #print(type(sum_conf_map))
                #print(sum_conf_map.dtype)
                #print('cur conf map', np.sum(cur_conf_map))
                #print('mean of sum_map', np.mean(sum_conf_map))
                #print('shape of sum_map', np.shape(sum_conf_map))
                #print('adding to original sum_conf_map')
        #print(np.shape(sum_conf_map))
        #print(np.max(sum_conf_map))
        # Get the average
        #sum_conf_map = sum_conf_map / float(len(input_folder_list))
        sum_conf_map = sum_conf_map.astype(np.uint8)
        # Output to the output dir
        io.imsave(os.path.join(output_dir, conf_map_name), sum_conf_map)

def ensemble_cross_domain_matrix_prediction():
    """
    The caller function for the ensemble prediction function that structures the 
    """
    mother_dir = '/scratch/sr365/Catalyst_data/'
    prediction_folder_prefix = 'train_domain_trail_'
    num_cpu = 64
    try: 
        pool = Pool(num_cpu)
        args_list = []
        # Get from d1 to d4 images
        for i in range(1, 5):
            # Loop over the model list from d1 to d4
            for j in range(1, 5):
                print('working on d{} imagery, model {}'.format(i, j))
                model_name = 'ecresnet50_dcdlinknet_dscatalyst_d{}_lre1e-03_lrd1e-02_ep80_bs16_ds50_75_dr0p1_crxent1p0_softiou0p5'.format(j)
                output_dir = os.path.join(mother_dir, 'd{}/images/'.format(i), 'train_domain_ensembled_img_d{}_model_d{}'.format(i, j))
                folder_list = []
                # Loop over the number of trails
                for trail in range(5):
                    # Add the folder name to the folder list
                    folder = os.path.join(mother_dir, 'd{}/images/'.format(i), prediction_folder_prefix + '{}'.format(trail), model_name)
                    folder_list.append(folder)
                args_list.append((folder_list, output_dir))
            #ensemble_maps(folder_list, output_dir)
        pool.starmap(ensemble_maps, args_list)
    finally:
        pool.close()
        pool.join()


    
if __name__ == '__main__':
    ensemble_cross_domain_matrix_prediction()