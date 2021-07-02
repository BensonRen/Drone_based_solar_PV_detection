# This script is to hyper sweep the experiment config files
from train import *
import numpy as np

# The catalyst dataset with d claasification (d1=50, 60m, d2=70,80m, d3=90,100m, d4=110,120m)
base_json_config = 'config_0529_d1.json'


# The RTI positive data
#base_json_config = 'config_ben_0509_RTI_sample_positive_from_h3.json'
#base_json_config = 'config_ben_0509_RTI_sample_positive_from_ct.json'



def sweep_strength():
    #loss_weight_list = [0.1, 0.5, 1, 1.5, 2]
    #strength_list = [1, 10]
    #strength_list = [100]
    #strength_list = [75, 30]
    #strength_list = [1, 10, 50, 100]
    #strength_list = [300]
    #for mb in range(2, 6):
    #for class_strength in strength_list:
    for i in range(2, 3):
#for loss_weight in loss_weight_list:
        # Setting up the cfg file
        cfg_dict = {}
        cfg_dict['config'] = base_json_config
        flags = json.load(open(base_json_config))
        flags = misc_utils.update_flags(flags, cfg_dict)
        
        ##########################################################################################################################################################
        # Change the cfg file for current sweep
        #flags['trainer']['save_root'] += 'loss_weight={}'.format(loss_weight)
        #flags['trainer']['loss_weights'] = '(1, ' + str(loss_weight) + ')'
        
        class_strength = eval(flags['trainer']['class_weight'])[1]
        flags['trainer']['save_root'] += '/class_weight_{}_trail_{}'.format(class_strength, i)
        #flags['trainer']['class_weight'] = '(1, ' + str(class_strength) + ')'
        
        # Change for the motion blur hypersweep
        #name_before = 'h3'
        #name_after = 'h3_mb_{}'.format(mb)
        #flags['dataset']['data_dir'] = flags['dataset']['data_dir'].replace(name_before, name_after)
        #flags['dataset']['train_file'] = flags['dataset']['train_file'].replace(name_before, name_after)
        #flags['dataset']['valid_file'] = flags['dataset']['valid_file'].replace(name_before, name_after)
        #flags['trainer']['save_root'] += name_after
        ##########################################################################################################################################################

        # Continue with the ordinary training procedure
        flags['save_dir'] = os.path.join(flags['trainer']['save_root'], network_utils.unique_model_name(flags))
        
        # set gpu to use
        device, parallel = misc_utils.set_gpu(flags['gpu'])
        # set random seed
        #!!!!!!!!!!!!!!!!!!!!!! Changing the random seed here !!!!!!!!!!!!!!!!!!!!!!!!!!
        misc_utils.set_random_seed(np.random.randint(20000))

        #misc_utils.set_random_seed(flags['random_seed'])
        # make training directory
        misc_utils.make_dir_if_not_exist(flags['save_dir'])
        misc_utils.save_file(os.path.join(flags['save_dir'], 'config.json'), flags)
        # train the model
        train_model(flags, device, parallel)
            


if __name__ == '__main__':
    sweep_strength()

