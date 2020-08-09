import os
os.environ['KMP_WARNINGS'] = '0'

import glob
import numpy as np

import pandas as pd
from utils import *

# Enter the experiment dataset name
dataset_name = 'CYP'
# Models to plot weight distribution and important features
plot_features_models = []
# Number of important features user want to draw
nodes = 1024

output_all_y = False

if __name__ == "__main__": 
    result_dir = 'outputs'
    if not os.path.exists(result_dir):
        os.mkdir(result_dir)
    datasets = glob.glob('data/%s*' % dataset_name)
    print (datasets)
    for input_file in datasets:
        experiment_dataset = input_file.split('/')[-1].split('.')[0]
        if os.path.exists('%s/%s_y_mc.csv' %  (result_dir, experiment_dataset)):
            print ('%s is done' % experiment_dataset)
            #continue
        file_df = pd.read_csv(input_file, header = None)
        file_df, data_set = get_data(experiment_dataset , file_df)
        print ('########## Start trinaing %s ##########, processed data: %d' % (experiment_dataset , len(file_df.index)))
        ys = pd.DataFrame()
        train_X, test_X, valid_X, train_y, test_y, valid_y = split_var(data_set)
        ys['true label'] = test_y

        for dr in [0, 0.2, 0.4, 0.6]:
            model_type = 'Dropout rate =' , dr
            model, hist, AUC, ACC, pred_y  = MC_NN(train_X, test_X, valid_X, train_y, test_y, valid_y, 'relu', nodes, dr, 300)			
            pred_ys = np.array(pred_y[0])
            ys['%s_mean'% dr] = pred_y[1]
            ys['%s_std' % dr] = pred_y[2]
            df = pd.DataFrame(pred_ys[:,:,0])
            if output_all_y == True:
                df.to_csv('%s/%s_%s_predicted_ys.csv' % (result_dir, dr, experiment_dataset), index = False)
        ys.to_csv('%s/%s_y_mc.csv' % (result_dir, experiment_dataset), index = False)



