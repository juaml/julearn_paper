#%%
import time
import math
import pickle
import argparse
import numpy as np
import pandas as pd
from pathlib import Path
import matplotlib.pyplot as plt
import seaborn as sns

from skrvm import RVR
import sklearn.gaussian_process as gp

from sklearn.model_selection import RepeatedStratifiedKFold
from julearn.model_selection import StratifiedGroupsKFold
from sklearn.metrics import mean_absolute_error, mean_squared_error

from julearn import run_cross_validation
from julearn.utils import configure_logging
from julearn.model_selection import StratifiedGroupsKFold


def performance_metric(y_true, y_pred):
    mae = round(mean_absolute_error(y_true, y_pred), 3)
    mse = round(mean_squared_error(y_true, y_pred), 3)
    corr = round(np.corrcoef(y_pred, y_true)[1, 0], 3)
    return mae, mse, corr

#%%
start_time = time.time()

if __name__ == '__main__':

    # Set the logging level to info to see extra information
    # configure_logging(level='INFO')
    
#    parser = argparse.ArgumentParser()
#    parser.add_argument("--input_file", type=str, help="Input file path")
#    parser.add_argument("--output_path", type=str, help="Path to output directory")
#    parser.add_argument("--pca_status", type=int, default=0,
#                       help="0: no pca, 1: yes pca")
#
#
#    # Parse the arguments
#    args = parser.parse_args()
#    input_file = args.input_file
#    output_path = Path(args.output_path)
#    pca_status = bool(args.pca_status)
#    output_path.mkdir(exist_ok=True, parents=True) # check and create output directory
    
    input_file = '../data/ixi.S4_R8.csv'
    output_path = '../results'
    pca_status = bool(1)

    output_prefix = input_file.split('/')[-1]
    output_prefix = output_prefix.split('.')[1]
    output_path = Path(output_path)

    # initialize random seed and define number of splits and repeats for CV
    rand_seed = 200
    num_repeats = 1 # for inner CV
    num_splits = 3  # how many train and test splits (both for other and inner)

    print('\nInput file: ', input_file)
    print('Ouput path : ', output_path)
    print('PCA status : ', pca_status)
    print('Random seed : ', rand_seed)
    print('Num of splits for kfolds : ', num_splits, '\n')

    # read the features, demographics and define X and y
    data_df = pd.read_csv(input_file)
    X = [col for col in data_df if col.startswith('f_')]
    y = 'age'

  
    scores_cv, models, results = {}, {}, {}
    var_threshold  = 1e-5

    # Define number of splits for CV and create bins/group for stratification
    num_bins = math.floor(len(data_df) / num_splits)  # num of bins to be created
    bins_on = data_df[y]  # variable to be used for stratification
    qc = pd.cut(bins_on.tolist(), num_bins)  # divides data in bins
    data_df['bins'] = qc.codes
    groups = 'bins'
   
    # Define all models and model parameters
    rvr = RVR() # not available in julearn
    preprocess_X = ['select_variance', 'zscore', 'pca']
    print('Preprocessing includes:', preprocess_X)

    
    model_list = {
    'rvr': [rvr, {'select_variance__threshold': var_threshold,
    'pca__n_components': None, 'rvr__kernel': ['linear', 'poly'],
    'rvr__degree': [1, 2], 'rvr__random_state': rand_seed}],

    'gpr': ['gauss', {'select_variance__threshold': var_threshold,
    'pca__n_components': None, 'gauss__kernel': gp.kernels.RBF(10.0, (1e-7, 10e7)),
    'gauss__n_restarts_optimizer': 100, 'gauss__normalize_y': True,
    'gauss__random_state': rand_seed}]
    }

    scores_cv =  pd.DataFrame()
    for key, value in model_list.items():  # run only for required models and not all
        print(key)
        # initialize dictionaries to save scores and models here to save every model separately
        

        cv = RepeatedStratifiedKFold(n_splits=num_splits, n_repeats=num_repeats, random_state=rand_seed).split(data_df, data_df.bins)
    
        scores = run_cross_validation(X=X, y=y, data=data_df, preprocess_X=preprocess_X,
                                             problem_type='regression', model=value[0], cv=cv,
                                             return_estimator='cv', model_params=value[1], seed=rand_seed,
                                             scoring=['neg_mean_absolute_error', 'neg_mean_squared_error', 'r2'])

        scores['model'] = key
        scores_cv = scores_cv.append(scores)
        
# plot CV results
    for scoring_item in ['test_neg_mean_absolute_error', 'test_neg_mean_squared_error', 'test_r2']:
        if scoring_item == 'test_neg_mean_absolute_error' or 'test_neg_mean_squared_error':
            scores_cv[scoring_item] = scores_cv[scoring_item] * -1
            fig, ax = plt.subplots(1, 1, figsize=(10, 7))
            sns.set_style("darkgrid")
            ax = sns.boxplot(x='model', y=scoring_item, data=scores_cv)
            ax = sns.swarmplot(x="model", y=scoring_item, data=scores_cv, color=".25")
            print(output_path + '_' + scoring_item + '.svg')
            plt.savefig(output_path + '_' + scoring_item + '.svg', dpi=500)




