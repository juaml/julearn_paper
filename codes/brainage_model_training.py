import time
import math
import pickle
import argparse
import numpy as np
import pandas as pd
from pathlib import Path

from skrvm import RVR
from sklearn.model_selection import RepeatedStratifiedKFold
from sklearn.metrics import mean_absolute_error, mean_squared_error

from julearn import run_cross_validation
from julearn.utils import configure_logging


def performance_metric(y_true, y_pred):
    mae = round(mean_absolute_error(y_true, y_pred), 3)
    mse = round(mean_squared_error(y_true, y_pred), 3)
    corr = round(np.corrcoef(y_pred, y_true)[1, 0], 3)
    return mae, mse, corr


start_time = time.time()

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--input_file", type=str, help="Input file path")
    parser.add_argument("--output_path", type=str, help="Path to output directory")
    parser.add_argument("--pca_status", type=int, default=0,
                       help="0: no pca, 1: yes pca")
    configure_logging(level='INFO')

    # Parse the arguments
    args = parser.parse_args()
    input_file = args.input_file
    output_path = Path(args.output_path)
    model_required = [x.strip() for x in args.models.split(',')]  # converts string into list
    pca_status = bool(args.pca_status)
    output_path.mkdir(exist_ok=True, parents=True) # check and create output directory
    

    # initialize random seed and create test indices
    rand_seed = 200
    n_repeats = 5 # for inner CV
    num_splits = 5  # how many train and test splits (both for other and inner)

    print('\nInput file: ', input_file)
    print('Ouput path : ', output_path)
    print('PCA status : ', pca_status)
    print('Random seed : ', rand_seed)
    print('Num of splits for kfolds : ', num_splits, '\n')

    # read the features, demographics and define X and y
    data_df = pd.read_csv(input_file)
    X = [col for col in data_df if col.startswith('f_')]
    y = 'age'

    # Initialize variables, set random seed, create classes for age
    scores_cv, models, results = {}, {}, {}
    qc = pd.cut(data_df['age'].tolist(), bins=5, precision=1)  # create bins for train data only
    print('age_bins', qc.categories, 'age_codes', qc.codes)
    data_df['bins'] = qc.codes # add bin/classes as a column in train df


    # Define all models and model parameters
    rvr = RVR() # not available in julearn

    var_threshold  = 1e-5
    preprocess_X = ['select_variance', 'zscore', 'pca']
    print('Preprocessing includes:', preprocess_X)
    
    model_names = ['rvr', 'gpr']
    model_list = [rvr, 'gauss']
    model_para_list =
                    [
                    {'select_variance__threshold': var_threshold, 'pca__n_components': None, 'rvr__kernel': ['linear', 'poly'],'rvr__degree': [1, 2], 'rvr__random_state': rand_seed},
    
                    {'select_variance__threshold': var_threshold, 'pca__n_components': None, 'gauss__kernel': gp.kernels.RBF(10.0, (1e-7, 10e7)), 'gauss__n_restarts_optimizer': 100, 'gauss__normalize_y': True, 'gauss__random_state': rand_seed}
                    ]

    # Get the model, its parameters, pca status and train
    for ind in range(0, model_list):  # run only for required models and not all
        
        # initialize dictionaries to save scores and models here to save every model separately
        scores_cv, models = {}, {}

        cv = RepeatedStratifiedKFold(n_splits=n_splits, n_repeats=n_repeats, random_state=rand_seed).split(data_df, data_df.bins)

        scores, model = run_cross_validation(X=X, y=y, data=data_df, preprocess_X=preprocess_X
                                             problem_type='regression', model=model_list[i], cv=cv,
                                             return_estimator='all', model_params=model_para_list[i], seed=rand_seed,
                                             scoring=['neg_mean_absolute_error', 'neg_mean_squared_error', 'r2'])

        scores_cv[model_names[i]] = scores

        if model_names[i] == 'rvr':
            models[model_names[i]] = model.best_estimator_
            print('best model', model.best_estimator_)
            print('best para', model.best_params_)
        else:
            models[model_names[i]] = model
            print('best model', model)

        print('Output file name')
        print(output_path / f'{output_prefix}.{model_names[i]}.models')
        pickle.dump(models, open(output_path / f'{output_prefix}.{model_names[i]}.models', "wb"))
        pickle.dump(scores_cv, open(output_path / f'{output_prefix}.{model_names[i]}.scores', "wb"))

    print('ALL DONE')
    print("--- %s seconds ---" % (time.time() - start_time))
    print("--- %s minutes ---" % ((time.time() - start_time) / 60))
    print("--- %s hours ---" % ((time.time() - start_time) / 3600))












