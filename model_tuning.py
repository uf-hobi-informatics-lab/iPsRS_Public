import os
import sys
import csv
import json
import shap
import warnings
import argparse
import numpy as np
import pandas as pd
import tqdm as tqdm
import matplotlib.pyplot as plt

from types import SimpleNamespace

from sklearn.metrics import confusion_matrix, roc_auc_score ,roc_curve,auc
from sklearn.metrics import f1_score, roc_auc_score, recall_score, precision_score
from sklearn.model_selection import GridSearchCV,StratifiedKFold, train_test_split, KFold, RandomizedSearchCV

from config.model import model_construction
from util.io import check_saving_path, save_model, load_model

from joblib import Parallel, parallel_backend
from joblib import register_parallel_backend
from joblib import delayed
from joblib import cpu_count
from ipyparallel import Client
from ipyparallel.joblib import IPythonParallelBackend



def execute(cfg,pfe):
    
    
    if cfg.parallel:
        FILE_DIR = os.path.dirname(os.path.abspath(__file__))
        sys.path.append(FILE_DIR)
        
        c = Client(profile=pfe)
        print("number of jobs = ",len(c))
        
        #c[:].map(os.chdir, [FILE_DIR]*len(c))
        
        bview = c.load_balanced_view()
        register_parallel_backend('ipyparallel', lambda : IPythonParallelBackend(view=bview))
    
    
    ''' Load settings '''
    model_name = cfg.model_name
    model_id = cfg.model_id
    model_alg = cfg.model_alg
    
    root_dir = cfg.root_dir
    models_dir = cfg.models_dir
    processed_dir = cfg.processed_dir
    output_roc_dir = cfg.output_roc_dir
    output_shap_dir = cfg.output_shap_dir
    output_score_dir = cfg.output_score_dir
    output_cv_dir = cfg.output_cv_dir
    
    ## full model
    raw_X = pd.read_csv(os.path.join(root_dir, processed_dir, cfg.input_features), index_col=0)
    used_variables = raw_X.columns
    
    X = raw_X.to_numpy()
    Y = pd.read_csv(os.path.join(root_dir, processed_dir, cfg.input_labels), index_col=0).to_numpy().flatten()
   

    #Generate Training and Testing Set
    X_train, X_test, y_train, y_test = train_test_split(X, Y, stratify=Y, test_size=cfg.setting_params.train_test_ratio, random_state=cfg.setting_params.random_state) 
    #Generate Training and Evaluation Set
    X_train, X_val, y_train, y_val = train_test_split(X_train, y_train, stratify=y_train, test_size=cfg.setting_params.train_val_ratio, random_state=cfg.setting_params.random_state) #0.125 * 0.8 = 0.1

    # check the config file /fairness/config/model.py to setup more experimental models
    if cfg.parallel:
        clf, _, fit_params = model_construction(cfg, X_val=X_val, y_val=y_val, n_jobs = len(c))
        with parallel_backend('ipyparallel'): 
            all_model = clf.fit(X_train, y_train, **fit_params)
    else:
        clf, _, fit_params = model_construction(cfg, X_val=X_val, y_val=y_val, n_jobs = cfg.setting_params.n_jobs)
        all_model = clf.fit(X_train, y_train, **fit_params)
        

    # save the best model
    save_model(all_model, root_dir, models_dir, model_name, model_alg+model_id )

    save_dir = os.path.join(cfg.root_dir, cfg.output_cv_dir, model_name.split("_")[0], model_name + model_alg + model_id)
    save_path = check_saving_path(save_dir, cfg.output_cv_filename) 
    df = pd.DataFrame(all_model.cv_results_)
    df.to_csv(save_path)

if __name__ == "__main__":

    parser = argparse.ArgumentParser()                                               

    parser.add_argument("--setting", "-s", type=str, required=True)
    
    parser.add_argument("-p", "--profile", default="ipy_profile", help="Name of IPython profile to use")
    
    args = parser.parse_args()
    
    with open(args.setting) as json_file:
        cfg = json.load(json_file, object_hook=lambda d: SimpleNamespace(**d))
    
    execute(cfg, args.profile)