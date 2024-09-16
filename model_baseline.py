import os
import sys
import csv
import json
import warnings
import argparse
import numpy as np
import pandas as pd

from sklearn.model_selection import train_test_split
from types import SimpleNamespace
from util.measure import performance_score
from sklearn.utils import resample
from config.baseline import model_finalize
from util.io import check_saving_path, save_model, load_model, save_params_as_json

def execute(cfg):
    
    ''' Load settings '''
    model_name = cfg.model_name
    model_alg = cfg.model_alg
    model_id = cfg.model_id
    root_dir = cfg.root_dir
    models_dir = cfg.models_dir
    
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
    
    model, _, fit_params = model_finalize(cfg, X_val=X_val, y_val=y_val)
    model.fit(X_train, y_train, **fit_params)
    
    model_save_name = model_name + model_alg + model_id
    save_model(model, root_dir, models_dir, model_save_name,"baseline.pk")

    test_pred = model.predict(X_test)
    test_pred_prob = model.predict_proba(X_test)

    performance = performance_score(y_test, test_pred, test_pred_prob[:, 1])
        
    print("MODEL INFO: ", model_name + model_alg + model_id)
    print("MODEL Settings: ", model)
    print("MODEL Performance: ", performance)
    
if __name__ == "__main__":

    parser = argparse.ArgumentParser()                                               

    parser.add_argument("--setting", "-s", type=str, required=True)
    
    args = parser.parse_args()
    
    with open(args.setting) as json_file:
        cfg = json.load(json_file, object_hook=lambda d: SimpleNamespace(**d))
    
    execute(cfg)