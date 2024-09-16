import os
import sys
import csv
import json
import warnings
import argparse
import numpy as np
import pandas as pd

from types import SimpleNamespace

from config.model import model_finalize
from util.io import *

from util.measure import performance_score, fariness_score
from sklearn.model_selection import train_test_split
from types import SimpleNamespace

def execute(cfg, mparam, grp):
    
    ''' Load settings '''
    model_name = cfg.model_name
    model_alg = cfg.model_alg
    model_id = cfg.model_id
    root_dir = cfg.root_dir
    models_dir = cfg.models_dir
    processed_dir = cfg.processed_dir
    
    output_roc_dir = cfg.output_roc_dir
    output_shap_dir = cfg.output_shap_dir
    output_score_dir = cfg.output_score_dir
    n_run = cfg.n_run
    
    performance_tab = pd.DataFrame(np.zeros((n_run, len(grp.perf_measure))))
    performance_tab.columns = grp.perf_measure

    X = pd.read_csv(os.path.join(root_dir, processed_dir, cfg.input_features), index_col=0)
    Y = pd.read_csv(os.path.join(root_dir, processed_dir, cfg.input_labels), index_col=0)
    
    for i in range(0,n_run):
        print("==================================== iterate",i," running ========================")
        random_seed = i
        model_save_name = model_name + model_alg + model_id + "bootstrap/"+ str(i)
        
        #Generate Training and Testing Set
        X_train, X_test, y_train, y_test = train_test_split(X, Y, stratify=Y, test_size=mparam.setting_params.train_test_ratio, random_state=random_seed) 
        #Generate Training and Evaluation Set
        X_train, X_val, y_train, y_val = train_test_split(X_train, y_train, stratify=y_train, test_size=mparam.setting_params.train_val_ratio, random_state=random_seed) #0.125 * 0.8 = 0.1
        
        print(X_train.shape, X_test.shape, X_val.shape)
        
        if (check_model_exist(root_dir, models_dir, model_save_name,"clf.pk")):
            clf = load_model(root_dir, models_dir, model_save_name,"clf.pk")
            print("Model Exists, Fairness Assessment Running !!!!!!!!!!!!")
        else:
            print("Model Training & Fairness Assessment Running !!!!!!!!!!!!")
            clf, _, fit_params = model_finalize(mparam, X_val=X_val.to_numpy(), y_val=y_val.to_numpy())
            clf.fit(X_train.to_numpy(), y_train.to_numpy(), **fit_params)
            save_model(clf, root_dir, models_dir, model_save_name,"clf.pk")
        
        test_pred = clf.predict(X_test.to_numpy())
        test_pred_prob = clf.predict_proba(X_test.to_numpy())
        
        performance_tab.iloc[i] = performance_score(y_test.to_numpy(), test_pred, test_pred_prob[:, 1])
        
        print(performance_tab.iloc[i])
        
    save_dataframe(performance_tab, root_dir, output_score_dir, model_name+model_alg+model_id+"bootstrap", "performance.csv" )
    save_model({"config":cfg, "param":mparam, "group":grp}, root_dir, models_dir,model_name+model_alg+model_id+"bootstrap","experimental_config.pk")
    
if __name__ == "__main__":

    parser = argparse.ArgumentParser()                                               

    parser.add_argument("--setting", "-s", type=str, required=True)
    
    parser.add_argument("--model_params", "-m", type=str, required=True)
    
    parser.add_argument("--group_info", "-g", type=str, required=True)
    
    args = parser.parse_args()
    
    with open(args.setting) as json_file:
        cfg = json.load(json_file, object_hook=lambda d: SimpleNamespace(**d))
    
    with open(args.model_params) as json_file:
        mparam = json.load(json_file, object_hook=lambda d: SimpleNamespace(**d))
        
    with open(args.group_info) as json_file:
        grp = json.load(json_file, object_hook=lambda d: SimpleNamespace(**d))
        
    execute(cfg, mparam, grp)