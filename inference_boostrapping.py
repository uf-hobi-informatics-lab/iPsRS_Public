import os
import sys
import csv
import json
import warnings
import argparse
import numpy as np
import pandas as pd

from types import SimpleNamespace

from util.io import *

from util.measure import performance_score, fariness_score
from sklearn.model_selection import train_test_split
from types import SimpleNamespace

def execute(cfg, mparam, grp):
    
    ''' Load settings '''
    model_name = cfg.model_name
    inference_model_name = cfg.inference_model_name
    model_alg = cfg.model_alg
    model_id = cfg.model_id
    root_dir = cfg.root_dir
    models_dir = cfg.models_dir
    processed_dir = cfg.processed_dir
    
    output_roc_dir = cfg.output_roc_dir
    output_shap_dir = cfg.output_shap_dir
    output_score_dir = cfg.output_score_dir
    n_run = cfg.n_run
    
    fairness_tab = pd.DataFrame(np.zeros((n_run, len(grp.fair_measure))))
    fairness_tab.columns = grp.fair_measure
    performance_tab = pd.DataFrame(np.zeros((n_run, len(grp.perf_measure))))
    performance_tab.columns = grp.perf_measure

    X = pd.read_csv(os.path.join(root_dir, processed_dir, cfg.input_features), index_col=0)
    Y = pd.read_csv(os.path.join(root_dir, processed_dir, cfg.input_labels), index_col=0)
    
    for i in range(0,n_run):
        print("==================================== iterate",i," running ========================")
        random_seed = i
        
        model_save_name = inference_model_name + model_alg + model_id + "bootstrap/"+ str(i)
        
        print(root_dir, models_dir, model_save_name,"clf.pk")
        
        ''' Pre-run Settings '''
        if(grp.discrete):
            protected_group_idx = np.where(X[grp.protected_feature_name] == 1) 
            privileged_group_idx = np.where(X[grp.privileged_feature_name] == 1)
        else:
            protected_group_idx = np.where(X[grp.protected_feature_name] < grp.cutoff) 
            privileged_group_idx = np.where(X[grp.privileged_feature_name] >= grp.cutoff)
            
        
        if (grp.is_mask_attr):
            X = X.drop(columns=grp.masked_attrs)
        
        print(X.shape)
        
        if (check_model_exist(root_dir, models_dir, model_save_name,"clf.pk")):
            clf = load_model(root_dir, models_dir, model_save_name,"clf.pk")
            print("Model Exists, Fairness Assessment Running !!!!!!!!!!!!")
        else:
            print("Error, no Model founded!!!!!!!!")
            exit()
            
        test_pred = clf.predict(X.to_numpy())
        test_pred_prob = clf.predict_proba(X.to_numpy())
        
        y_protected_test = Y.iloc[protected_group_idx]
        y_privileged_test = Y.iloc[privileged_group_idx]
        
        y_protected_pred = test_pred[protected_group_idx]
        y_privileged_pred = test_pred[privileged_group_idx]
        
        fairness_tab.iloc[i] = fariness_score(y_protected_test, y_privileged_test, y_protected_pred, y_privileged_pred)
        performance_tab.iloc[i] = performance_score(Y.to_numpy(), test_pred, test_pred_prob[:, 1])
        
        for row in test_pred_prob:
            print(row)
            
        print(fairness_tab)
        print(performance_tab)
        
        break
        
    save_dataframe(fairness_tab, root_dir, output_score_dir, model_name+model_alg+model_id+grp.subgroup+"bootstrap", "fairness.csv" )
    save_dataframe(performance_tab, root_dir, output_score_dir, model_name+model_id+model_alg+"bootstrap", "performance.csv" )
    
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