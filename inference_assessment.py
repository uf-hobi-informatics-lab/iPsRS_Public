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

def execute(cfg, grp):
    
    ''' Load settings '''
    model_name = cfg.model_name
    #inference_model_name = cfg.inference_model_name
    model_alg = cfg.model_alg
    model_id = cfg.model_id
    root_dir = cfg.root_dir
    models_dir = cfg.models_dir
    processed_dir = cfg.processed_dir
    
    output_roc_dir = cfg.output_roc_dir
    output_shap_dir = cfg.output_shap_dir
    output_score_dir = cfg.output_score_dir
    n_run = cfg.n_run
    
    fairness_tab = pd.DataFrame(np.zeros((2, len(grp.fair_measure))))
    fairness_tab.columns = grp.fair_measure
    performance_tab = pd.DataFrame(np.zeros((2, len(grp.perf_measure))))
    performance_tab.columns = grp.perf_measure

    MX = pd.read_csv(os.path.join(root_dir, processed_dir, cfg.input_full_features), index_col=0)
    MY = pd.read_csv(os.path.join(root_dir, processed_dir, cfg.input_labels), index_col=0)
    
    #Generate Training and Testing Set
    X_train, X_test, y_train, y_test = train_test_split(MX, MY, stratify=MY, test_size=cfg.setting_params.train_test_ratio, random_state=cfg.setting_params.random_state) 
    X_train, X_val, y_train, y_val = train_test_split(X_train, y_train, stratify=y_train, test_size=cfg.setting_params.train_val_ratio, random_state=cfg.setting_params.random_state) #0.125 * 0.8 = 0.1
    
    X = pd.read_csv(os.path.join(root_dir, processed_dir, cfg.independent_full_features), index_col=0)
    Y = pd.read_csv(os.path.join(root_dir, processed_dir, cfg.independent_labels), index_col=0)
    
    '''
    print(X_train.shape)
    print(X_test.shape)
    print(X_val.shape)
    print(X.shape)                                    
    exit()
    '''
    
    clf = load_model(root_dir, models_dir, model_name, model_alg+model_id  )
    
    ''' Pre-run Settings '''
    if(grp.discrete):
        protected_group_idx = np.where(X[grp.protected_feature_name] == 1) 
        privileged_group_idx = np.where(X[grp.privileged_feature_name] == 1)
        modeling_tr_protected_group_idx = np.where(X_train[grp.protected_feature_name] == 1) 
        modeling_tr_privileged_group_idx = np.where(X_train[grp.privileged_feature_name] == 1) 
        modeling_te_protected_group_idx = np.where(X_test[grp.protected_feature_name] == 1)
        modeling_te_privileged_group_idx = np.where(X_test[grp.privileged_feature_name] == 1) 
    else:
        protected_group_idx = np.where(X[grp.protected_feature_name] < grp.cutoff) 
        privileged_group_idx = np.where(X[grp.privileged_feature_name] >= grp.cutoff)
        modeling_tr_protected_group_idx = np.where(X_train[grp.protected_feature_name] < grp.cutoff) 
        modeling_tr_privileged_group_idx = np.where(X_train[grp.privileged_feature_name] >= grp.cutoff) 
        modeling_te_protected_group_idx = np.where(X_test[grp.protected_feature_name] < grp.cutoff) 
        modeling_te_privileged_group_idx = np.where(X_test[grp.privileged_feature_name] >= grp.cutoff) 
    
    if (grp.is_mask_attr):
        X = X.drop(columns=grp.masked_attrs)
        X_train = X_train.drop(columns=grp.masked_attrs)
        X_test = X_test.drop(columns=grp.masked_attrs)
    
    # Performance
    # inference set
    test_pred = clf.predict(X.to_numpy())
    test_pred_prob = clf.predict_proba(X.to_numpy())
    # modeling set
    modeling_test_pred = clf.predict(X_test.to_numpy())
    modeling_test_pred_prob = clf.predict_proba(X_test.to_numpy())
    
    #Fairness
    # inference set
    y_protected_test = Y.iloc[protected_group_idx]
    y_privileged_test = Y.iloc[privileged_group_idx]
        
    y_protected_pred = test_pred[protected_group_idx]
    y_privileged_pred = test_pred[privileged_group_idx]
    
    # modeling set
    modeling_y_protected_test = y_test.iloc[modeling_te_protected_group_idx]
    modeling_y_privileged_test = y_test.iloc[modeling_te_privileged_group_idx]
        
    modeling_y_protected_pred = modeling_test_pred[modeling_te_protected_group_idx]
    modeling_y_privileged_pred = modeling_test_pred[modeling_te_privileged_group_idx]
    
    
    fairness_tab.iloc[0] = fariness_score(y_protected_test, y_privileged_test, y_protected_pred, y_privileged_pred)
    fairness_tab.iloc[1] = fariness_score(modeling_y_protected_test, modeling_y_privileged_test, modeling_y_protected_pred, modeling_y_privileged_pred)
    performance_tab.iloc[0] = performance_score(Y.to_numpy(), test_pred, test_pred_prob[:, 1])
    performance_tab.iloc[1] = performance_score(y_test.to_numpy(), modeling_test_pred, modeling_test_pred_prob[:, 1])
    
    save_dataframe(fairness_tab, root_dir, output_score_dir, model_name+model_alg+model_id+grp.subgroup+"inference", "fairness(row0_is_independent_set)(row1_is_modeling_set).csv" )
    save_dataframe(performance_tab, root_dir, output_score_dir, model_name+model_alg+model_id+"inference", "performance(row0_is_independent_set)(row1_is_modeling_set).csv" )
    
    print("MODEL INFO: ", root_dir+ models_dir+ model_name+ model_alg+model_id)
    print("MODEL Settings: ", clf)
    print("MODEL Performance: ", performance_tab)
    print("MODEL Fairness: ", fairness_tab)
    print("GV BEST PARAMETERS: ", clf.best_params_)
    
if __name__ == "__main__":

    parser = argparse.ArgumentParser()                                               

    parser.add_argument("--setting", "-s", type=str, required=True)
    
    parser.add_argument("--group_info", "-g", type=str, required=True)
    
    args = parser.parse_args()
    
    with open(args.setting) as json_file:
        cfg = json.load(json_file, object_hook=lambda d: SimpleNamespace(**d))
    
    with open(args.group_info) as json_file:
        grp = json.load(json_file, object_hook=lambda d: SimpleNamespace(**d))
    
    execute(cfg, grp)