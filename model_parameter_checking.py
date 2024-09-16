import os
import sys
import csv
import json
import warnings
import argparse
import pandas as pd
import numpy as np

from sklearn.model_selection import train_test_split
from types import SimpleNamespace
from util.measure import performance_score

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
    used_variables = raw_X.columns.tolist()
    X = raw_X.to_numpy()
    Y = pd.read_csv(os.path.join(root_dir, processed_dir, cfg.input_labels), index_col=0).to_numpy().flatten()
   

    #Generate Training and Testing Set
    X_train, X_test, y_train, y_test = train_test_split(X, Y, stratify=Y, test_size=cfg.setting_params.train_test_ratio, random_state=cfg.setting_params.random_state) 
    #Generate Training and Evaluation Set
    X_train, X_val, y_train, y_val = train_test_split(X_train, y_train, stratify=y_train, test_size=cfg.setting_params.train_val_ratio, random_state=cfg.setting_params.random_state) #0.125 * 0.8 = 0.1
    
    model = load_model(root_dir, models_dir, model_name, model_alg+model_id  )

    #prediction
    train_pred = model.predict(X_train)
    train_pred_prob = model.predict_proba(X_train)

    test_pred = model.predict(X_test)
    test_pred_prob = model.predict_proba(X_test)

    val_pred = model.predict(X_val)
    val_pred_prob = model.predict_proba(X_val)

    #create status index
    train_flag_list = np.array(["train" for x in y_train])
    test_flag_list = np.array(["test" for x in y_test])
    val_flag_list = np.array(["val" for x in y_val])
    
    #format transformation
    train_flag_list = np.reshape(train_flag_list, (-1, 1))
    test_flag_list = np.reshape(test_flag_list, (-1, 1))
    val_flag_list = np.reshape(val_flag_list, (-1, 1))
    
    train_score = np.reshape(train_pred_prob[:, 1], (-1,1))
    test_score = np.reshape(test_pred_prob[:, 1], (-1,1))
    val_score = np.reshape(val_pred_prob[:, 1], (-1,1))
    
    #data formation
    train_label = np.reshape(y_train, (-1, 1))
    train_with_outcome = np.concatenate((X_train, train_label), axis=1)
    train_with_score = np.concatenate((train_with_outcome, train_score), axis=1)
    train_output = np.concatenate((train_with_score, train_flag_list), axis=1)
    
    test_label = np.reshape(y_test, (-1, 1))
    test_with_outcome = np.concatenate((X_test, test_label), axis=1)
    test_with_score = np.concatenate((test_with_outcome, test_score), axis=1)
    test_output = np.concatenate((test_with_score, test_flag_list), axis=1)
    
    val_label = np.reshape(y_val, (-1, 1))
    val_with_outcome = np.concatenate((X_val, val_label), axis=1)
    val_with_score = np.concatenate((val_with_outcome, val_score), axis=1)
    val_output = np.concatenate((val_with_score, val_flag_list), axis=1)
    
    #column name creation
    used_variables.append("hospitalization")
    used_variables.append(model_alg+"score")
    used_variables.append("Train/Val/Test")
    
    #output df generation
    train_output_df = pd.DataFrame(data=train_output, columns = used_variables)
    test_output_df = pd.DataFrame(data=test_output, columns = used_variables)
    val_output_df = pd.DataFrame(data=val_output, columns = used_variables)
    
    #output combination
    total_output_df = train_output_df
    total_output_df = pd.concat([total_output_df, val_output_df])
    total_output_df = pd.concat([total_output_df, test_output_df])
    
    #total_output_df.to_csv(model_name + model_alg + model_id+".csv")
    
    performance = performance_score(y_test, test_pred, test_pred_prob[:, 1])
    performance_on_training = performance_score(y_train, train_pred, train_pred_prob[:, 1])
    
    print("MODEL INFO: ", model_name + model_alg + model_id)
    print("MODEL Settings: ", model)
    print("MODEL Training Performance: ", performance_on_training)
    print("MODEL Performance: ", performance)
    print("GV BEST PARAMETERS: ", model.best_params_)
    
    save_params = dict()
    
    save_params['model_alg'] = model_alg
    save_params['setting_params'] = cfg.setting_params.__dict__
    
    for (key, value) in model.best_params_.items():
        save_params[key.split("__")[-1]] = value
    
    save_params_as_json(save_params,cfg.output_best_param_dir, model_name, model_alg, model_id+"best")
    
    print("Save BEST PARAMETERS TO: ", cfg.output_best_param_dir)

if __name__ == "__main__":

    parser = argparse.ArgumentParser()                                               

    parser.add_argument("--setting", "-s", type=str, required=True)
    
    args = parser.parse_args()
    
    with open(args.setting) as json_file:
        cfg = json.load(json_file, object_hook=lambda d: SimpleNamespace(**d))
    
    execute(cfg)