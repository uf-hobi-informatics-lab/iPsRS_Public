from util.measure import performance_score, fariness_score
from util.io import save_dataframe, check_saving_path, save_model
from config.model import prep_construction, imb_construction, model_fml
from sklearn.model_selection import train_test_split
from aif360.datasets import StandardDataset
from aif360.algorithms.postprocessing.calibrated_eq_odds_postprocessing import CalibratedEqOddsPostprocessing
from types import SimpleNamespace

import os
import csv
import json
import argparse
import numpy as np
import pandas as pd


def execute(cfg, mparam, grp):
    
    ''' Load settings '''
    model_name = cfg.model_name
    model_alg = mparam.model_alg
    root_dir = cfg.root_dir
    processed_dir = cfg.processed_dir
    models_dir = cfg.models_dir
    output_roc_dir = cfg.output_roc_dir
    output_shap_dir = cfg.output_shap_dir
    output_score_dir = cfg.output_score_dir
    n_run = cfg.n_run
    
    fairness_tab = pd.DataFrame(np.zeros((n_run, len(grp.fair_measure))))
    fairness_tab.columns = grp.fair_measure
    performance_tab = pd.DataFrame(np.zeros((n_run, len(grp.perf_measure))))
    performance_tab.columns = grp.perf_measure

    id_fairness_tab = pd.DataFrame(np.zeros((n_run, len(grp.fair_measure))))
    id_fairness_tab.columns = grp.fair_measure
    id_performance_tab = pd.DataFrame(np.zeros((n_run, len(grp.perf_measure))))
    id_performance_tab.columns = grp.perf_measure
    
    X = pd.read_csv(os.path.join(root_dir, processed_dir, cfg.input_features), index_col=0)
    Y = pd.read_csv(os.path.join(root_dir, processed_dir, cfg.input_labels), index_col=0)
    
    IX = pd.read_csv(os.path.join(root_dir, processed_dir, cfg.independent_full_features), index_col=0)
    IY = pd.read_csv(os.path.join(root_dir, processed_dir, cfg.independent_labels), index_col=0)
    
    for i in range(0,n_run):
        print("==================================== iterate",i," running ========================")
        random_seed = i
        model_save_name = model_name + model_alg + grp.subgroup +"/"+ str(i)

        #Generate Training and Testing Set
        X_train, X_test, y_train, y_test = train_test_split(X, Y, stratify=Y, test_size=mparam.setting_params.train_test_ratio, random_state=random_seed) 
        #Generate Training and Evaluation Set
        X_train, X_val, y_train, y_val = train_test_split(X_train, y_train, stratify=y_train, test_size=mparam.setting_params.train_val_ratio, random_state=random_seed) #0.125 * 0.8 = 0.1

        ''' Data Preprocessing'''
        prep_c = prep_construction()
        imbp = imb_construction(mparam)

        X_train_features = prep_c.fit_transform(X_train)
        X_test_features = prep_c.transform(X_test)
        X_val_features = prep_c.transform(X_val)
        
        #X_backup = X_train_features
        
        IX_features = prep_c.fit_transform(IX)

        if imbp is not None:
            X_train_features, y_train = imbp.fit_resample(X_train_features, y_train)
            X_val_imp_features, y_val_imp = imbp.fit_resample(X_val_features, y_val)
            X_val_imp = pd.DataFrame(X_val_imp_features, columns=X_val.columns)

        X_train = pd.DataFrame(X_train_features, columns=X_train.columns)
        X_test = pd.DataFrame(X_test_features, index = X_test.index, columns=X_test.columns)
        X_val = pd.DataFrame(X_val_features, index = X_val.index, columns=X_val.columns)
        IX_test = pd.DataFrame(IX_features, index = IX.index, columns=IX.columns)
        
        ''' Transform the Data to AIF-360 Data Format '''
        train_df = pd.concat([X_train, y_train], axis=1, join='inner')
        dataset_orig_train = StandardDataset(train_df,
                                             grp.label_name,
                                             grp.favorable_classes,
                                             grp.protected_attribute_names,
                                             grp.privileged_classes)

        test_df = pd.concat([X_test, y_test], axis=1, join='inner')
        dataset_orig_test = StandardDataset(test_df,
                                             grp.label_name,
                                             grp.favorable_classes,
                                             grp.protected_attribute_names,
                                             grp.privileged_classes)
        
        val_df = pd.concat([X_val, y_val], axis=1, join='inner')
        dataset_orig_valid = StandardDataset(val_df,
                                             grp.label_name,
                                             grp.favorable_classes,
                                             grp.protected_attribute_names,
                                             grp.privileged_classes)
        
        independent_df = pd.concat([IX_test, IY], axis=1, join='inner')
        dataset_orig_independent = StandardDataset(independent_df,
                                             grp.label_name,
                                             grp.favorable_classes,
                                             grp.protected_attribute_names,
                                             grp.privileged_classes)
        '''
        print(train_df.shape)
        print(test_df.shape)
        print(val_df.shape)
        print(independent_df.shape)                                    
        exit()
        '''
        
        ''' Pre-run Settings '''
        unprivileged_groups = [{grp.protected_feature_name: grp.protected_feature_val}]
        privileged_groups  = [{grp.privileged_feature_name: grp.privileged_feature_val}]
        protected_index = dataset_orig_train.feature_names.index(grp.protected_feature_name)
        privileged_index = dataset_orig_train.feature_names.index(grp.privileged_feature_name)

        tr_protected_features = np.reshape(dataset_orig_train.features[:,protected_index],[-1,1])
        te_protected_features = np.reshape(dataset_orig_test.features[:,protected_index],[-1,1])
        vl_protected_features = np.reshape(dataset_orig_valid.features[:,protected_index],[-1,1])
        id_protected_features = np.reshape(dataset_orig_independent.features[:,protected_index],[-1,1])

        tr_protected_group_idx = dataset_orig_train.features[:,protected_index] == 1
        tr_privileged_group_idx = dataset_orig_train.features[:,privileged_index] == 1
        te_protected_group_idx = dataset_orig_test.features[:,protected_index] == 1
        te_privileged_group_idx = dataset_orig_test.features[:,privileged_index] == 1
        vl_protected_group_idx = dataset_orig_valid.features[:,protected_index] == 1
        vl_privileged_group_idx = dataset_orig_valid.features[:,privileged_index] == 1

        id_protected_group_idx = dataset_orig_independent.features[:,protected_index] == 1
        id_privileged_group_idx = dataset_orig_independent.features[:,privileged_index] == 1
        
        if (grp.is_mask_attr):
            masked_index = []
            masked_attrs = grp.masked_attrs
            for attr in masked_attrs:
                masked_index.append(dataset_orig_train.feature_names.index(attr) )

            dataset_orig_train.features = np.delete(dataset_orig_train.features, masked_index, axis=1)
            dataset_orig_test.features = np.delete(dataset_orig_test.features, masked_index, axis=1)
            dataset_orig_valid.features = np.delete(dataset_orig_valid.features, masked_index, axis=1)
            dataset_orig_independent.features = np.delete(dataset_orig_independent.features, masked_index, axis=1)

        '''Run Modeling'''
        clf, _, fit_params = model_fml(mparam, 
                                     X_val=dataset_orig_valid.features, 
                                     y_val=dataset_orig_valid.labels)
        clf.fit(dataset_orig_train.features, dataset_orig_train.labels, **fit_params)
        
        class_thresh = cfg.class_thresh
        cost_constraint = cfg.cost_constraint

        fav_idx = np.where(clf.classes_ == dataset_orig_train.favorable_label)[0][0]
        
        #validation set prediction
        y_valid_pred_prob = clf.predict_proba(dataset_orig_valid.features)
        dataset_orig_valid_pred = dataset_orig_valid.copy(deepcopy=True)
        y_valid_pred_prob = y_valid_pred_prob[:,fav_idx]

        y_valid_pred = clf.predict(dataset_orig_valid.features)
        dataset_orig_valid_pred.labels = np.reshape(y_valid_pred, (-1, 1))
        
        #testing set prediction
        y_test_pred_prob = clf.predict_proba(dataset_orig_test.features)
        dataset_orig_test_pred = dataset_orig_test.copy(deepcopy=True)
        y_test_pred_prob = y_test_pred_prob[:,fav_idx]

        y_test_pred = clf.predict(dataset_orig_test.features)
        dataset_orig_test_pred.labels = np.reshape(y_test_pred, (-1, 1))

        #independent set prediction
        y_independent_pred_prob = clf.predict_proba(dataset_orig_independent.features)
        dataset_orig_independent_pred = dataset_orig_independent.copy(deepcopy=True)
        y_independent_pred_prob = y_independent_pred_prob[:,fav_idx]

        y_independent_pred = clf.predict(dataset_orig_independent.features)
        dataset_orig_independent.labels = np.reshape(y_independent_pred, (-1, 1))

        dataset_orig_valid_pred.scores = y_valid_pred_prob.reshape(-1,1)
        dataset_orig_test_pred.scores = y_test_pred_prob.reshape(-1,1)
        dataset_orig_independent.scores = y_independent_pred_prob.reshape(-1,1)

        #validation set prediction
        y_valid_pred = np.zeros_like(dataset_orig_valid_pred.labels)
        y_valid_pred[y_valid_pred_prob >= class_thresh] = dataset_orig_valid_pred.favorable_label
        y_valid_pred[~(y_valid_pred_prob >= class_thresh)] = dataset_orig_valid_pred.unfavorable_label
        dataset_orig_valid_pred.labels = y_valid_pred

        #testing set prediction
        y_test_pred = np.zeros_like(dataset_orig_test_pred.labels)
        y_test_pred[y_test_pred_prob >= class_thresh] = dataset_orig_test_pred.favorable_label
        y_test_pred[~(y_test_pred_prob >= class_thresh)] = dataset_orig_test_pred.unfavorable_label
        dataset_orig_test_pred.labels = y_test_pred
        
        #independent set prediction
        y_independent_pred = np.zeros_like(dataset_orig_independent_pred.labels)
        y_independent_pred[y_independent_pred_prob >= class_thresh] = dataset_orig_independent_pred.favorable_label
        y_independent_pred[~(y_independent_pred_prob >= class_thresh)] = dataset_orig_independent_pred.unfavorable_label
        dataset_orig_independent_pred.labels = y_independent_pred
        
        ''' Calibration '''
        cpp = CalibratedEqOddsPostprocessing(privileged_groups = privileged_groups,
                                     unprivileged_groups = unprivileged_groups,
                                     cost_constraint=cost_constraint,
                                     seed=mparam.setting_params.random_state)
        
        cpp = cpp.fit(dataset_orig_valid, dataset_orig_valid_pred)
        
        dataset_transf_valid_pred = cpp.predict(dataset_orig_valid_pred)
        dataset_transf_test_pred = cpp.predict(dataset_orig_test_pred)
        dataset_transf_independent_pred = cpp.predict(dataset_orig_independent_pred)
        
        
        '''Prepare Results to the Output Format'''
        y_protected_test = dataset_orig_test.labels[te_protected_group_idx]
        y_privileged_test = dataset_orig_test.labels[te_privileged_group_idx]
        
        y_protected_pred = dataset_transf_test_pred.labels[te_protected_group_idx]
        y_privileged_pred = dataset_transf_test_pred.labels[te_privileged_group_idx]
        
        y_protected_IY = dataset_orig_independent.labels[id_protected_group_idx]
        y_privileged_IY = dataset_orig_independent.labels[id_privileged_group_idx]
        
        y_protected_IY_pred = dataset_transf_independent_pred.labels[id_protected_group_idx]
        y_privileged_IY_pred = dataset_transf_independent_pred.labels[id_privileged_group_idx]
        
        
        fairness_tab.iloc[i] = fariness_score(y_protected_test, y_privileged_test, y_protected_pred, y_privileged_pred)
        performance_tab.iloc[i] = performance_score(dataset_orig_test.labels, dataset_transf_test_pred.labels, dataset_transf_test_pred.scores)
        
        id_fairness_tab.iloc[i] = fariness_score(y_protected_IY, y_privileged_IY, y_protected_IY_pred, y_privileged_IY_pred)
        id_performance_tab.iloc[i] = performance_score(dataset_orig_independent.labels, dataset_transf_independent_pred.labels, dataset_transf_independent_pred.scores)
        
        save_model(cpp, root_dir, models_dir,model_save_name,"cpp.pk")
        save_model(clf, root_dir, models_dir,model_save_name,"clf.pk")
        
    save_dataframe(fairness_tab, root_dir, output_score_dir, model_name+model_alg+grp.subgroup, "fairness.csv" )
    save_dataframe(performance_tab, root_dir, output_score_dir, model_name+model_alg+grp.subgroup, "performance.csv" )
    save_dataframe(id_fairness_tab, root_dir, output_score_dir, model_name+model_alg+grp.subgroup, "indenpendent_fairness.csv" )
    save_dataframe(id_performance_tab, root_dir, output_score_dir, model_name+model_alg+grp.subgroup, "indenpendent_performance.csv" )
    save_model({"config":cfg, "param":mparam, "group":grp}, root_dir, models_dir,model_name+model_alg+grp.subgroup,"experimental_config.pk")
    
    print(fairness_tab)
    print(performance_tab)
    
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
