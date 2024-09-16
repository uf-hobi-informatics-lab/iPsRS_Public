from util.measure import performance_score, fariness_score
from util.io import save_dataframe, check_saving_path, save_model
from config.adl import AdversarialDebiasing
from config.model import prep_construction, imb_construction
from sklearn.model_selection import train_test_split
from aif360.datasets import StandardDataset

from types import SimpleNamespace

import os
import csv
import json
import argparse
import numpy as np
import pandas as pd
import tensorflow.compat.v1 as tf
tf.disable_eager_execution()
os.environ['CUDA_VISIBLE_DEVICES'] = '-1'

def execute(cfg, grp):
    
    ''' Load settings '''
    model_name = cfg.model_name
    model_alg = cfg.model_alg
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

    X = pd.read_csv(os.path.join(root_dir, processed_dir, cfg.input_full_features), index_col=0)
    Y = pd.read_csv(os.path.join(root_dir, processed_dir, cfg.input_labels), index_col=0)
    
    IX = pd.read_csv(os.path.join(root_dir, processed_dir, cfg.independent_full_features), index_col=0)
    IY = pd.read_csv(os.path.join(root_dir, processed_dir, cfg.independent_labels), index_col=0)
    
    unique, counts = np.unique(Y, return_counts=True)

    print(np.asarray((unique, counts)).T)
        
    for i in range(0,n_run):
        
        print("==================================== iterate",i," running ========================")
        random_seed = i
        model_save_name = model_name + model_alg + grp.subgroup +"/"+ str(i)
        
        #Generate Training and Testing Set
        X_train, X_test, y_train, y_test = train_test_split(X, Y, stratify=Y, test_size=0.2, random_state=random_seed) 
        #Generate Training and Evaluation Set
        X_train, X_val, y_train, y_val = train_test_split(X_train, y_train, stratify=y_train, test_size=0.125, random_state=random_seed) #0.125 * 0.8 = 0.1

        ''' Data Preprocessing'''
        prep_c = prep_construction()
        imbp = imb_construction(cfg)

        X_train_features = prep_c.fit_transform(X_train)
        X_test_features = prep_c.transform(X_test)
        X_val_features = prep_c.transform(X_val)
        
        #X_backup = X_train_features
        
        IX_features = prep_c.fit_transform(IX)
        
        if imbp is not None:
            X_train_features, y_train = imbp.fit_resample(X_train_features, y_train)

        X_train = pd.DataFrame(X_train_features, columns=X_train.columns)
        X_test = pd.DataFrame(X_test_features, index = X_test.index, columns=X_test.columns)
        X_val = pd.DataFrame(X_val_features, index = X_val.index, columns=X_val.columns)
        IX_test = pd.DataFrame(IX_features, index = IX.index, columns=IX.columns)

        #y_train = pd.DataFrame(y_train.values, columns=y_train.columns)
        #y_test = pd.DataFrame(y_test.values, columns=y_test.columns)
        #y_val = pd.DataFrame(y_val.values, columns=y_val.columns)
        
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
        print(X_train_features.shape)
        print(X_test_features.shape)
        print(X_val_features.shape)
        print(IX_features.shape)   
        
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
            
        sess = tf.Session()
        sess.close()
        tf.reset_default_graph()
        sess = tf.Session()

        # Learn parameters with debias set to True
        debiased_model = AdversarialDebiasing(privileged_groups = privileged_groups,
                                              unprivileged_groups = unprivileged_groups,
                                              scope_name='debiased_classifier',
                                              debias=True,
                                              num_epochs = cfg.model_params.num_epochs,
                                              classifier_num_hidden_units=cfg.model_params.classifier_num_hidden_units,
                                              learning_rate = cfg.model_params.learning_rate,
                                              seed = cfg.model_params.seed,
                                              sess=sess)

        '''Model Traning and Testing'''
        debiased_model.fit(dataset_orig_train, tr_protected_features)
        save_dir = os.path.join(root_dir, models_dir, model_save_name)
        save_path = check_saving_path(save_dir,"save_param")
        saver = tf.compat.v1.train.Saver()
        save_path = saver.save(sess, save_path)
        print("Model saved in path: %s" % save_path)

        # Apply the plain model to test data
        dataset_debiasing_train = debiased_model.predict(dataset_orig_train, tr_protected_features)
        dataset_debiasing_test = debiased_model.predict(dataset_orig_test, te_protected_features)
        dataSet_debiasing_independent = debiased_model.predict(dataset_orig_independent, id_protected_features)

        '''Prepare Results to the Output Format'''
        y_protected_test = dataset_orig_test.labels[te_protected_group_idx]
        y_privileged_test = dataset_orig_test.labels[te_privileged_group_idx]
        
        y_protected_pred = dataset_debiasing_test.labels[te_protected_group_idx]
        y_privileged_pred = dataset_debiasing_test.labels[te_privileged_group_idx]

        y_protected_IY = dataset_orig_independent.labels[id_protected_group_idx]
        y_privileged_IY = dataset_orig_independent.labels[id_privileged_group_idx]
        
        y_protected_IY_pred = dataSet_debiasing_independent.labels[id_protected_group_idx]
        y_privileged_IY_pred = dataSet_debiasing_independent.labels[id_privileged_group_idx]
        
        fairness_tab.iloc[i] = fariness_score(y_protected_test, y_privileged_test, y_protected_pred, y_privileged_pred)
        performance_tab.iloc[i] = performance_score(dataset_orig_test.labels, dataset_debiasing_test.labels, dataset_debiasing_test.scores)
        
        id_fairness_tab.iloc[i] = fariness_score(y_protected_IY, y_privileged_IY, y_protected_IY_pred, y_privileged_IY_pred)
        id_performance_tab.iloc[i] = performance_score(dataset_orig_independent.labels, dataSet_debiasing_independent.labels, dataSet_debiasing_independent.scores)
        #print(dataset_debiasing_test.scores)
        
    save_dataframe(fairness_tab, root_dir, output_score_dir, model_name+model_alg+grp.subgroup, "fairness.csv" )
    save_dataframe(performance_tab, root_dir, output_score_dir, model_name+model_alg+grp.subgroup, "performance.csv" )
    save_dataframe(id_fairness_tab, root_dir, output_score_dir, model_name+model_alg+grp.subgroup, "indenpendent_fairness.csv" )
    save_dataframe(id_performance_tab, root_dir, output_score_dir, model_name+model_alg+grp.subgroup, "indenpendent_performance.csv" )
    save_model({"config":cfg, "group":grp}, root_dir, models_dir,model_name+model_alg+grp.subgroup,"experimental_config.pk")
    
    print(fairness_tab)
    print(performance_tab)
    print(id_fairness_tab)
    print(id_performance_tab)
    
if __name__ == "__main__":

    parser = argparse.ArgumentParser()                                               

    parser.add_argument("--setting", "-s", type=str, required=True)
    
    parser.add_argument("--group_info", "-g", type=str, required=True)
    
    parser.add_argument("--model_param", "-m", type=str, required=True)
    
    args = parser.parse_args()
    
    with open(args.setting) as json_file:
        cfg = json.load(json_file, object_hook=lambda d: SimpleNamespace(**d))
    
    with open(args.group_info) as json_file:
        grp = json.load(json_file, object_hook=lambda d: SimpleNamespace(**d))
        
    execute(cfg, grp)