import os
import csv
import shap
import pickle
import pyreadr
import numpy as np
import tqdm as tqdm
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.metrics import confusion_matrix, roc_auc_score ,roc_curve,auc
from sklearn.metrics import f1_score, roc_auc_score, recall_score, precision_score
from sklearn.model_selection import GridSearchCV,StratifiedKFold, train_test_split, KFold, RandomizedSearchCV

from config.model import xgb_finalize
from config.gbm import gbm_construction
from util.io import check_saving_path, save_model, load_model

def calculate_score(score_dict,root_dir, output_dir,model_name,model_id):
    head = score_dict.keys()
    head = [val for val in head for _ in (0, 1)]
    
    for idx, val in enumerate(head): 
        if idx%2!=0: head[idx] = val+"_std"
        else: head[idx] = val+"_mean"

    flatten_result = []
    for key in score_dict:
        flatten_result.append(np.mean(score_dict[key]))
        flatten_result.append(np.std(score_dict[key]))
    
    save_path = check_saving_path(root_dir, output_dir,model_name.split("_")[0],model_name+model_id) 
    
    with open(save_path, 'w') as csvfile: 
        # creating a csv writer object 
        csvwriter = csv.writer(csvfile) 
        # writing the fields 
        csvwriter.writerow(head) 
        # writing the data rows 
        csvwriter.writerow(flatten_result)
        
        
def calculate_auroc(model, X_train, X_test, y_train, y_test, root_dir, output_dir,model_name,model_id):
    # predicted value: training error
    pred_logtrain = model.predict_proba(X_train)
    fpr_logtrain, tpr_logtrain, thresh_logtrain = roc_curve(y_train, pred_logtrain[:, 1], pos_label=1)
    auc_logtrain = roc_auc_score(y_train, pred_logtrain[:, 1])
    
    plt.figure()
    plt.plot(fpr_logtrain, tpr_logtrain, linestyle='-', label='Training set AUC (' + str(round(auc_logtrain, 3)) + ')')
    plt.legend()
    plt.xlabel('1 - specificity (FPR)')
    plt.ylabel('Sensitivity (TPR)')

    # predicted: tesing error
    pred_logtest = model.predict_proba(X_test)
    fpr_logtest, tpr_logtest, thresh_logtest = roc_curve(y_test, pred_logtest[:, 1], pos_label=1)
    auc_logtest = roc_auc_score(y_test, pred_logtest[:, 1])
    plt.plot(fpr_logtest, tpr_logtest, linestyle='-', label='Testing set AUC (' + str(round(auc_logtest, 3)) + ')')
    plt.legend()
    plt.xlabel('1 - specificity (FPR)')
    plt.ylabel('Sensitivity (TPR)')
    plt.title('Logistic model training and testing ROC for "hospitalization_flag" - ROC Curve (AUC)')
    
    save_path = check_saving_path(root_dir, output_dir,model_name.split("_")[0],model_name+model_id) 
    plt.savefig(save_path)
    plt.close()
    
    
## beeswarm
def calculate_shap(model, data, used_variables, root_dir, output_dir,model_name,model_id):
    explainer_log = shap.Explainer(model, data, feature_names = used_variables)
    shap_values_log = explainer_log(data)
    shap.plots.beeswarm(shap_values_log)
    
    shap.summary_plot(shap_values_log, data, max_display=25, feature_names=used_variables, plot_type="bar",show=False)   
    save_path = check_saving_path(root_dir, output_dir,model_name.split("_")[0],model_name+model_id) 
    plt.savefig(save_path)