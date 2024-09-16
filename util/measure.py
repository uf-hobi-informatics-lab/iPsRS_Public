import os
import csv
import shap
import pickle
import matplotlib.pyplot as plt

from util.io import check_saving_path, save_model, load_model
from sklearn.metrics import confusion_matrix, roc_auc_score ,roc_curve,auc,accuracy_score
from sklearn.metrics import f1_score, roc_auc_score, recall_score, precision_score

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

def calculate_shap(model, data, used_variables, root_dir, output_dir,model_name,model_id):
    explainer_log = shap.Explainer(model, data, feature_names = used_variables)
    shap_values_log = explainer_log(data)
    shap.plots.beeswarm(shap_values_log)
    
    shap.summary_plot(shap_values_log, data, max_display=25, feature_names=used_variables, plot_type="bar",show=False)   
    save_path = check_saving_path(root_dir, output_dir,model_name.split("_")[0],model_name+model_id) 
    plt.savefig(save_path)
        
def performance_score(ground_truth, pred, pred_prob):
    
    f1 = f1_score(ground_truth, pred, zero_division=1, average='binary')
    auroc = roc_auc_score(ground_truth, pred_prob)
    recall = recall_score(ground_truth, pred, average='binary')
    precision = precision_score(ground_truth, pred, average='binary')
    
    tn, fp, fn, tp = confusion_matrix(ground_truth, pred).ravel()
    
    specificity = tn / (tn+fp)
    
    return(f1, auroc, recall, precision, specificity)
        
def fariness_score(protected_ground_truth, privileged_ground_truth, protected_pred, privileged_pred):
    ## confusion matrix
    protected_cm = confusion_matrix(protected_ground_truth, protected_pred)
    privileged_cm = confusion_matrix(privileged_ground_truth, privileged_pred)
    
    # basic model accuracy metrics
    ## protected
    protected_TP = protected_cm[1, 1]
    protected_FP = protected_cm[0, 1]
    protected_TN = protected_cm[0, 0]
    protected_FN = protected_cm[1, 0]
    
    ## privileged
    privileged_TP = privileged_cm[1, 1]
    privileged_FP = privileged_cm[0, 1]
    privileged_TN = privileged_cm[0, 0]
    privileged_FN = privileged_cm[1, 0]
    
    # No.1 Predictive parity: PPV
    protected_PPV = protected_TP/(protected_TP + protected_FP)
    privileged_PPV = privileged_TP/(privileged_TP + privileged_FP)
    
    # No.2 False positive error rate balance: FPR
    protected_FPR = protected_FP/(protected_FP + protected_TN)
    privileged_FPR = privileged_FP/(privileged_FP + privileged_TN)
    
    # No.3 Equalized odds: equal TPR and FPR
    protected_TPR = protected_TP/(protected_TP + protected_FN)
    privileged_TPR = privileged_TP/(privileged_TP + privileged_FN)
    
    # No.4 Conditional use accuracy equality: equal PPV, NPV
    protected_NPV = protected_TN/(protected_TN + protected_FN)
    privileged_NPV = privileged_TN/(privileged_TN + privileged_FN)
    
    # No.5 Treatment equality: FN/FP
    protected_te = protected_FN/protected_FP
    privileged_te = privileged_FN/privileged_FP
    
    # No.6 False negative error rate balance: FNR = FN/(FN + TP)
    protected_FNR = protected_FN/(protected_FN + protected_TP)
    privileged_FNR = privileged_FN/(privileged_FN + privileged_TP)
    
    # No.7 Overall accuracy equality
    protected_ACC = accuracy_score(protected_ground_truth, protected_pred)
    privileged_ACC = accuracy_score(privileged_ground_truth, privileged_pred)
    
    return(protected_PPV, privileged_PPV, protected_FPR, privileged_FPR, 
           protected_TPR, privileged_TPR, protected_NPV, privileged_NPV, 
           protected_te, privileged_te, protected_FNR, privileged_FNR,
           protected_ACC, privileged_ACC)
'''           
def shap_accumulation(model,X_test)
    explainer = shap.Explainer(model.predict, X_test)
    shap_values = explainer(X_test)
    return shap_values
'''