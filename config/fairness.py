import scipy
import math
import pingouin as pg
from statistics import mean
import os
import csv
import shap
import pickle
import pyreadr
import numpy as np
import tqdm as tqdm
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import MinMaxScaler
from imblearn.over_sampling import SMOTE 
import xgboost as xgb
from imblearn.pipeline import Pipeline

from sklearn.metrics import confusion_matrix, accuracy_score
from util.io import check_saving_path, save_model, load_model
from sklearn.model_selection import train_test_split


def fariness_seven(protected, privileged, X_full, X_test, y_test, clf):
    # subugroup protected and privileged by index
    ## extract index
    X_full_pro = X_full.loc[X_full[protected] == 1]
    X_full_pri = X_full.loc[X_full[privileged] == 1]
    pro_index = X_full_pro.iloc[:,0]
    pri_index = X_full_pri.iloc[:,0]
    
    # subgroup test set
    ## X_test
    X_test_pro = X_test.loc[X_test['index'].isin(pro_index)]
    X_test_pri = X_test.loc[X_test['index'].isin(pri_index)]
    
    X_realtest_pro = X_test_pro.drop('index', axis=1)
    X_realtest_pri = X_test_pri.drop('index', axis=1)
    
    ## Y_test
    y_test_pro = y_test.loc[y_test['index'].isin(pro_index)]
    y_test_pri = y_test.loc[y_test['index'].isin(pri_index)]
    
    y_realtest_pro = y_test_pro.drop('index', axis=1)
    y_realtest_pri = y_test_pri.drop('index', axis=1)

    ## generate predicted value
    y_protected_pred = clf.predict(X_realtest_pro)
    y_privileged_pred = clf.predict(X_realtest_pri)
    
    ## confusion matrix
    protected_cm = confusion_matrix(y_realtest_pro, y_protected_pred)
    privileged_cm = confusion_matrix(y_realtest_pri, y_privileged_pred)
    
    # basic model accuracy metrics
    ## protected
    protected_TP = protected_cm[1, 1]
    protected_FP = protected_cm[0,1]
    protected_TN = protected_cm[0, 0]
    protected_FN = protected_cm[1,0]
    
    ## privileged
    privileged_TP = privileged_cm[1, 1]
    privileged_FP = privileged_cm[0,1]
    privileged_TN = privileged_cm[0, 0]
    privileged_FN = privileged_cm[1,0]
    
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
    protected_ACC = accuracy_score(y_realtest_pro, y_protected_pred)
    privileged_ACC = accuracy_score(y_realtest_pri, y_privileged_pred)
    
    return(protected_PPV, privileged_PPV, protected_FPR, privileged_FPR, 
           protected_TPR, privileged_TPR, protected_NPV, privileged_NPV, 
           protected_te, privileged_te, protected_FNR, privileged_FNR,
           protected_ACC, privileged_ACC)

def f_test(x, y):
    x = np.array(x)
    y = np.array(y)
    f = np.var(x[np.isfinite(x)], ddof=1)/np.var(y[np.isfinite(y)], ddof=1) #calculate F test statistic 
    dfn = x.size-1 #define degrees of freedom numerator 
    dfd = y.size-1 #define degrees of freedom denominator 
    p = 1-scipy.stats.f.cdf(f, dfn, dfd) #find p-value of F test statistic 
    # if p >= 0.001:
    #     p = p
    # else:
    #     p = 0.001
    return p

def fariness_report_tab(n, fariness_tab_raw):
    colname_14 = ["PPV", "PPV", "FPR", "FPR", "TPR", "TPR", "NPV", "NPV", "TE", "TE", "FNR", "FNR", "ACC", "ACC"]
    colname_7 = ["PPV", "FPR", "TPR", "NPV", "TE", "FNR", "ACC"]
    
    fariness_tab = fariness_tab_raw[~fariness_tab_raw.isin([np.nan, np.inf, -np.inf]).any(1)]
    # sum statistics
    summary_table = pd.DataFrame(np.zeros((4, 14)))
    ## mean
    summary_table.iloc[0] = np.mean(fariness_tab[np.isfinite(fariness_tab)], axis=0)
    ## std
    summary_table.iloc[1] = np.std(fariness_tab[np.isfinite(fariness_tab)], axis=0)
    ## CI upper
    summary_table.iloc[2] = summary_table.iloc[0] - 1.96 * summary_table.iloc[1]/math.sqrt(n)
    ## CI lower
    summary_table.iloc[3] = summary_table.iloc[0] + 1.96 * summary_table.iloc[1]/math.sqrt(n)
    
    # test statistics
    test_table = pd.DataFrame(np.zeros((3, 7)))

    for i in range(0,14,2):
        ## F-test
        test_table.iloc[0, int(i/2)] = f_test(fariness_tab.iloc[:,i],fariness_tab.iloc[:,i+1])
        
        ## two sample t test
        if test_table.iloc[0, int(i/2)] > 0.05:
            test_table.iloc[1, int(i/2)] = pg.ttest(fariness_tab.iloc[:,i], fariness_tab.iloc[:,i+1], correction=True)['p-val']
        else: 
            test_table.iloc[1, int(i/2)] = pg.ttest(fariness_tab.iloc[:,i], fariness_tab.iloc[:,i+1], correction=False)['p-val']
        
        ## ratio
        test_table.iloc[2, int(i/2)] = np.nanmean(fariness_tab.iloc[:,i]/fariness_tab.iloc[:,i+1])
        
    summary_table.columns = colname_14
    test_table.columns = colname_7
    return(summary_table, test_table)