import numpy as np
import xgboost as xgb

from imblearn.pipeline import Pipeline
from imblearn.over_sampling import SMOTE 
from imblearn.over_sampling import RandomOverSampler

from sklearn.decomposition import PCA
from sklearn.impute import SimpleImputer
from sklearn.feature_selection import RFE
from sklearn.preprocessing import MinMaxScaler, MaxAbsScaler
from sklearn.model_selection import GridSearchCV
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.svm import SVC
from sklearn.model_selection import RandomizedSearchCV
from lightgbm import LGBMClassifier
from catboost import CatBoostClassifier

def model_finalize(pm_config, X_val = None, y_val= None):
    if pm_config.model_alg == "xgboost_":
        return xgb_finalize(pm_config, X_val, y_val)
    elif pm_config.model_alg == "xgboost_smote_":
        return xgb_smote_finalize(pm_config, X_val, y_val)
    elif pm_config.model_alg == "lrc_":
        return lr_finalize(pm_config, X_val, y_val)
    elif pm_config.model_alg == "lrc_smote_":
        return lr_smote_finalize(pm_config, X_val, y_val)
    elif pm_config.model_alg == "lasso_":
        return lasso_finalize(pm_config, X_val, y_val)
    elif pm_config.model_alg == "rfc_":
        return rfc_finalize(pm_config, X_val, y_val)
    elif pm_config.model_alg == "rfc_smote_":
        return rfc_smote_finalize(pm_config, X_val, y_val)
    elif pm_config.model_alg == "light_":
        return light_finalize(pm_config, X_val, y_val)
    elif pm_config.model_alg == "cat_":
        return cat_finalize(pm_config, X_val, y_val)
    else:
        return None

def xgb_smote_finalize(pm_config, X_val = None, y_val= None):
    model_name="xgb_smote_"
    #imputation
    simim = SimpleImputer(strategy="mean")
    #preprocessing
    scaler = MinMaxScaler() 
    #imbalanced data processing
    ibp = SMOTE(random_state=pm_config.setting_params.random_state)
    # ibp = RandomOverSampler(random_state=42)
    #base model
    base_model = xgb.XGBClassifier(verbosity = pm_config.setting_params.verbosity, 
                                   random_state = pm_config.setting_params.random_state)
    
    fit_params={
        "base__eval_set" : [[X_val, y_val]],
        "base__eval_metric" : pm_config.setting_params.eval_metric,
        "base__early_stopping_rounds":pm_config.setting_params.early_stopping_rounds}
    
    pipe = Pipeline([
        ("imputation",simim),
        ("scaler",scaler),
        ("ibp", ibp),
        ("base", base_model)
    ])
    
    return pipe, model_name, fit_params
    
def xgb_finalize(pm_config, X_val = None, y_val= None):
    model_name="xgb_"
    #imputation
    simim = SimpleImputer(strategy="mean")
    #preprocessing
    scaler = MinMaxScaler() 
    #imbalanced data processing
    base_model = xgb.XGBClassifier(verbosity = pm_config.setting_params.verbosity, 
                                   random_state = pm_config.setting_params.random_state)
    
    fit_params={
        "base__eval_set" : [[X_val, y_val]],
        "base__eval_metric" : pm_config.setting_params.eval_metric,
        "base__early_stopping_rounds":pm_config.setting_params.early_stopping_rounds}
    
    pipe = Pipeline([
        ("imputation",simim),
        ("scaler",scaler),
        ("base", base_model)
    ])
    
    return pipe, model_name, fit_params

def light_finalize(pm_config, X_val = None, y_val= None):
    model_name="light_FINAL_SMT_"
    #imputation
    simim = SimpleImputer(strategy="mean")
    #preprocessing
    scaler = MinMaxScaler() 
    #imbalanced data processing
    ibp = SMOTE(random_state=pm_config.setting_params.random_state)
    # ibp = RandomOverSampler(random_state=42)
    #base model
    base_model = LGBMClassifier(random_state = pm_config.setting_params.random_state, 
                                   n_jobs = pm_config.setting_params.n_jobs)
    
    fit_params={
        "base__eval_set" : [[X_val, y_val]],
        "base__eval_metric" : pm_config.setting_params.eval_metric,
        "base__early_stopping_rounds":pm_config.setting_params.early_stopping_rounds}
    
    pipe = Pipeline([
        ("imputation",simim),
        ("scaler",scaler),
        ("ibp", ibp),
        ("base", base_model)
    ])
    
    return pipe, model_name, fit_params
    
def cat_finalize(pm_config, X_val = None, y_val= None):
    model_name="cat_FINAL_SMT_"
    #imputation
    simim = SimpleImputer(strategy="mean")
    #preprocessing
    scaler = MinMaxScaler() 
    #imbalanced data processing
    ibp = SMOTE(random_state=pm_config.setting_params.random_state)
    # ibp = RandomOverSampler(random_state=42)
    #base model
    base_model = CatBoostClassifier(
        random_state = pm_config.setting_params.random_state, 
        )
    
    fit_params={
        "base__eval_set" : (X_val, y_val),
        "base__early_stopping_rounds":pm_config.setting_params.early_stopping_rounds}
    
    pipe = Pipeline([
        ("imputation",simim),
        ("scaler",scaler),
        ("ibp", ibp),
        ("base", base_model)
    ])
    
    return pipe, model_name, fit_params
    
def lr_finalize(pm_config, X_val = None, y_val= None):
    model_name="lrc_final_"
    #imputation
    simim = SimpleImputer(strategy="mean")
    #preprocessing
    scaler = MinMaxScaler() 
    #base model
    base_model = LogisticRegression(random_state = pm_config.setting_params.random_state, 
                                   n_jobs = pm_config.setting_params.n_jobs)
    
    fit_params = {}
    
    pipe = Pipeline([
        ("imputation",simim),
        ("scaler",scaler),
        ("base", base_model)
    ])
    
    return pipe, model_name, fit_params
    
def lr_smote_finalize(pm_config, X_val = None, y_val= None):
    model_name="lrc_smote_"
    #imputation
    simim = SimpleImputer(strategy="mean")
    #preprocessing
    scaler = MinMaxScaler() 
    #imbalanced data processing
    ibp = SMOTE(random_state=pm_config.setting_params.random_state)
    #base model
    base_model = LogisticRegression(random_state = pm_config.setting_params.random_state, 
                                   n_jobs = pm_config.setting_params.n_jobs)
    
    fit_params = {}
    
    pipe = Pipeline([
        ("imputation",simim),
        ("scaler",scaler),
        ("ibp", ibp),
        ("base", base_model)
    ])
    
    return pipe, model_name, fit_params
    
def lasso_finalize(pm_config, X_val = None, y_val= None):
    model_name="lasso_final_"
    #imputation
    simim = SimpleImputer(strategy="mean")
    #preprocessing
    scaler = MinMaxScaler() 
    #imbalanced data processing
    ibp = SMOTE(random_state=pm_config.setting_params.random_state)
    # ibp = RandomOverSampler(random_state=42)
    #base model
    base_model = LogisticRegression(
        penalty='l1', 
        random_state = pm_config.setting_params.random_state,
        n_jobs = pm_config.setting_params.n_jobs,
        C = pm_config.C,
        solver = pm_config.solver)
    
    fit_params = {}
    
    pipe = Pipeline([
        ("imputation",simim),
        ("scaler",scaler),
        ("ibp", ibp),
        ("base", base_model)
    ])
    
    return pipe, model_name, fit_params
    
def rfc_finalize(pm_config, X_val = None, y_val= None):
    model_name="rfc_final_"
    #imputation
    simim = SimpleImputer(strategy="mean")
    #preprocessing
    scaler = MinMaxScaler() 
    #imbalanced data processing
    #base model
    base_model = RandomForestClassifier(random_state = pm_config.setting_params.random_state, 
                                   n_jobs = pm_config.setting_params.n_jobs)
    
    fit_params = {}
    
    pipe = Pipeline([
        ("imputation",simim),
        ("scaler",scaler),
        ("base", base_model)
    ])
    
    return pipe, model_name, fit_params 
    
def rfc_smote_finalize(pm_config, X_val = None, y_val= None):
    model_name="rfc_smote_"
    #imputation
    simim = SimpleImputer(strategy="mean")
    #preprocessing
    scaler = MinMaxScaler() 
    #imbalanced data processing
    ibp = SMOTE(random_state=pm_config.setting_params.random_state)
    # ibp = RandomOverSampler(random_state=42)
    #base model
    base_model = RandomForestClassifier(random_state = pm_config.setting_params.random_state, 
                                   n_jobs = pm_config.setting_params.n_jobs)
    
    fit_params = {}
    
    pipe = Pipeline([
        ("imputation",simim),
        ("scaler",scaler),
        ("ibp", ibp),
        ("base", base_model)
    ])
    
    return pipe, model_name, fit_params    
    