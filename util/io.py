import os
import json
import pickle
from pathlib import Path

def check_model_exist(root_dir, models_dir, model_name, model_id):
    save_dir = os.path.join(root_dir, models_dir, model_name)
    save_path = check_saving_path(save_dir,model_id)
    path = Path(save_path)
    return path.is_file()

def check_saving_path(save_dir,model_id):
    
    isExist = os.path.exists(save_dir)
    if isExist is False: os.makedirs(save_dir)
    save_path = os.path.join(save_dir, model_id)
    return save_path

def save_model(model, root_dir, models_dir, model_name, model_id):
    save_dir = os.path.join(root_dir, models_dir, model_name)
    save_path = check_saving_path(save_dir,model_id)
    pickle.dump(model, open(save_path, 'wb'))
    
def load_model(root_dir, models_dir, model_name, model_id):
    save_dir = os.path.join(root_dir, models_dir, model_name)
    isExist = os.path.exists(save_dir)
    if isExist is False: 
        print("No folder exist!!!")
        return None
    save_path = os.path.join(save_dir, model_id)
    loaded_model = pickle.load(open(save_path, 'rb'))
    return loaded_model

def save_dataframe(dataframe, root_dir, models_dir, model_name, model_id):
    save_dir = os.path.join(root_dir, models_dir, model_name)
    save_path = check_saving_path(save_dir, model_id)
    dataframe.to_csv(save_path)
    
def save_params_as_json(params, save_dir,model_alg, model_name, model_id):
    save_dir = os.path.join(save_dir)
    save_path = check_saving_path(save_dir, model_alg+model_name+model_id+".json")
    with open(save_path, 'w', encoding='utf-8') as f:
        json.dump(params, f, ensure_ascii=False, indent=4)
    
## can be removed
def save_fairness(dataframe, root_dir, output_fariness, model_name, model_id, filename):
    save_dir =  os.path.join(root_dir, output_fariness, model_name, model_id)
    isExist = os.path.exists(save_dir)
    if isExist is False: os.makedirs(save_dir)
    save_path = os.path.join(root_dir, output_fariness, model_name, model_id, filename)
    dataframe.to_csv(save_path)
    