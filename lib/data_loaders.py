""" """
import os
import pickle
from typing import Literal, Dict, Tuple, List
import pandas as pd


def load_feature_importances(
        gas: Literal['co2', 'ch4'],
        feature_type: Literal['shap', 'breakdown'] = 'shap', 
        model_name: Literal['xgboost', 'lightgbm', 'catboost'] = 'xgboost',
        interaction_preference: int = 0,
        folder: str = "bin/model_explanations_precalculated/dalex"):
    """Loads the feature importance data for co2 and ch4 regression"""
    subfolder_per_type: Dict[str, str] = dict(
        shap='dalex', 
        breakdown = os.path.join('breakdown_interactions', f'interaction_preference_{interaction_preference}'))
    try:
        subfolder = subfolder_per_type[feature_type]
    except KeyError as e:
        raise KeyError(
            f"Feature type {feature_type} not supported, Pick either 'shap' or 'breakdown'.") from e
    
    if feature_type == "shap":
        filename_noext: str = f'shap_{model_name}_{gas}_dalex'
    elif feature_type == "breakdown":
        filename_noext: str = f'breakdown_{model_name}_{gas}_dalex' 
    else:
        print(f"Feature type {feature_type} not recognized")
        
    # Create filename paths
    features_table_path = os.path.join(folder, subfolder, ".".join((filename_noext,'xlsx')))
    features_fullfile_path = os.path.join(folder, subfolder, ".".join((filename_noext,'pkl')))
    
    # Load data
    features_table = pd.read_excel(features_table_path)
    with open(features_fullfile_path, 'rb') as file:
        features_full = pickle.load(file)
                                          
    return features_table, features_full


def load_input_output(filename: str, sheets: Tuple[str, ...]) -> pd.DataFrame:
    """ """
    df_list: List[pd.DataFrame] = []
    for sheet in sheets:
        df_list.append(pd.read_excel(filename, sheet_name = sheet))
    return pd.concat(df_list, axis = 1).T.drop_duplicates().T
