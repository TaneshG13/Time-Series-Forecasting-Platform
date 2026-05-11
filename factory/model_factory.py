from models.random_forest import RandomForestModel
from models.xgboost import XGBoostModel
from models.lightgbm import LightGBMModel
from models.catboost import CatBoostModel
from models.HistGradientBoosting import HistGradientBoostingModel

# New strictly notebook-aligned models
from models.sarimax import SARIMAXModel
from models.varmax import VARMAXModel
from models.fbprophet import ProphetModel
from models.lstm import LSTMModel

# model mapping
def get_model(name):
    mapping = {
        "Random Forest": RandomForestModel,
        "XGBoost": XGBoostModel,
        "LightGBM": LightGBMModel,
        "CatBoost": CatBoostModel,
        "HistGradientBoosting": HistGradientBoostingModel,
        "SARIMAX": SARIMAXModel,
        "VARMAX": VARMAXModel,
        "FBProphet": ProphetModel,
        "LSTM": LSTMModel
    }
    
    if name not in mapping:
        raise ValueError(f"Model '{name}' not found in factory mapping.")
        
    return mapping[name]()