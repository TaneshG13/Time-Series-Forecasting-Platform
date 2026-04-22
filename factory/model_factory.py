from models.random_forest import RandomForestModel
from models.xgboost import XGBoostModel
from models.lightgbm import LightGBMModel
from models.catboost import CatBoostModel
from models.HistGradientBoosting import HistGradientBoostingModel

# model mapping
def get_model(name):
    mapping = {
        "Random Forest": RandomForestModel,
        "XGBoost": XGBoostModel,
        "LightGBM": LightGBMModel,
        "CatBoost": CatBoostModel,
        "HistGradientBoosting": HistGradientBoostingModel
    }
    return mapping[name]()