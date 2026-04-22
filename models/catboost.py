# imports
from catboost import CatBoostRegressor
from skforecast.model_selection import grid_search_forecaster, TimeSeriesFold
from skforecast.recursive import ForecasterRecursive

class CatBoostModel:
    def __init__(self):
        self.use_log = True
        self.lags = [1, 2, 3, 4, 8, 12, 26, 52]

    def get_base_model(self):
        return ForecasterRecursive(
        regressor=CatBoostRegressor(random_state=42),
        lags=self.lags
    )

    def get_param_grid(self):
        return {
        'iterations': [200, 300, 500],
        'learning_rate': [0.03, 0.05],
        'depth': [3, 4, 6],
        'l2_leaf_reg': [1, 3],
        'subsample': [0.7, 0.8],
        'rsm': [0.8, 1.0],
        'min_data_in_leaf': [20, 50],
        'random_strength': [1, 5],
    }

    def tune(self, y_train, X_train):
        forecaster = self.get_base_model()

        cv = TimeSeriesFold(
            steps=12,
            initial_train_size=int(len(y_train) * 0.7),
            fixed_train_size= False
        )

        results = grid_search_forecaster(
            forecaster= forecaster,
            y = y_train,
            exog= X_train,
            param_grid=self.get_param_grid(),
            lags_grid=[self.lags],
            cv=cv,
            metric='mean_absolute_error',
            return_best=True
        )

        best = results.iloc[0]

        return best['params'], best['lags']
    
    def build(self, best_params, best_lags):
        forecaster = ForecasterRecursive(
            regressor=CatBoostRegressor(**best_params, random_state=42),
            lags=best_lags
        )

        forecaster.use_log = self.use_log

        return forecaster