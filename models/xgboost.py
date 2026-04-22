import xgboost as xgb
from skforecast.model_selection import grid_search_forecaster, TimeSeriesFold
from skforecast.recursive import ForecasterRecursive

class XGBoostModel:
    def __init__(self):
        self.use_log = True
        self.lags = [1, 2, 3, 4, 8, 12, 26, 52]

    def get_base_model(self):
        return ForecasterRecursive(
        regressor=xgb.XGBRegressor(random_state=42,objective='reg:squarederror',verbosity=0),
        lags=self.lags
    )

    def get_param_grid(self):
        return {
        'n_estimators': [300, 500, 1000],
        'learning_rate': [0.01, 0.03, 0.05, 0.1],
        'max_depth': [3, 5, 7],
        'min_child_weight': [1, 5, 10],
        'subsample': [0.8, 1.0],
        'colsample_bytree': [0.8, 1.0],
        'gamma': [0, 0.1],
        'reg_alpha': [0, 0.1],
        'reg_lambda': [0, 0.1]
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
            regressor=xgb.XGBRegressor(**best_params, random_state=42),
            lags=best_lags
        )

        forecaster.use_log = self.use_log

        return forecaster