# imports
from sklearn.ensemble import HistGradientBoostingRegressor
from skforecast.model_selection import grid_search_forecaster, TimeSeriesFold
from skforecast.recursive import ForecasterRecursive

class HistGradientBoostingModel:
    def __init__(self):
        self.use_log = True
        self.lags = [1, 2, 3, 4, 8, 12, 26, 52]

    def get_base_model(self):
        return ForecasterRecursive(
        regressor=HistGradientBoostingRegressor(random_state=42),
        lags=self.lags
    )

    def get_param_grid(self):
        return {
        'max_iter': [300, 500, 800],
        'learning_rate': [0.01, 0.03, 0.05, 0.1],
        'max_depth': [3, 5, 7, None],
        'min_samples_leaf': [10, 20, 50],
        'l2_regularization': [0, 0.1, 1],
        'max_bins': [255]
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
            regressor=HistGradientBoostingRegressor(**best_params, random_state=42),
            lags=best_lags
        )

        forecaster.use_log = self.use_log

        return forecaster