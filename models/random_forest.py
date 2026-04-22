# imports
from sklearn.ensemble import RandomForestRegressor
from skforecast.model_selection import grid_search_forecaster, TimeSeriesFold
from skforecast.recursive import ForecasterRecursive

class RandomForestModel:
    def __init__(self):
        self.use_log = False
        self.lags = [1, 2, 3, 4, 8, 12, 26, 52]

    def get_base_model(self):
        return ForecasterRecursive(
        regressor=RandomForestRegressor(random_state=42),
        lags=self.lags
    )

    def get_param_grid(self):
        return {
        'n_estimators': [300, 500, 800],
        'max_depth': [5, 10, 20, None],
        'min_samples_split': [2, 5, 10],
        'min_samples_leaf': [1, 2, 5, 10],
        'max_features': ['sqrt', 'log2', 0.8],
        'bootstrap': [True]
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
            regressor=RandomForestRegressor(**best_params, random_state=42),
            lags=best_lags
        )

        forecaster.use_log = self.use_log

        return forecaster