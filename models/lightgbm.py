# imports
from lightgbm import LGBMRegressor
from skforecast.model_selection import grid_search_forecaster, TimeSeriesFold
from skforecast.recursive import ForecasterRecursive
import pandas as pd


class LightGBMModel:
    def __init__(self):
        self.use_log = True

        self.weekly_lags = [1, 2, 3, 4, 8, 12, 26, 52]
        self.monthly_lags = [1, 2, 3, 6, 12]

    def _detect_frequency(self, y):
        freq = pd.infer_freq(y.index)

        if freq is None:
            return "Unknown"

        if "W" in freq:
            return "Weekly"
        elif "M" in freq:
            return "Monthly"
        else:
            return "Unknown"

    def _get_lags(self, y):

        freq = self._detect_frequency(y)

        if freq == "Monthly":
            return self.monthly_lags
        else:
            return self.weekly_lags

    def get_base_model(self, lags):
        return ForecasterRecursive(
            estimator=LGBMRegressor(
                random_state=42
            ),
            lags=lags
        )

    def get_param_grid(self):
        return {
            'n_estimators': [300, 500, 1000],
            'learning_rate': [0.01, 0.03, 0.05, 0.1],
            'min_child_samples': [10, 20, 50],
            'subsample': [0.8, 1.0],
            'colsample_bytree': [0.8, 1.0],
            'reg_alpha': [0, 0.1],
            'reg_lambda': [0, 0.1]
        }

    def tune(self, y_train, X_train):

        n = len(y_train)

        base_lags = self._get_lags(y_train)

        max_lag = max(base_lags)

        if n <= max_lag:
            safe_lags = [l for l in base_lags if l < n // 2]
            if not safe_lags:
                safe_lags = [1, 2, 3]
        else:
            safe_lags = base_lags

        if n < 20:
            steps = max(1, n // 4)
            initial_train_size = max(5, int(n * 0.6))

        elif n < 60:
            steps = max(2, n // 5)
            initial_train_size = int(n * 0.65)

        else:
            steps = 12 if n > 150 else 6
            initial_train_size = int(n * 0.7)

        steps = max(1, steps)
        initial_train_size = min(initial_train_size, n - steps - 1)

        cv = TimeSeriesFold(
            steps=steps,
            initial_train_size=initial_train_size,
            fixed_train_size=False
        )

        forecaster = self.get_base_model(safe_lags)

        try:
            results = grid_search_forecaster(
                forecaster=forecaster,
                y=y_train,
                exog=X_train,
                param_grid=self.get_param_grid(),
                lags_grid=[safe_lags],
                cv=cv,
                metric='mean_absolute_error',
                return_best=True
            )

            if results is None or len(results) == 0:
                return {}, safe_lags

            best = results.iloc[0]

            return best["params"], best["lags"]

        except Exception:
            return {}, safe_lags

    def build(self, best_params, best_lags):

        best_params = best_params or {}

        regressor = LGBMRegressor(
            **best_params,
            random_state=42
        )

        forecaster = ForecasterRecursive(
            estimator=regressor,
            lags=best_lags
        )

        forecaster.estimator = regressor

        forecaster.use_log = self.use_log

        return forecaster