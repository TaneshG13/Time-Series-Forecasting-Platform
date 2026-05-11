# catboost_model.py
from catboost import CatBoostRegressor
from skforecast.model_selection import grid_search_forecaster, TimeSeriesFold
from skforecast.recursive import ForecasterRecursive
import pandas as pd
import numpy as np

class CatBoostModel:
    def __init__(self):
        self.use_log = True
        # Lags aligned with the project's seasonal requirements
        self.weekly_lags = [1, 2, 3, 4, 8, 12, 26, 52]
        self.monthly_lags = [1, 2, 3, 6, 12]

    def _detect_frequency(self, y):
        """Infers frequency to adjust the cross-validation window."""
        freq = pd.infer_freq(y.index)
        if freq is None:
            if len(y) > 1:
                diff = (y.index[1] - y.index[0]).days
                if diff <= 7: return "Weekly"
                if diff > 25: return "Monthly"
            return "Unknown"
        
        if "W" in freq:
            return "Weekly"
        elif "M" in freq or "MS" in freq:
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
            estimator=CatBoostRegressor(
                random_state=42,
                silent=True  # Keeps logs clean during grid search
            ),
            lags=lags
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
        n = len(y_train)
        freq = self._detect_frequency(y_train)
        
        base_lags = self._get_lags(y_train)
        
        # Ensure lags don't exceed data capacity
        safe_lags = [l for l in base_lags if l < n // 2]
        if not safe_lags:
            safe_lags = [1, 2, 3]

        # Target steps for annual seasonality (52 weeks or 12 months)
        if freq == "Weekly":
            target_steps = 52
        else:
            target_steps = 12

        # 70% initial training size
        initial_train_size = int(n * 0.7)
        
        # Safety Guard: Ensure initial_train + steps < n
        if initial_train_size + target_steps >= n:
            # Scale down validation window if dataset is too small
            steps = max(1, (n - initial_train_size) // 2)
            if steps == 0:
                initial_train_size = int(n * 0.5)
                steps = max(1, n - initial_train_size - 1)
        else:
            steps = target_steps

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

            if results is None or results.empty:
                return {}, safe_lags

            best = results.iloc[0]
            return best["params"], best["lags"]

        except Exception as e:
            print(f"CatBoost Tuning Exception: {e}")
            return {}, safe_lags

    def build(self, best_params, best_lags):
        best_params = best_params or {}
        regressor = CatBoostRegressor(
            **best_params,
            random_state=42,
            silent=True
        )
        forecaster = ForecasterRecursive(
            estimator=regressor,
            lags=best_lags
        )
        forecaster.use_log = self.use_log
        return forecaster