import numpy as np
import pandas as pd
import warnings
from statsmodels.tsa.statespace.sarimax import SARIMAX

warnings.filterwarnings("ignore")


class SARIMAXModel:

    def __init__(self):
        self.use_log = False

        self.order_grid = [
            (1, 1, 1),
            (1, 1, 0),
            (0, 1, 1),
            (2, 1, 1),
            (1, 1, 2),
            (2, 1, 2),
        ]

        self._seasonal_period = {"Weekly": 52, "Monthly": 12}

    def _detect_frequency(self, y: pd.Series) -> str:
        freq = pd.infer_freq(y.index)
        if freq is None:
            return "Unknown"
        if "W" in freq:
            return "Weekly"
        if "M" in freq:
            return "Monthly"
        return "Unknown"

    def _seasonal_period_for(self, y: pd.Series) -> int:
        freq = self._detect_frequency(y)
        return self._seasonal_period.get(freq, 52)


    def tune(self, y_train: pd.Series, X_train: pd.DataFrame):
        S = self._seasonal_period_for(y_train)
        seasonal_order = (0, 0, 0, S)

        best_order = (1, 1, 1)
        best_mae = np.inf

        for order in self.order_grid:
            try:
                mdl = SARIMAX(
                    y_train,
                    exog=X_train,
                    order=order,
                    seasonal_order=seasonal_order,
                    enforce_stationarity=False,
                    enforce_invertibility=False,
                )
                res = mdl.fit(disp=False)
                mae = np.mean(np.abs(res.resid))
                if mae < best_mae:
                    best_mae = mae
                    best_order = order
            except Exception:
                continue

        return best_order, seasonal_order

    def build(self, best_order, seasonal_order):
        return {
            "order": best_order,
            "seasonal_order": seasonal_order,
            "use_log": self.use_log,
        }

    def fit(self, config: dict, y: pd.Series, X: pd.DataFrame):
        model = SARIMAX(
            y,
            exog=X,
            order=config["order"],
            seasonal_order=config["seasonal_order"],
            enforce_stationarity=False,
            enforce_invertibility=False,
        )
        results = model.fit(disp=False)
        return results