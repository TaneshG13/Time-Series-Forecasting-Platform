import numpy as np
import pandas as pd
import statsmodels.api as sm
import warnings
from sklearn.metrics import mean_squared_error, mean_absolute_percentage_error

from pipelines.common import sarimax_recursive_forecast

warnings.filterwarnings("ignore")


class SARIMAXPipeline:

    def __init__(self, model, freq="Weekly"):
        self.model_obj = model
        self.freq = freq


    def run(self, df, target, future_periods=26):

        if target not in df.columns:
            raise ValueError(f"Target column '{target}' not found in dataframe.")

        if df.isna().sum().sum() > 0:
            raise ValueError("Data contains missing values. Clean data before pipeline.")

        exog_cols = [col for col in df.columns if col != target]
        if not exog_cols:
            raise ValueError("No exogenous feature columns found.")

        split_idx = int(len(df) * 0.9)

        X_full = df[exog_cols]
        y_full = df[target]

        X_train = X_full.iloc[:split_idx]
        X_test  = X_full.iloc[split_idx:]
        y_train = y_full.iloc[:split_idx]
        y_test  = y_full.iloc[split_idx:]

        selected_features = self._ols_feature_selection(X_train, y_train)

        if not selected_features:
            selected_features = exog_cols

        X_train_sel = X_train[selected_features]
        X_test_sel  = X_test[selected_features]

        best_order, seasonal_order = self.model_obj.tune(y_train, X_train_sel)
        config = self.model_obj.build(best_order, seasonal_order)

        fitted_results = self.model_obj.fit(config, y_train, X_train_sel)

        test_forecast_df = sarimax_recursive_forecast(
            results=fitted_results,
            df=df,
            X_base=X_train_sel,
            periods=len(X_test_sel),
            target=target,
            final_features=selected_features,
            freq=self.freq,
        )

        test_preds   = test_forecast_df[target]
        y_test_index = y_test.index

        test_preds.index = y_test_index

        rmse = float(np.sqrt(mean_squared_error(y_test, test_preds)))
        mape = float(mean_absolute_percentage_error(y_test, test_preds))

        X_full_sel = X_full[selected_features]

        fitted_results_full = self.model_obj.fit(config, y_full, X_full_sel)

        future_forecast_df = sarimax_recursive_forecast(
            results=fitted_results_full,
            df=df,
            X_base=X_full_sel,
            periods=future_periods,
            target=target,
            final_features=selected_features,
            freq=self.freq,
        )

        return {
            "Metrics": {
                "rmse": rmse,
                "mape": mape,
            },
            "features": {
                "selected": selected_features,
            },
            "predictions": {
                "test_actual": y_test,
                "test_pred":   test_preds,
                "future":      future_forecast_df,
            },
        }


    def _ols_feature_selection(
        self,
        X_train: pd.DataFrame,
        y_train: pd.Series,
        pvalue_threshold: float = 0.10,
    ):
        X_train_const = sm.add_constant(X_train, has_constant="add")
        ols_model = sm.OLS(y_train, X_train_const).fit()

        pvalues = ols_model.pvalues.drop("const", errors="ignore")
        selected = pvalues[pvalues < pvalue_threshold].index.tolist()

        return selected