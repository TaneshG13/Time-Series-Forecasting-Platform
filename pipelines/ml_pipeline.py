import numpy as np
from sklearn.metrics import mean_absolute_percentage_error, mean_squared_error

from pipelines.common import (
    split_data,
    recursive_forecast,
    get_top_features
)

class MLPipeline:
    def __init__(self, model, freq="Weekly"):
        self.model_obj = model
        self.freq = freq

    def run(self, df, target, top_n=20, future_periods=52):

        exog_cols = [col for col in df.columns if col != target]

        X_train, X_test, y_train, y_test = split_data(df, target, exog_cols)


        if self.model_obj.use_log:
            y_train_transformed = np.log1p(y_train)
        else:
            y_train_transformed = y_train.copy()

        best_params, best_lags = self.model_obj.tune(
            y_train_transformed,
            X_train
        )

        model = self.model_obj.build(best_params, best_lags)

        model.fit(
            y=y_train_transformed,
            exog=X_train
        )

        selected_exog, importance = get_top_features(
            model,
            top_n,
            X_train=X_train,
            y_train=y_train_transformed
        )

        X_selected = df[selected_exog]

        split_idx = int(len(df) * 0.9)

        X_train_sel = X_selected.iloc[:split_idx]
        X_test_sel = X_selected.iloc[split_idx:]

        if self.model_obj.use_log:
            y_full = np.log1p(df[target])
        else:
            y_full = df[target]

        y_train_sel = y_full.iloc[:split_idx]

        model.fit(
            y=y_train_sel,
            exog=X_train_sel
        )

        test_forecast = recursive_forecast(
            model=model,
            df=df,
            X_base=X_train_sel,
            y_base=y_train_sel,
            periods=len(X_test_sel),
            target=target,
            freq=self.freq
        )

        test_preds = test_forecast[target]
        y_test_actual = df[target].iloc[split_idx:]

        rmse = np.sqrt(mean_squared_error(y_test_actual, test_preds))
        mape = mean_absolute_percentage_error(y_test_actual, test_preds)

        future_forecast = recursive_forecast(
            model=model,
            df=df,
            X_base=X_selected,
            y_base=y_full,
            periods=future_periods,
            target=target,
            freq=self.freq
        )

        return {
            "Metrics": {
                "rmse": rmse,
                "mape": mape
            },
            "features": {
                "importance": importance,
                "selected": selected_exog
            },
            "predictions": {
                "test_actual": y_test_actual,
                "test_pred": test_preds,
                "future": future_forecast
            }
        }