import numpy as np
import pandas as pd
import statsmodels.api as sm

from sklearn.metrics import (
    mean_squared_error,
    mean_absolute_percentage_error
)

from statsmodels.tsa.statespace.varmax import (
    VARMAX
)

from pipelines.common import (
    varmax_recursive_forecast
)


class VARMAXPipeline:

    def __init__(
        self,
        model,
        freq="Weekly"
    ):

        self.model_obj = model
        self.freq = freq

    def _select_features_ols(
        self,
        train_df,
        target,
        features,
        p_threshold=0.10
    ):

        X = train_df[features]
        y = train_df[target]

        X = sm.add_constant(
            X,
            has_constant='add'
        )

        ols_model = sm.OLS(
            y,
            X
        ).fit()

        pvalues = (
            ols_model.pvalues
            .drop(
                'const',
                errors='ignore'
            )
            .sort_values()
        )

        selected = (
            pvalues[
                pvalues < p_threshold
            ]
            .index
            .tolist()
        )

        if len(selected) == 0:

            selected = (
                pvalues
                .sort_values()
                .head(5)
                .index
                .tolist()
            )

        importance_df = pd.DataFrame({
            "feature": pvalues.index.tolist(),
            "importance": pvalues.values
        })

        importance_df = (
            importance_df
            .sort_values(
                by="importance",
                ascending=True
            )
        )

        return (
            selected,
            importance_df,
            pvalues
        )

    def run(
        self,
        df,
        target,
        future_periods=26
    ):

        data = df.copy()

        data.index = pd.to_datetime(
            data.index
        )

        if self.freq == "Weekly":

            data = data.asfreq(
                'W-SUN'
            )

        else:

            data = data.asfreq(
                'ME'
            )

        split_idx = int(
            len(data) * 0.9
        )

        train_df = (
            data
            .iloc[:split_idx]
            .copy()
        )

        test_df = (
            data
            .iloc[split_idx:]
            .copy()
        )

        if self.model_obj.use_log:

            train_df[target] = np.log1p(
                train_df[target]
            )

            test_df[target] = np.log1p(
                test_df[target]
            )

            data[target] = np.log1p(
                data[target]
            )

        trend_cols = [
            col for col in data.columns
            if col.startswith("trend_")
        ]

        if len(trend_cols) == 0:

            trend_cols = [
                col for col in data.columns
                if col not in [
                    target,
                    "Quarter",
                    "Month",
                    "Week_Number",
                    "contri_week_quarter",
                    "contri_week_month",
                    "contri_month_quarter",
                    "SI_Quarter_week",
                    "SI_Quarter_month"
                ]
            ]

        (
            selected_features,
            importance_df,
            pvalues
        ) = self._select_features_ols(
            train_df,
            target,
            trend_cols,
            p_threshold=0.10
        )

        endog_cols = list(
            dict.fromkeys(
                [target] + selected_features
            )
        )

        if len(endog_cols) < 2:

            fallback_cols = (
                pvalues
                .sort_values()
                .head(2)
                .index
                .tolist()
            )

            endog_cols = list(
                dict.fromkeys(
                    [target] + fallback_cols
                )
            )

        if self.freq == "Weekly":

            exog_cols = [
                'Quarter',
                'contri_week_quarter'
            ]

        else:

            exog_cols = [
                'Quarter',
                'contri_month_quarter'
            ]

        exog_cols = [
            col for col in exog_cols
            if col in data.columns
        ]

        best_order = self.model_obj.tune(
            train_df,
            test_df,
            target,
            endog_cols,
            exog_cols
        )

        model = VARMAX(
            endog=train_df[
                endog_cols
            ],
            exog=train_df[
                exog_cols
            ],
            order=best_order,
            trend='c'
        )

        fitted = model.fit(
            disp=False,
            maxiter=200
        )

        test_forecast = fitted.forecast(
            steps=len(test_df),
            exog=test_df[
                exog_cols
            ]
        )

        test_actual = test_df[target]
        test_pred = test_forecast[target]

        if self.model_obj.use_log:

            test_actual_eval = np.expm1(
                test_actual
            )

            test_pred_eval = np.expm1(
                test_pred
            )

        else:

            test_actual_eval = test_actual
            test_pred_eval = test_pred

        rmse = np.sqrt(
            mean_squared_error(
                test_actual_eval,
                test_pred_eval
            )
        )

        mape = mean_absolute_percentage_error(
            test_actual_eval,
            test_pred_eval
        )

        model_full = VARMAX(
            endog=data[
                endog_cols
            ],
            exog=data[
                exog_cols
            ],
            order=best_order,
            trend='c'
        )

        fitted_full = model_full.fit(
            disp=False,
            maxiter=200
        )

        future_forecast = (
            varmax_recursive_forecast(
                fitted_model=fitted_full,
                history_df=data,
                target=target,
                endog_cols=endog_cols,
                exog_cols=exog_cols,
                periods=future_periods,
                freq=self.freq
            )
        )

        if self.model_obj.use_log:

            test_actual = np.expm1(
                test_actual
            )

            test_pred = np.expm1(
                test_pred
            )

            future_forecast[target] = (
                np.expm1(
                    future_forecast[target]
                )
            )

        test_actual.name = target
        test_pred.name = target

        future_forecast.index = pd.to_datetime(
            future_forecast.index
        )

        return {
            "Metrics": {
                "rmse": float(rmse),
                "mape": float(mape)
            },
            "features": {
                "selected": selected_features,
                "importance": importance_df
            },
            "predictions": {
                "test_actual": test_actual,
                "test_pred": test_pred,
                "future": future_forecast
            }
        }