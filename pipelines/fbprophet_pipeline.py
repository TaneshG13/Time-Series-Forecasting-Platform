import numpy as np
import pandas as pd
import statsmodels.api as sm

from sklearn.metrics import (
    mean_squared_error,
    mean_absolute_percentage_error
)

from pipelines.common import (
    fbprophet_recursive_forecast
)


class FBProphetPipeline:

    def __init__(
        self,
        model,
        freq="Weekly"
    ):

        self.model_obj = model
        self.freq = freq

    def _prepare_dataframe(
        self,
        df,
        target
    ):

        prophet_df = df.copy()

        if 'ds' not in prophet_df.columns:

            if isinstance(
                prophet_df.index,
                pd.DatetimeIndex
            ):

                prophet_df = (
                    prophet_df
                    .reset_index()
                )

                prophet_df = (
                    prophet_df.rename(
                        columns={
                            prophet_df.columns[0]: 'ds'
                        }
                    )
                )

            else:

                raise ValueError(
                    "DataFrame must contain datetime index or ds column"
                )

        prophet_df['ds'] = pd.to_datetime(
            prophet_df['ds']
        )

        if self.freq == "Weekly":

            prophet_df = (
                prophet_df
                .set_index('ds')
                .asfreq('W-SUN')
                .reset_index()
            )

        else:

            prophet_df = (
                prophet_df
                .set_index('ds')
                .asfreq('ME')
                .reset_index()
            )

        prophet_df = prophet_df.rename(
            columns={
                target: 'y'
            }
        )

        return prophet_df

    def _select_features_ols(
        self,
        train_df,
        features
    ):

        X = train_df[features]
        y = train_df['y']

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
        )

        selected = (
            pvalues[
                pvalues < 0.10
            ]
            .index
            .tolist()
        )

        if len(selected) == 0:
            selected = features

        return selected

    def run(
        self,
        df,
        target,
        future_periods=26
    ):

        prophet_df = self._prepare_dataframe(
            df,
            target
        )

        split_idx = int(
            len(prophet_df) * 0.9
        )

        train_df = (
            prophet_df
            .iloc[:split_idx]
            .copy()
        )

        test_df = (
            prophet_df
            .iloc[split_idx:]
            .copy()
        )

        if self.model_obj.use_log:

            train_df['y'] = np.log1p(
                train_df['y']
            )

            test_df['y'] = np.log1p(
                test_df['y']
            )

            prophet_df['y'] = np.log1p(
                prophet_df['y']
            )

        features = [
            col for col in prophet_df.columns
            if col not in ['ds', 'y']
        ]

        selected_features = (
            self._select_features_ols(
                train_df,
                features
            )
        )

        best_params = self.model_obj.tune(
            train_df,
            test_df,
            selected_features
        )

        model = self.model_obj.build(
            best_params
        )

        for reg in selected_features:
            model.add_regressor(reg)

        model.fit(
            train_df[
                ['ds', 'y'] + selected_features
            ]
        )

        test_forecast = (
            fbprophet_recursive_forecast(
                model=model,
                train_df=train_df,
                base_df=prophet_df,
                periods=len(test_df),
                features=selected_features,
                freq=self.freq,
                future_exog=test_df[
                    ['ds'] + selected_features
                ]
            )
        )

        test_actual = test_df['y']
        test_pred = test_forecast['y']

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

        model_full = self.model_obj.build(
            best_params
        )

        for reg in selected_features:
            model_full.add_regressor(reg)

        model_full.fit(
            prophet_df[
                ['ds', 'y'] + selected_features
            ]
        )

        future_forecast = (
            fbprophet_recursive_forecast(
                model=model_full,
                train_df=prophet_df,
                base_df=prophet_df,
                periods=future_periods,
                features=selected_features,
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

            future_forecast['y'] = np.expm1(
                future_forecast['y']
            )

        future_forecast[target] = (
            future_forecast['y']
        )

        future_forecast.index = pd.to_datetime(
            future_forecast.index
        )

        test_actual.name = target
        test_pred.name = target

        test_actual.index = test_df['ds']
        test_pred.index = test_df['ds']

        importance_df = pd.DataFrame({
            "feature": selected_features,
            "importance": np.arange(
                len(selected_features),
                0,
                -1
            )
        })

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