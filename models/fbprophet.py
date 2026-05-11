import itertools

import numpy as np
from prophet import Prophet
from sklearn.metrics import mean_absolute_percentage_error


class ProphetModel:

    def __init__(
        self,
        freq="Weekly",
        use_log=True
    ):

        self.freq = freq
        self.use_log = use_log
        self.best_params = None

    def build(
        self,
        params=None
    ):

        params = params or {}

        model = Prophet(
            changepoint_prior_scale=params.get(
                "changepoint_prior_scale",
                0.05
            ),
            seasonality_prior_scale=params.get(
                "seasonality_prior_scale",
                10
            ),
            holidays_prior_scale=params.get(
                "holidays_prior_scale",
                10
            ),
            seasonality_mode=params.get(
                "seasonality_mode",
                "additive"
            ),
            yearly_seasonality=True,
            weekly_seasonality=self.freq == "Weekly",
            daily_seasonality=False
        )

        return model

    def tune(
        self,
        train_df,
        test_df,
        features
    ):

        param_grid = {
            "changepoint_prior_scale": [
                0.001,
                0.01,
                0.05,
                0.1
            ],
            "seasonality_prior_scale": [
                1,
                5,
                10
            ],
            "holidays_prior_scale": [
                1,
                5,
                10
            ],
            "seasonality_mode": [
                "additive",
                "multiplicative"
            ]
        }

        best_mape = np.inf
        best_params = None

        for cps, sps, hps, smode in itertools.product(
            param_grid["changepoint_prior_scale"],
            param_grid["seasonality_prior_scale"],
            param_grid["holidays_prior_scale"],
            param_grid["seasonality_mode"]
        ):

            model = Prophet(
                changepoint_prior_scale=cps,
                seasonality_prior_scale=sps,
                holidays_prior_scale=hps,
                seasonality_mode=smode,
                yearly_seasonality=True,
                weekly_seasonality=self.freq == "Weekly",
                daily_seasonality=False
            )

            for reg in features:
                model.add_regressor(reg)

            model.fit(
                train_df[
                    ['ds', 'y'] + features
                ]
            )

            future_test = test_df[
                ['ds'] + features
            ]

            forecast = model.predict(
                future_test
            )

            mape = mean_absolute_percentage_error(
                test_df['y'],
                forecast['yhat']
            )

            if mape < best_mape:

                best_mape = mape

                best_params = {
                    "changepoint_prior_scale": cps,
                    "seasonality_prior_scale": sps,
                    "holidays_prior_scale": hps,
                    "seasonality_mode": smode
                }

        self.best_params = best_params

        return best_params
