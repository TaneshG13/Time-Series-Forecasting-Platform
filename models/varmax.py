import itertools
import warnings

import numpy as np

from sklearn.metrics import (
    mean_absolute_percentage_error
)

from statsmodels.tsa.statespace.varmax import (
    VARMAX
)

warnings.filterwarnings("ignore")


class VARMAXModel:

    def __init__(
        self,
        freq="Weekly",
        use_log=True
    ):

        self.freq = freq
        self.use_log = use_log
        self.best_order = None

    def tune(
        self,
        train_df,
        test_df,
        target,
        endog_cols,
        exog_cols
    ):

        best_mape = np.inf
        best_order = (1, 0)

        orders = list(
            itertools.product(
                [1, 2],
                [0, 1]
            )
        )

        for order in orders:

            try:

                model = VARMAX(
                    endog=train_df[
                        endog_cols
                    ],
                    exog=train_df[
                        exog_cols
                    ],
                    order=order,
                    trend='c'
                )

                fitted = model.fit(
                    disp=False,
                    maxiter=200
                )

                preds = fitted.forecast(
                    steps=len(test_df),
                    exog=test_df[
                        exog_cols
                    ]
                )

                preds = preds[target]

                mape = (
                    mean_absolute_percentage_error(
                        test_df[target],
                        preds
                    )
                )

                if mape < best_mape:

                    best_mape = mape
                    best_order = order

            except Exception:
                continue

        self.best_order = best_order

        return best_order