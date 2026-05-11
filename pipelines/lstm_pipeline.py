import numpy as np
import pandas as pd
import statsmodels.api as sm

from sklearn.preprocessing import (
    MinMaxScaler
)

from sklearn.metrics import (
    mean_squared_error,
    mean_absolute_percentage_error
)

from pipelines.common import (
    create_lstm_sequences,
    lstm_recursive_forecast
)


class LSTMPipeline:

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
        features
    ):

        X = train_df[features]
        y = train_df[target]

        X_const = sm.add_constant(
            X,
            has_constant='add'
        )

        model = sm.OLS(
            y,
            X_const
        ).fit()

        pvalues = (
            model.pvalues
            .drop(
                'const',
                errors='ignore'
            )
            .sort_values()
        )

        selected = (
            pvalues[
                pvalues < 0.05
            ]
            .index
            .tolist()
        )

        if len(selected) == 0:

            selected = (
                pvalues
                .head(10)
                .index
                .tolist()
            )

        importance_df = pd.DataFrame({
            "feature": pvalues.index.tolist(),
            "importance": pvalues.values
        })

        return (
            selected,
            importance_df
        )

    def run(
        self,
        df,
        target,
        future_periods=52
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

        trend_cols = [
            col for col in data.columns
            if col.startswith("trend_")
        ]

        if len(trend_cols) == 0:

            trend_cols = [
                col for col in data.columns
                if col != target
            ]

        split = int(
            len(data) * 0.9
        )

        train_df = (
            data
            .iloc[:split]
            .copy()
        )

        (
            selected_features,
            importance_df
        ) = self._select_features_ols(
            train_df,
            target,
            trend_cols
        )

        final_features = (
            selected_features
        )

        sequence_features = list(
            dict.fromkeys(
                [target] + final_features
            )
        )

        X = data[
            sequence_features
        ]

        y = data[target]

        scaler_X = MinMaxScaler()
        scaler_y = MinMaxScaler()

        X_scaled = (
            scaler_X.fit_transform(X)
        )

        y_scaled = (
            scaler_y.fit_transform(
                y.values.reshape(-1, 1)
            )
        )

        X_seq, y_seq = (
            create_lstm_sequences(
                X_scaled,
                y_scaled,
                time_steps=self.model_obj.lookback
            )
        )

        split_seq = int(
            len(X_seq) * 0.9
        )

        X_train = X_seq[:split_seq]
        X_test = X_seq[split_seq:]

        y_train = y_seq[:split_seq]
        y_test = y_seq[split_seq:]

        model = self.model_obj.build(
            n_features=len(
                sequence_features
            )
        )

        model.fit(
            X_train,
            y_train,
            epochs=self.model_obj.epochs,
            batch_size=self.model_obj.batch_size,
            validation_data=(
                X_test,
                y_test
            ),
            verbose=0,
            shuffle=False
        )

        test_preds = []

        current_seq = (
            X_seq[split_seq - 1]
            .copy()
        )

        for i in range(len(X_test)):

            pred = model.predict(
                current_seq.reshape(
                    1,
                    self.model_obj.lookback,
                    len(sequence_features)
                ),
                verbose=0
            )

            pred_scaled = pred[0, 0]

            pred_value = (
                scaler_y
                .inverse_transform(
                    [[pred_scaled]]
                )[0][0]
            )

            test_preds.append(
                pred_value
            )

            current_seq = X_test[i]

        y_test_actual = y.iloc[
            self.model_obj.lookback + split_seq:
        ]

        test_index = y_test_actual.index

        test_pred_series = pd.Series(
            test_preds,
            index=test_index[:len(test_preds)],
            name=target
        )

        y_test_actual = y_test_actual.iloc[
            :len(test_preds)
        ]

        rmse = np.sqrt(
            mean_squared_error(
                y_test_actual,
                test_pred_series
            )
        )

        mape = (
            mean_absolute_percentage_error(
                y_test_actual,
                test_pred_series
            )
        )

        future_df = (
            lstm_recursive_forecast(
                model=model,
                current_seq=X_seq[-1],
                scaler_X=scaler_X,
                scaler_y=scaler_y,
                feature_df=data,
                final_features=sequence_features,
                target=target,
                periods=future_periods,
                lookback=self.model_obj.lookback,
                freq=self.freq
            )
        )

        return {
            "Metrics": {
                "rmse": float(rmse),
                "mape": float(mape)
            },
            "features": {
                "selected": final_features,
                "importance": importance_df
            },
            "predictions": {
                "test_actual": y_test_actual,
                "test_pred": test_pred_series,
                "future": future_df
            }
        }