import pandas as pd
import numpy as np
from sklearn.inspection import permutation_importance

def split_data(df, target, exog_cols):
    y = df[target]
    X = df[exog_cols]
    split = int(len(df) * 0.9)
    return (
        X.iloc[:split], X.iloc[split:],
        y.iloc[:split], y.iloc[split:]
    )

def get_top_features(model, top_pct=0.4, X_train=None, y_train=None):
    try:
        estimator = None
        if hasattr(model, "estimator"):
            estimator = model.estimator
        elif hasattr(model, "regressor"):
            estimator = model.regressor
        else:
            raise ValueError("Model has no estimator/regressor")

        X_transformed, y_transformed = model.create_train_X_y(
            y=y_train,
            exog=X_train
        )

        perm = permutation_importance(
            estimator=estimator,
            X=X_transformed,
            y=y_transformed,
            n_repeats=10,
            random_state=42,
            n_jobs=-1
        )

        importance = pd.DataFrame({
            "feature": X_transformed.columns,
            "importance": perm.importances_mean
        }).sort_values(by="importance", ascending=False)

    except Exception:
        importance = pd.DataFrame({
            "feature": X_train.columns,
            "importance": np.random.rand(len(X_train.columns))
        }).sort_values(by="importance", ascending=False)

    n_features = len(importance)
    top_n = 20
    top_features = importance["feature"].head(top_n).tolist()

    selected_exog = [
        f for f in top_features
        if not f.startswith("lag_")
    ]

    return selected_exog, importance

def ml_recursive_forecast(
    model,
    df,
    X_base,
    y_base,
    periods,
    target,
    freq="Weekly"
):

    if freq == "Weekly":
        step = pd.Timedelta(weeks=1)
        date_freq = "W-SUN"
    elif freq == "Monthly":
        step = pd.DateOffset(months=1)
        date_freq = "ME"
    else:
        raise ValueError("Unsupported frequency")

    last_date = X_base.index[-1]

    future_dates = pd.date_range(
        start=last_date + step,
        periods=periods,
        freq=date_freq
    )

    last_window = y_base.iloc[-model.window_size:].copy()
    last_window = last_window.asfreq(date_freq)

    if freq == "Weekly":
        engineered_cols = [
            'Quarter', 'Week_Number', 'Month', 'Year',
            'dayofyear', 'weekday',
            'is_month_start', 'is_month_end',
            'SMA_3', 'SMA_4', 'SMA_5',
            'contri_week_month', 'contri_week_quarter',
            'SI_Quarter_week',
        ]
    else:
        engineered_cols = [
            'Quarter', 'Month', 'Year',
            'SMA_3', 'SMA_4', 'SMA_5',
            'contri_month_quarter',
            'SI_Quarter_month',
        ]

    search_cols = [col for col in X_base.columns if col not in engineered_cols]

    search_hist = {
        col: X_base[col].iloc[-4:].tolist() for col in search_cols
    }

    if model.use_log:
        target_hist = np.expm1(y_base.iloc[-5:]).tolist()
    else:
        target_hist = y_base.iloc[-5:].tolist()


    weekly_contrib_month_avg    = {}
    weekly_contrib_quarter_avg  = {}

    if freq == "Weekly":
        if "contri_week_month" in df.columns:
            weekly_contrib_month_avg = (
                df.groupby(['Month', 'Week_Number'])['contri_week_month']
                .mean().to_dict()
            )
        if "contri_week_quarter" in df.columns:
            weekly_contrib_quarter_avg = (
                df.groupby(['Quarter', 'Week_Number'])['contri_week_quarter']
                .mean().to_dict()
            )

    monthly_contrib_quarter_avg = {}

    if freq == "Monthly":
        if "contri_month_quarter" in df.columns:
            monthly_contrib_quarter_avg = (
                df.groupby(['Quarter', 'Month'])['contri_month_quarter']
                .mean().to_dict()
            )

    records = []

    for next_dt in future_dates:
        rec = {}

        rec['Month']   = next_dt.month
        rec['Quarter'] = (next_dt.month - 1) // 3 + 1
        rec['Year']    = next_dt.year

        if freq == "Weekly":
            rec['Week_Number']    = next_dt.isocalendar().week
            rec['dayofyear']      = next_dt.timetuple().tm_yday
            rec['weekday']        = next_dt.weekday()
            rec['is_month_start'] = int(next_dt.is_month_start)
            rec['is_month_end']   = int(next_dt.is_month_end)

        rec['SMA_3'] = np.mean(target_hist[-3:])
        rec['SMA_4'] = np.mean(target_hist[-4:])
        rec['SMA_5'] = np.mean(target_hist[-5:])

        if freq == "Weekly":
            key_m = (rec['Month'],   rec['Week_Number'])
            key_q = (rec['Quarter'], rec['Week_Number'])

            if weekly_contrib_month_avg:
                rec['contri_week_month'] = weekly_contrib_month_avg.get(
                    key_m, np.mean(list(weekly_contrib_month_avg.values()))
                )
            if weekly_contrib_quarter_avg:
                rec['contri_week_quarter'] = weekly_contrib_quarter_avg.get(
                    key_q, np.mean(list(weekly_contrib_quarter_avg.values()))
                )

            if "SI_Quarter_week" in df.columns:
                seasonal_match = df[df['Week_Number'] == rec['Week_Number']]['SI_Quarter_week']
                rec['SI_Quarter_week'] = (
                    seasonal_match.mean()
                    if not seasonal_match.empty
                    else df['SI_Quarter_week'].mean()
                )

        else:  
            key_q = (rec['Quarter'], rec['Month'])

            if monthly_contrib_quarter_avg:
                rec['contri_month_quarter'] = monthly_contrib_quarter_avg.get(
                    key_q, np.mean(list(monthly_contrib_quarter_avg.values()))
                )

            if "SI_Quarter_month" in df.columns:
                seasonal_match = df[df['Month'] == rec['Month']]['SI_Quarter_month']
                rec['SI_Quarter_month'] = (
                    seasonal_match.mean()
                    if not seasonal_match.empty
                    else df['SI_Quarter_month'].mean()
                )

        for col in search_cols:
            rec[col] = np.mean(search_hist[col])

        X_curr = pd.DataFrame([rec], index=[next_dt])
        X_curr = X_curr[X_base.columns]

        pred = model.predict(
            steps=1,
            last_window=last_window,
            exog=X_curr
        )

        raw_pred = float(pred.iloc[0])
        val = np.expm1(raw_pred) if model.use_log else raw_pred

        rec[target] = val
        records.append(rec)

        next_index = last_window.index[-1] + step
        new_row = pd.Series(
            [np.log1p(val) if model.use_log else val],
            index=[next_index]
        )
        last_window = pd.concat([last_window.iloc[1:], new_row])
        last_window = last_window.asfreq(date_freq)

        target_hist.append(val)
        target_hist.pop(0)

        for col in search_cols:
            search_hist[col].append(rec[col])
            search_hist[col].pop(0)

    return pd.DataFrame(records, index=future_dates)

def sarimax_recursive_forecast(
    results,
    df,
    X_base,
    periods,
    target,
    final_features,
    freq="Weekly",
):
    
    if freq == "Weekly":
        step = pd.Timedelta(weeks=1)
        date_freq = "W-SUN"
        pd_freq_label = "W"
    elif freq == "Monthly":
        step = pd.DateOffset(months=1)
        date_freq = "ME"
        pd_freq_label = "ME"
    else:
        raise ValueError(f"Unsupported frequency: {freq}. Use 'Weekly' or 'Monthly'.")

    last_date = X_base.index[-1]
    future_dates = pd.date_range(
        start=last_date + step,
        periods=periods,
        freq=date_freq,
    )

    if freq == "Weekly":
        engineered_cols = {
            "Quarter", "Week_Number", "Month", "Year",
            "dayofyear", "weekday",
            "is_month_start", "is_month_end",
            "SMA_3", "SMA_4", "SMA_5",
            "contri_week_month", "contri_week_quarter",
            "SI_Quarter_week",
        }
    else:  
        engineered_cols = {
            "Quarter", "Month", "Year",
            "SMA_3", "SMA_4", "SMA_5",
            "contri_month_quarter",
            "SI_Quarter_month",
        }

    search_cols = [c for c in final_features if c not in engineered_cols]

    search_hist = {
        col: X_base[col].iloc[-4:].tolist()
        for col in search_cols
        if col in X_base.columns
    }

    weekly_contrib_month_avg = {}
    weekly_contrib_quarter_avg = {}

    contri_hist = []

    if freq == "Weekly":
        if "contri_week_month" in df.columns:
            weekly_contrib_month_avg = (
                df.groupby(["Month", "Week_Number"])["contri_week_month"]
                .mean()
                .to_dict()
            )

        if "contri_week_quarter" in df.columns:
            weekly_contrib_quarter_avg = (
                df.groupby(["Quarter", "Week_Number"])["contri_week_quarter"]
                .mean()
                .to_dict()
            )
            contri_hist = df["contri_week_quarter"].iloc[-12:].tolist()

    monthly_contrib_quarter_avg = {}

    if freq == "Monthly":
        if "contri_month_quarter" in df.columns:
            monthly_contrib_quarter_avg = (
                df.groupby(["Quarter", "Month"])["contri_month_quarter"]
                .mean()
                .to_dict()
            )
            contri_hist = df["contri_month_quarter"].iloc[-12:].tolist()

    records = []

    for next_dt in future_dates:
        rec = {}

        rec["Month"]   = next_dt.month
        rec["Quarter"] = (next_dt.month - 1) // 3 + 1
        rec["Year"]    = next_dt.year

        if freq == "Weekly":
            wk = next_dt.isocalendar().week
            rec["Week_Number"]    = wk
            rec["dayofyear"]      = next_dt.timetuple().tm_yday
            rec["weekday"]        = next_dt.weekday()
            rec["is_month_start"] = int(next_dt.is_month_start)
            rec["is_month_end"]   = int(next_dt.is_month_end)

        for col in search_cols:
            if col in search_hist:
                rec[col] = np.mean(search_hist[col])

        if freq == "Weekly":
            if weekly_contrib_quarter_avg:
                key_q = (rec["Quarter"], wk)
                rec["contri_week_quarter"] = weekly_contrib_quarter_avg.get(
                    key_q,
                    np.mean(contri_hist) if contri_hist else 0.0,
                )

            if weekly_contrib_month_avg and "contri_week_month" in final_features:
                key_m = (rec["Month"], wk)
                rec["contri_week_month"] = weekly_contrib_month_avg.get(
                    key_m,
                    np.mean(list(weekly_contrib_month_avg.values())),
                )

            if "SI_Quarter_week" in final_features and "SI_Quarter_week" in df.columns:
                seasonal_match = df[df["Week_Number"] == wk]["SI_Quarter_week"]
                rec["SI_Quarter_week"] = (
                    seasonal_match.mean()
                    if not seasonal_match.empty
                    else df["SI_Quarter_week"].mean()
                )

        else:
            if monthly_contrib_quarter_avg:
                key_q = (rec["Quarter"], rec["Month"])
                rec["contri_month_quarter"] = monthly_contrib_quarter_avg.get(
                    key_q,
                    np.mean(contri_hist) if contri_hist else 0.0,
                )

            if "SI_Quarter_month" in final_features and "SI_Quarter_month" in df.columns:
                seasonal_match = df[df["Month"] == rec["Month"]]["SI_Quarter_month"]
                rec["SI_Quarter_month"] = (
                    seasonal_match.mean()
                    if not seasonal_match.empty
                    else df["SI_Quarter_month"].mean()
                )

        for sma_col in ["SMA_3", "SMA_4", "SMA_5"]:
            if sma_col in final_features and sma_col not in rec:
                if sma_col in X_base.columns:
                    rec[sma_col] = float(X_base[sma_col].iloc[-1])

        X_curr = pd.DataFrame([rec])
        X_curr = X_curr[final_features]
        X_curr.index = pd.DatetimeIndex([next_dt], freq=pd_freq_label)

        pred = results.forecast(steps=1, exog=X_curr)
        pred_value = max(0, float(pred.iloc[0]))

        results = results.append(
            endog=[pred_value],
            exog=X_curr,
            refit=False,
        )

        rec[target] = pred_value
        records.append(rec)

        if freq == "Weekly" and weekly_contrib_quarter_avg:
            contri_hist.append(rec.get("contri_week_quarter", 0.0))
            contri_hist.pop(0)
        elif freq == "Monthly" and monthly_contrib_quarter_avg:
            contri_hist.append(rec.get("contri_month_quarter", 0.0))
            contri_hist.pop(0)

        for col in search_cols:
            if col in search_hist:
                search_hist[col].append(rec.get(col, 0.0))
                search_hist[col].pop(0)
    forecast_df = pd.DataFrame(records, index=future_dates)
    return forecast_df

def varmax_recursive_forecast(
    fitted_model,
    history_df,
    target,
    endog_cols,
    exog_cols,
    periods,
    freq="Weekly"
):

    import numpy as np
    import pandas as pd

    history = history_df.copy()

    future_rows = []

    if freq == "Weekly":

        future_dates = pd.date_range(
            start=history.index.max() + pd.Timedelta(weeks=1),
            periods=periods,
            freq='W-SUN'
        )

    else:

        future_dates = pd.date_range(
            start=history.index.max() + pd.offsets.MonthEnd(1),
            periods=periods,
            freq='ME'
        )

    for next_date in future_dates:

        row = {}

        if freq == "Weekly":

            quarter = (
                (next_date.month - 1) // 3
            ) + 1

            week_number = (
                next_date.isocalendar().week
            )

            similar = history[
                (
                    history["Quarter"]
                    == quarter
                )
                &
                (
                    history["Week_Number"]
                    == week_number
                )
            ]

        else:

            quarter = (
                (next_date.month - 1) // 3
            ) + 1

            month = next_date.month

            similar = history[
                (
                    history["Quarter"]
                    == quarter
                )
                &
                (
                    history["Month"]
                    == month
                )
            ]

        for col in exog_cols:

            if (
                col in similar.columns
                and not similar.empty
            ):

                row[col] = (
                    similar[col].mean()
                )

            else:

                row[col] = (
                    history[col]
                    .iloc[-4:]
                    .mean()
                )

        exog_future = pd.DataFrame(
            [row]
        )

        pred = fitted_model.forecast(
            steps=1,
            exog=exog_future
        )

        pred_value = max(
            0,
            float(pred.iloc[0][target])
        )

        row[target] = pred_value

        for col in endog_cols:

            if col == target:
                continue

            if (
                col in similar.columns
                and not similar.empty
            ):

                row[col] = (
                    similar[col].mean()
                )

            else:

                row[col] = (
                    history[col]
                    .iloc[-4:]
                    .mean()
                )

        row_df = pd.DataFrame(
            [row],
            index=[next_date]
        )

        future_rows.append(
            row_df
        )

        history = pd.concat(
            [
                history,
                row_df
            ]
        )

    future_df = pd.concat(
        future_rows
    )

    future_df.index = pd.to_datetime(
        future_df.index
    )

    return future_df

def create_lstm_sequences(
    X,
    y,
    time_steps=8
):

    import numpy as np

    Xs = []
    ys = []

    for i in range(
        len(X) - time_steps
    ):

        Xs.append(
            X[i:i+time_steps]
        )

        ys.append(
            y[i+time_steps]
        )

    return (
        np.array(Xs),
        np.array(ys)
    )


def lstm_recursive_forecast(
    model,
    current_seq,
    scaler_X,
    scaler_y,
    feature_df,
    final_features,
    target,
    periods,
    lookback,
    freq="Weekly"
):

    import numpy as np
    import pandas as pd

    future_records = []

    seq = current_seq.copy()

    last_date = feature_df.index[-1]

    pred_scaled_history = []

    target_idx = final_features.index(target)

    for step in range(periods):

        pred_scaled = model.predict(
            seq.reshape(
                1,
                lookback,
                len(final_features)
            ),
            verbose=0
        )[0, 0]

        pred_scaled_history.append(
            pred_scaled
        )

        pred_value = (
            scaler_y
            .inverse_transform(
                [[pred_scaled]]
            )[0][0]
        )

        pred_value = max(
            0,
            pred_value
        )

        if freq == "Weekly":

            next_date = (
                last_date
                + pd.Timedelta(
                    weeks=step + 1
                )
            )

            seasonal_lag = 4

        else:

            next_date = (
                last_date
                + pd.offsets.MonthEnd(
                    step + 1
                )
            )

            seasonal_lag = 3

        future_records.append(
            pd.DataFrame(
                {
                    target: [pred_value]
                },
                index=[next_date]
            )
        )

        next_row = seq[-1].copy()

        if len(pred_scaled_history) >= 2:

            momentum = (
                pred_scaled_history[-1]
                -
                pred_scaled_history[-2]
            )

        else:

            momentum = 0

        seasonal_reference = (
            seq[
                -min(
                    seasonal_lag,
                    lookback
                ),
                target_idx
            ]
        )

        next_target = (
            0.55 * pred_scaled
            +
            0.30 * seasonal_reference
            +
            0.15 * (
                pred_scaled + momentum
            )
        )

        next_row[target_idx] = np.clip(
            next_target,
            0,
            1
        )

        for idx, col in enumerate(final_features):

            if col == target:
                continue

            hist_mean = np.mean(
                seq[:, idx]
            )

            hist_std = np.std(
                seq[:, idx]
            )

            drift = np.random.normal(
                0,
                max(
                    hist_std * 0.15,
                    0.01
                )
            )

            next_feature = (
                0.8 * seq[-1][idx]
                +
                0.2 * hist_mean
                +
                drift
            )

            next_row[idx] = np.clip(
                next_feature,
                0,
                1
            )

        seq = np.vstack([
            seq[1:],
            next_row
        ])

    future_df = pd.concat(
        future_records
    )

    future_df.index = pd.to_datetime(
        future_df.index
    )

    return future_df


def fbprophet_recursive_forecast(
    model,
    train_df,
    base_df,
    periods,
    features,
    freq="Weekly",
    future_exog=None
):

    import numpy as np
    import pandas as pd

    df = base_df.copy()

    df['ds'] = pd.to_datetime(
        df['ds']
    )

    if freq == "Weekly":

        future_dates = pd.date_range(
            start=df['ds'].max() + pd.Timedelta(weeks=1),
            periods=periods,
            freq='W-SUN'
        )

    else:

        future_dates = pd.date_range(
            start=df['ds'].max() + pd.offsets.MonthEnd(1),
            periods=periods,
            freq='ME'
        )

    forecast_rows = []

    history = df.copy()

    for i, next_date in enumerate(
        future_dates
    ):

        row = {
            "ds": next_date
        }

        if future_exog is not None:

            exog_row = future_exog.iloc[i]

            for col in features:
                row[col] = exog_row[col]

        else:

            if freq == "Weekly":

                quarter = (
                    (next_date.month - 1) // 3
                ) + 1

                week_number = (
                    next_date.isocalendar().week
                )

                similar = history[
                    (
                        history["Quarter"]
                        == quarter
                    )
                    &
                    (
                        history["Week_Number"]
                        == week_number
                    )
                ]

            else:

                quarter = (
                    (next_date.month - 1) // 3
                ) + 1

                month = next_date.month

                similar = history[
                    (
                        history["Quarter"]
                        == quarter
                    )
                    &
                    (
                        history["Month"]
                        == month
                    )
                ]

            for col in features:

                if (
                    col in similar.columns
                    and not similar.empty
                ):

                    row[col] = (
                        similar[col].mean()
                    )

                else:

                    row[col] = (
                        history[col]
                        .iloc[-4:]
                        .mean()
                    )

        future_input = pd.DataFrame(
            [row]
        )

        pred = model.predict(
            future_input
        )

        pred_value = max(
            0,
            float(pred['yhat'].iloc[0])
        )

        row['y'] = pred_value

        forecast_rows.append(
            row
        )

        history = pd.concat(
            [
                history,
                pd.DataFrame([row])
            ],
            ignore_index=True
        )

    forecast_df = pd.DataFrame(
        forecast_rows
    )

    forecast_df['ds'] = pd.to_datetime(
        forecast_df['ds']
    )

    forecast_df = forecast_df.set_index(
        'ds'
    )

    return forecast_df