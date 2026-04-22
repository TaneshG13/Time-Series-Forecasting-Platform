import pandas as pd
import numpy as np
from sklearn.inspection import permutation_importance

# load preview
def load_preview(file, n_rows=100):
    file.seek(0)

    if file.name.endswith('.csv'):
        df = pd.read_csv(file, nrows=n_rows)
    else:
        df = pd.read_excel(file, nrows=n_rows)

    return df

# load data
def load_data(file, date_col, freq="Weekly"):
    file.seek(0)

    if file.name.endswith('.csv'):
        df = pd.read_csv(file)
    else:
        df = pd.read_excel(file)

    df[date_col] = pd.to_datetime(df[date_col])

    df = df.set_index(date_col)
    df = df.sort_index()

    if freq == "Weekly":
        df = df.asfreq('W-SUN')
    elif freq == "Monthly":
        df = df.asfreq('M')
    else:
        raise ValueError("Unsupported frequency")

    return df

# split data
def split_data(df, target, exog_cols):
    y = df[target]
    X = df[exog_cols]

    split = int(len(df) * 0.9)

    return (
        X.iloc[:split], X.iloc[split:],
        y.iloc[:split], y.iloc[split:]
    )

# Feture selection
def get_top_features(model, top_n=20, X_train=None, y_train=None):

    importance = None

    if hasattr(model, "get_feature_importances"):
        try:
            importance = model.get_feature_importances()
        except:
            importance = None

    if importance is None or len(importance) == 0:
        try:
            X_transformed, y_transformed = model.create_train_X_y(
                y=y_train,
                exog=X_train
            )

            perm = permutation_importance(
                estimator=model.regressor,
                X=X_transformed,
                y=y_transformed,
                n_repeats=10,
                random_state=42,
                n_jobs=-1
            )

            importance = pd.DataFrame({
                'feature': X_transformed.columns,
                'importance': perm.importances_mean
            }).sort_values(by='importance', ascending=False)

        except Exception as e:
            raise ValueError(f"Feature importance failed: {e}")

    if importance is None or 'feature' not in importance.columns:
        raise ValueError("Feature importance computation failed")

    top_features = importance['feature'].head(top_n).tolist()

    blacklist = ['contri_week_month']

    selected_exog = [
        f for f in top_features
        if not f.startswith("lag_") and f not in blacklist
    ]

    return selected_exog, importance

# recursive forecast
def recursive_forecast(
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
        date_freq = "M"
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

    engineered_cols = [
        'Quarter', 'Week_Number', 'Month', 'Year',
        'dayofyear', 'weekday',
        'is_month_start', 'is_month_end',
        'SMA_3', 'SMA_4', 'SMA_5',
        'contri_week_month', 'contri_week_quarter',
        'SI_Quarter_week'
    ]

    search_cols = [col for col in X_base.columns if col not in engineered_cols]

    search_hist = {
        col: X_base[col].iloc[-4:].tolist() for col in search_cols
    }

    if model.use_log:
        target_hist = np.expm1(y_base.iloc[-5:]).tolist()
    else:
        target_hist = y_base.iloc[-5:].tolist()

    weekly_contrib_month_avg = {}
    weekly_contrib_quarter_avg = {}

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

    records = []

    for next_dt in future_dates:
        rec = {}

        if freq == "Weekly":
            rec['Week_Number'] = next_dt.isocalendar().week
        else:
            rec['Week_Number'] = 1

        rec['Month'] = next_dt.month
        rec['Quarter'] = (next_dt.month - 1)//3 + 1
        rec['Year'] = next_dt.year
        rec['dayofyear'] = next_dt.timetuple().tm_yday
        rec['weekday'] = next_dt.weekday()
        rec['is_month_start'] = int(next_dt.is_month_start)
        rec['is_month_end'] = int(next_dt.is_month_end)

        rec['SMA_3'] = np.mean(target_hist[-3:])
        rec['SMA_4'] = np.mean(target_hist[-4:])
        rec['SMA_5'] = np.mean(target_hist[-5:])

        key_m = (rec['Month'], rec['Week_Number'])
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

        if model.use_log:
            val = np.expm1(raw_pred)
        else:
            val = raw_pred

        rec[target] = val
        records.append(rec)

        next_index = last_window.index[-1] + step

        if model.use_log:
            new_row = pd.Series([np.log1p(val)], index=[next_index])
        else:
            new_row = pd.Series([val], index=[next_index])

        last_window = pd.concat([
            last_window.iloc[1:],
            new_row
        ])

        last_window = last_window.asfreq(date_freq)

        target_hist.append(val)
        target_hist.pop(0)

        for col in search_cols:
            search_hist[col].append(rec[col])
            search_hist[col].pop(0)

    return pd.DataFrame(records, index=future_dates)