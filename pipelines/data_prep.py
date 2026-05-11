import pandas as pd
import numpy as np
import time
import random
from pytrends.request import TrendReq

def check_missing_indices(df, freq):
    """
    Identifies gaps in the time series index based on desired frequency.
    Returns the missing dates and the target frequency string for UI display.
    """
    df = df.copy().sort_index()
    df.index = pd.to_datetime(df.index)
    
    if freq == "Weekly":
        target_freq = "W-SUN"
    elif freq == "Monthly":
        target_freq = "ME"
    else:
        target_freq = pd.infer_freq(df.index) or "D"

    full_range = pd.date_range(start=df.index.min(), end=df.index.max(), freq=target_freq)
    missing_dates = full_range.difference(df.index)
    
    return missing_dates, target_freq

def get_missing_report(df, target, freq):
    """
    Analyzes the dataframe to report both missing indices (rows) 
    and missing values (NaNs) within existing rows for the UI.
    """
    missing_dates, _ = check_missing_indices(df, freq)
    
    # Check for NaNs in columns within rows that DO exist
    nan_counts = df.isna().sum()
    cols_with_nans = nan_counts[nan_counts > 0].to_dict()
    
    return {
        "missing_timestamps": missing_dates.tolist(),
        "total_missing_timestamps": len(missing_dates),
        "existing_row_nans": cols_with_nans
    }

def check_and_fix_interval(df, freq):
    """
    Checks if the index is regular and fixes it by reindexing to the full theoretical range.
    """
    df = df.copy().sort_index()
    df.index = pd.to_datetime(df.index)

    missing, target_freq = check_missing_indices(df, freq)
    
    # Reindex to expose missing dates as NaNs so they can be imputed
    full_range = pd.date_range(start=df.index.min(), end=df.index.max(), freq=target_freq)
    df = df.reindex(full_range)
    
    is_regular = len(missing) == 0
    return df, is_regular, target_freq

def impute_data(df, target, freq, method="Seasonal Average"):
    """
    Fills missing values using the user-selected strategy.
    Options: 'Seasonal Average', 'Rolling Average (Linear)', 'Forward Fill', 'Remove Missing'.
    """
    df = df.copy()
    df.index = pd.to_datetime(df.index)

    # Option to remove missing values (drops rows where target is NaN)
    if method == "Remove Missing":
        return df.dropna(subset=[target])

    # Create seasonal features for grouping
    df["Month"] = df.index.month
    df["Quarter"] = (df.index.month - 1)//3 + 1
    
    if freq == "Weekly":
        df["Week_Number"] = df.index.isocalendar().week.astype(int)
        group_cols = ["Quarter", "Week_Number"]
    else:
        group_cols = ["Quarter", "Month"]

    cols_to_fill = [target] + [c for c in df.columns if c.startswith("trend_")]

    for col in cols_to_fill:
        if col not in df.columns:
            continue

        if method == "Seasonal Average":
            # Impute based on historical average for that specific week/month across years
            df[col] = df[col].fillna(df.groupby(group_cols)[col].transform("mean"))
            # Fallback for remaining gaps (e.g., if a specific week is missing in all history)
            df[col] = df[col].interpolate(method='linear')
            
        elif method == "Rolling Average (Linear)":
            df[col] = df[col].interpolate(method='linear')
            
        elif method == "Forward Fill":
            df[col] = df[col].ffill()

    # Final cleanup for edges (remaining NaNs at start/end)
    df = df.ffill().bfill()
    
    # NOTE: We no longer drop Month, Quarter, or Week_Number here to prevent 
    # KeyError in the recursive forecast lookup logic.
    return df

def get_related_keywords(seed_keyword, df=None, geo="US", max_keywords=50):
    """
    Fetches as many 'Top' related keywords as requested.
    Includes 'df' in signature to maintain compatibility with UI calls.
    """
    pytrends = TrendReq(hl="en-US", tz=360)
    # Use a set to ensure unique keywords
    keyword_pool = {seed_keyword.lower()}

    def fetch_top_queries(kw):
        new_kws = []
        try:
            pytrends.build_payload([kw], timeframe="today 12-m", geo=geo)
            data = pytrends.related_queries()
            if kw in data and data[kw]["top"] is not None:
                # Explicitly fetching 'top' instead of 'rising'
                new_kws = data[kw]["top"]["query"].tolist()
        except Exception:
            pass
        return [k.lower() for k in new_kws]

    # Level 1 Search (Seed)
    initial_results = fetch_top_queries(seed_keyword)
    keyword_pool.update(initial_results)

    # Level 2 Search (Recursive Expansion to get more keywords)
    if len(keyword_pool) < max_keywords:
        for kw in initial_results:
            if len(keyword_pool) >= max_keywords:
                break
            expanded = fetch_top_queries(kw)
            keyword_pool.update(expanded)
            time.sleep(1) # Safety delay for Google rate limits

    final_list = list(keyword_pool)
    return final_list[:max_keywords]

def fetch_trends(keywords, df, freq="Weekly", geo="US"):
    """
    Fetches Google Trends scores and aligns them perfectly to the dataframe's index.
    """
    if not keywords:
        return pd.DataFrame()

    pt = TrendReq(hl="en-US", tz=360)
    start_date = df.index.min().strftime('%Y-%m-%d')
    end_date = df.index.max().strftime('%Y-%m-%d')
    timeframe = f"{start_date} {end_date}"

    trend_df = pd.DataFrame(index=df.index)
    success = 0

    for kw in keywords:
        try:
            pt.build_payload([kw], timeframe=timeframe, geo=geo)
            temp = pt.interest_over_time()

            if temp is not None and not temp.empty:
                if "isPartial" in temp.columns:
                    temp = temp.drop(columns=["isPartial"])
                
                # Resample trend to match main DF frequency
                resample_rule = "W-SUN" if freq == "Weekly" else "ME"
                temp = temp.resample(resample_rule).mean().reindex(df.index)
                temp.columns = [f"trend_{kw}"]
                
                trend_df = trend_df.join(temp, how="left")
                success += 1
                time.sleep(random.uniform(1, 2))
        except Exception:
            continue

    print(f"✅ Trends fetched: {success}/{len(keywords)}")
    return trend_df

def merge_trends(df, trends_df):
    """Joins external signals to the main dataset."""
    if trends_df is None or trends_df.empty:
        return df
    return df.join(trends_df)

def generate_features(df, target, freq="Weekly"):
    """
    Creates time indicators, rolling averages, and 
    Seasonal Index/Contribution features for the forecasting model.
    """
    df = df.copy()
    df.index = pd.to_datetime(df.index)
    
    # 1. Basic Time Features
    df["Month"] = df.index.month
    df["Quarter"] = (df.index.month - 1)//3 + 1
    df["Year"] = df.index.year

    if freq == "Weekly":
        df["Week_Number"] = df.index.isocalendar().week.astype(int)
        df["dayofyear"] = df.index.dayofyear
        df["weekday"] = df.index.weekday
        df["is_month_start"] = df.index.is_month_start.astype(int)
        df["is_month_end"] = df.index.is_month_end.astype(int)

        # --- Engineering SI and Contribution for Weekly ---
        # Calculation: (Target / Quarter Mean)
        q_mean = df.groupby(["Year", "Quarter"])[target].transform("mean")
        df["SI_Quarter_week"] = df[target] / q_mean

        # Calculation: (Target / Quarter Total) * 100
        q_total = df.groupby(["Year", "Quarter"])[target].transform("sum")
        df["contri_week_quarter"] = (df[target] / q_total) * 100
        
        # Calculation: (Target / Month Total) * 100
        m_total = df.groupby(["Year", "Month"])[target].transform("sum")
        df["contri_week_month"] = (df[target] / m_total) * 100

    elif freq == "Monthly":
        # --- Engineering SI and Contribution for Monthly ---
        q_mean = df.groupby(["Year", "Quarter"])[target].transform("mean")
        df["SI_Quarter_month"] = df[target] / q_mean

        q_total = df.groupby(["Year", "Quarter"])[target].transform("sum")
        df["contri_month_quarter"] = (df[target] / q_total) * 100

    # 2. Rolling averages (SMA)
    for w in [3, 4, 5]:
        df[f"SMA_{w}"] = df[target].rolling(window=w, min_periods=1).mean()

    # 3. Clean up any Infinities caused by division by zero total
    df = df.replace([np.inf, -np.inf], np.nan).fillna(0)

    return df

def filter_exogenous_columns(df, target, use_trends):
    """Drops trend-related columns if user disables Google Trends."""
    if not use_trends:
        df = df.drop(
            columns=[c for c in df.columns if c.startswith("trend_")],
            errors="ignore"
        )

    return df

def clean_data(df, target=None, freq="Weekly", method="Seasonal Average"):
    """
    Final data scrubbing and deduplication.
    """
    df = df.sort_index()
    df = df.loc[:, ~df.columns.duplicated()]

    if target and target in df.columns:
        df = impute_data(df, target, freq, method=method)

    # Drop purely empty trend columns that couldn't be fetched or filled
    drop_cols = [
        col for col in df.columns
        if col.startswith("trend_") and df[col].isna().all()
    ]
    df = df.drop(columns=drop_cols, errors="ignore")

    return df

def keyword_summary(df, target):
    """
    Calculates stats for ALL fetched keywords to be shown in the UI.
    """
    trend_cols = [c for c in df.columns if c.startswith("trend_")]
    if not trend_cols:
        return pd.DataFrame({"Keyword": ["No trend data available"]})

    rows = []
    for col in trend_cols:
        series = df[col]
        display_name = col.replace("trend_", "").title()
        
        # Calculate Correlation
        aligned = df[[target, col]].dropna()
        corr = aligned[target].corr(aligned[col]) if len(aligned) > 2 else 0
            
        rows.append({
            "Keyword": display_name,
            "Signal Strength (Var)": round(series.var(), 2),
            "Correlation to Sales": round(corr, 3) if not np.isnan(corr) else 0
        })

    return pd.DataFrame(rows).sort_values(by="Correlation to Sales", ascending=False)