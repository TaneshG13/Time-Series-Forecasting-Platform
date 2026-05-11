import streamlit as st
import pandas as pd

from pipelines.data_prep import (
    generate_features,
    fetch_trends,
    merge_trends,
    get_related_keywords,
    filter_exogenous_columns,
    clean_data,
    check_missing_indices,
    impute_data,
    get_missing_report,
    keyword_summary
)

from pipelines.ml_pipeline import MLPipeline
from pipelines.sarimax_pipeline import SARIMAXPipeline
from pipelines.varmax_pipeline import VARMAXPipeline
from pipelines.lstm_pipeline import LSTMPipeline
from factory.model_factory import get_model
from utils.session_state import set_data, set_config, set_results


# =====================================================
# 🎯 CLEAN COLUMN NAMES FOR UI
# =====================================================
def pretty_columns(df):
    """Renames trend columns for better display in the UI."""
    rename_map = {}
    for col in df.columns:
        if col.startswith("trend_"):
            clean = col.replace("trend_", "").replace("_", " ").title()
            rename_map[col] = clean
        else:
            rename_map[col] = col
    return df.rename(columns=rename_map)


# =====================================================
# MAIN RENDER FUNCTION
# =====================================================
def render():
    st.title("📥 Data Setup & Preparation")

    # =====================================================
    # 1. UPLOAD
    # =====================================================
    st.markdown("### 1. Upload Dataset")
    file = st.file_uploader("Upload CSV or Excel", type=["csv", "xlsx"])

    if file:
        file.seek(0)
        if file.name.endswith(".csv"):
            df_preview = pd.read_csv(file, nrows=100)
            file.seek(0)
            df_full = pd.read_csv(file)
        else:
            df_preview = pd.read_excel(file, nrows=100)
            file.seek(0)
            df_full = pd.read_excel(file)

        set_data(df_preview, df_full, file.name)

    df_preview = st.session_state.get("preview")
    df_full = st.session_state.get("raw_df")

    if df_preview is None:
        st.info("Upload a dataset to begin")
        return

    cols = list(df_preview.columns)

    # =====================================================
    # 2. CONFIGURATION
    # =====================================================
    st.markdown("### 2. Configuration")
    col1, col2, col3 = st.columns(3)

    with col1:
        date_col = st.selectbox("Date Column", cols)
    with col2:
        target_col = st.selectbox("Target Column", cols, index=1 if len(cols) > 1 else 0)
    with col3:
        freq = st.selectbox("Frequency", ["Weekly", "Monthly"])

    # =====================================================
    # 3. DATA INTEGRITY REPORT
    # =====================================================
    st.markdown("### 3. Data Integrity & Missing Values")
    
    # Analyze the data before preparation
    temp_df = df_full.copy()
    temp_df[date_col] = pd.to_datetime(temp_df[date_col], errors="coerce")
    temp_df = temp_df.set_index(date_col).sort_index()
    
    report = get_missing_report(temp_df, target_col, freq)
    
    # A. Show Missing Indices (Missing Rows)
    if report["total_missing_timestamps"] > 0:
        st.warning(f"⚠️ **Missing Index Alert:** {report['total_missing_timestamps']} timestamps are missing from the timeline.")
        with st.expander("View Missing Timestamps"):
            st.write(report["missing_timestamps"])
    else:
        st.success("✅ **Timeline Integrity:** No missing timestamps detected.")

    # B. Show Missing Values (NaNs in existing rows)
    if report["existing_row_nans"]:
        st.info("🔍 **Column-Specific NaNs:** Some existing rows have missing values.")
        st.json(report["existing_row_nans"])
    
    # C. Choose Treatment
    impute_method = st.selectbox(
        "Select Treatment Strategy",
        ["Seasonal Average", "Rolling Average (Linear)", "Forward Fill", "Remove Missing"],
        help="Seasonal Average uses historical patterns to fill gaps. 'Remove Missing' drops any row with a missing target."
    )

    # =====================================================
    # 4. EXTERNAL SIGNALS (Optional)
    # =====================================================
    st.markdown("### 4. External Signals (Optional)")
    use_trends = st.toggle("Use Google Trends")
    keywords = []
    geo = "US"

    if use_trends:
        geo = st.selectbox("Geography", ["", "US", "IN", "GB", "CA", "AU"], help="Empty = Worldwide")
        seed_keyword = st.text_input("Seed Keyword", placeholder="e.g. retail sales")
        num_keywords = st.number_input("Number of keywords", min_value=10, max_value=50, value=25, step=5)

        if seed_keyword:
            with st.spinner("Fetching related keywords..."):
                related = get_related_keywords(seed_keyword, df=df_full, geo=geo, max_keywords=num_keywords)

            if not related:
                st.warning("No related keywords found.")
            else:
                keywords = st.multiselect(
                    f"Select Keywords ({len(related)} available)",
                    related,
                    default=related[:min(10, len(related))]
                )

    # =====================================================
    # 5. PREPARE DATA
    # =====================================================
    st.markdown("### 5. Run Treatment & Preparation")

    if st.button("⚙️ Prepare Data"):
        df = df_full.copy()

        # 1. Base formatting
        df[date_col] = pd.to_datetime(df[date_col], errors="coerce")
        df = df.set_index(date_col).sort_index()
        df[target_col] = pd.to_numeric(df[target_col], errors="coerce")

        # 2. Re-indexing (Force regular intervals if missing rows were found)
        _, target_freq = check_missing_indices(df, freq)
        full_range = pd.date_range(start=df.index.min(), end=df.index.max(), freq=target_freq)
        df = df.reindex(full_range)

        # 3. Imputation / Treatment
        df = impute_data(df, target_col, freq, method=impute_method)

        # 4. Trends Fetching
        if use_trends and keywords:
            with st.spinner("Fetching trend data..."):
                trends = fetch_trends(keywords, df, freq, geo=geo)
            if not trends.empty:
                df = merge_trends(df, trends)

        # 5. Feature Engineering (Ensures Quarter/Month/Week_Number are preserved)
        df = generate_features(df, target_col, freq)
        df = filter_exogenous_columns(df, target_col, use_trends)
        
        # 6. Final Cleaning
        df = clean_data(df, target_col, freq, method=impute_method)

        st.session_state.prepared_df = df
        st.success("Data treated and prepared successfully!")

    # PREVIEW
    df_prepared = st.session_state.get("prepared_df")
    if df_prepared is not None:
        st.markdown("### 📄 Treated Data Preview")
        st.dataframe(pretty_columns(df_prepared.head(20)), width="stretch")

        st.markdown("### 📊 Keyword Signal Summary")
        summary_df = keyword_summary(df_prepared, target_col)
        st.dataframe(summary_df, width="stretch")

    # =====================================================
    # 6. MODEL SETUP & FORECAST
    # =====================================================
    st.markdown("### 6. Run Forecast")
    models = st.multiselect(
        "Select Models",
        ["Random Forest", "XGBoost", "LightGBM", "CatBoost", "HistGradientBoosting", "VARMAX", "FBProphet", "SARIMAX", "LSTM"],
        default=st.session_state.get("selected_models", [])
    )

    future_periods = st.number_input("Forecast Horizon", min_value=1, max_value=200, value=26)

    if st.button("🚀 Start Training"):
        if df_prepared is None:
            st.warning("Please prepare data first")
            return
        if not models:
            st.warning("Select at least one model")
            return

        set_config(date_col, target_col, freq, models, future_periods)

        with st.spinner("Executing pipeline..."):

            all_results = {}

            for m in models:

                model_obj = get_model(m)

                if hasattr(model_obj, "freq"):
                    model_obj.freq = freq

                if m == "SARIMAX":

                    pipeline = SARIMAXPipeline(
                        model_obj,
                        freq=freq
                    )

                elif m == "VARMAX":

                    pipeline = VARMAXPipeline(
                        model_obj,
                        freq=freq
                    )

                elif m == "LSTM":

                    pipeline = LSTMPipeline(
                        model_obj,
                        freq=freq
                    )

                elif m == "FBProphet":

                    from pipelines.fbprophet_pipeline import (
                        FBProphetPipeline
                    )

                    pipeline = FBProphetPipeline(
                        model=model_obj,
                        freq=freq
                    )

                else:

                    pipeline = MLPipeline(
                        model_obj,
                        freq=freq
                    )

                res = pipeline.run(
                    df=df_prepared,
                    target=target_col,
                    future_periods=future_periods
                )

                all_results[m] = res

            set_results(all_results, df_prepared)
        
        st.success("Training completed!")