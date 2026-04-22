import streamlit as st
import pandas as pd

from pipelines.ml_pipeline import MLPipeline
from factory.model_factory import get_model
from utils.session_state import set_data, set_config, set_results


def render():
    st.header("📥 Input & Configuration")

    file = st.file_uploader("Upload Dataset", type=['csv', 'xlsx'])

    if file is not None:
        file.seek(0)

        if file.name.endswith('.csv'):
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

    if df_preview is None or df_full is None:
        st.info("📁 Upload a dataset to begin")
        return

    st.subheader("📄 Data Preview")
    st.dataframe(df_preview.head(20), use_container_width=True)

    cols = list(df_preview.columns)

    date_col = st.selectbox(
        "📅 Date Column",
        cols,
        index=cols.index(st.session_state.get("date_col"))
        if st.session_state.get("date_col") in cols else 0
    )

    target_col = st.selectbox(
        "🎯 Target Column",
        cols,
        index=cols.index(st.session_state.get("target_col"))
        if st.session_state.get("target_col") in cols else 1
    )

    freq = st.selectbox(
        "📊 Frequency",
        ["Weekly", "Monthly"],
        index=["Weekly", "Monthly"].index(st.session_state.get("freq", "Weekly"))
    )

    models = st.multiselect(
        "🤖 Models",
        ["Random Forest", "XGBoost", "LightGBM", "CatBoost", "HistGradientBoosting"],
        default=st.session_state.get("selected_models", [])
    )

    future_periods = st.slider(
        "📈 Forecast Horizon",
        4, 104,
        st.session_state.get("future_periods", 26)
    )

    set_config(date_col, target_col, freq, models, future_periods)

    if st.button("🚀 Train Models"):

        if not models:
            st.warning("Select at least one model")
            return

        with st.spinner("Training..."):

            df = df_full.copy()
            df[date_col] = pd.to_datetime(df[date_col])
            df = df.set_index(date_col).sort_index()

            df = df.asfreq("W-SUN") if freq == "Weekly" else df.asfreq("M")

            all_results = {}

            for m in models:
                model = get_model(m)
                pipeline = MLPipeline(model, freq=freq)

                res = pipeline.run(
                    df=df,
                    target=target_col,
                    future_periods=future_periods
                )

                all_results[m] = res

            set_results(all_results, df)

            st.success("✅ Training completed")
            st.info("👉 Go to Model Comparison page")