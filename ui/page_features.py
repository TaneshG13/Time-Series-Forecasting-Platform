import streamlit as st
import pandas as pd
import plotly.express as px


def render():
    st.header("🧠 Feature Importance Analysis")

    results = st.session_state.get("results")

    if not results:
        st.warning("Run models first")
        return

    model_names = list(results.keys())

    tabs = st.tabs(model_names)

    for tab, model_name in zip(tabs, model_names):
        with tab:
            res = results[model_name]

            importance_df = res["features"]["importance"].copy()

            importance_df = importance_df.sort_values(
                by="importance",
                ascending=False
            ).head(20)

            importance_df["importance"] = importance_df["importance"].round(4)

            st.subheader(f"{model_name} - Top Features")

            st.dataframe(
                importance_df,
                use_container_width=True
            )

            fig = px.bar(
                importance_df.sort_values("importance"),
                x="importance",
                y="feature",
                orientation="h",
                title="Feature Importance"
            )

            st.plotly_chart(fig, use_container_width=True)