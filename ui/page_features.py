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

            # ----------------------------------
            # 🔥 Remove lag features (optional but cleaner)
            # ----------------------------------
            importance_df = importance_df[
                ~importance_df["feature"].str.startswith("lag_")
            ]

            # ----------------------------------
            # 🔥 Dynamic top 40%
            # ----------------------------------
            total_features = len(importance_df)
            top_n = max(1, int(total_features * 0.4))

            importance_df = importance_df.sort_values(
                by="importance",
                ascending=False
            ).head(top_n)

            importance_df["importance"] = importance_df["importance"].round(4)

            # ----------------------------------
            # UI Info
            # ----------------------------------
            st.subheader(f"{model_name} - Top {top_n} Features")

            st.caption(
                f"Showing top 40% features ({top_n} out of {total_features})"
            )

            # ----------------------------------
            # Table
            # ----------------------------------
            st.dataframe(
                importance_df,
                width='stretch'
            )

            # ----------------------------------
            # Chart
            # ----------------------------------
            fig = px.bar(
                importance_df.sort_values("importance"),
                x="importance",
                y="feature",
                orientation="h",
                title="Feature Importance"
            )

            fig.update_layout(
                height=400 + (top_n * 10)  # dynamic height (clean UI)
            )

            st.plotly_chart(fig, width='stretch')