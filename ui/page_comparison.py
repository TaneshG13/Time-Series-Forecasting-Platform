import streamlit as st
import pandas as pd
import plotly.graph_objects as go

from utils.session_state import get_config

def render():
    st.header("📊 Model Comparison")

    results = st.session_state.get("results")
    df = st.session_state.get("df")
    config = get_config()

    target = config["target_col"]

    if results is None or df is None or target is None:
        st.warning("Run models from Input page first")
        return

    if target not in df.columns:
        st.error(f"Target column '{target}' not found")
        return

    rows = []
    for name, res in results.items():
        rows.append({
            "Model": name,
            "RMSE": res["Metrics"]["rmse"],
            "MAPE": res["Metrics"]["mape"]
        })

    metrics_df = pd.DataFrame(rows)
    st.subheader("📈 Model Performance")
    st.dataframe(metrics_df.sort_values("RMSE"), use_container_width=True)

    tabs = st.tabs(list(results.keys()))

    for tab, model_name in zip(tabs, results.keys()):
        with tab:
            res = results[model_name]
            preds = res["predictions"]

            fig = go.Figure()

            # Actual
            fig.add_trace(go.Scatter(
                x=df.index,
                y=df[target],
                name="Actual"
            ))

            # Test
            fig.add_trace(go.Scatter(
                x=preds["test_pred"].index,
                y=preds["test_pred"],
                name="Test Forecast"
            ))

            # Future
            fig.add_trace(go.Scatter(
                x=preds["future"].index,
                y=preds["future"][target],
                name="Future Forecast"
            ))

            fig.update_layout(
                title=f"{model_name} Forecast",
                xaxis_title="Date",
                yaxis_title=target,
                hovermode="x unified"
            )

            st.plotly_chart(fig, use_container_width=True)