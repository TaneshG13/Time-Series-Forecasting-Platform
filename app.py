import streamlit as st

# session state
from utils.session_state import init_state

# UI pages
from ui import page_input, page_comparison, page_features

st.set_page_config(
    page_title="Time Series Forecasting Platform",
    layout="wide"
)

init_state()

st.sidebar.title("📊 Navigation")

page = st.sidebar.radio(
    "Go to",
    ["Input & Setup", "Model Comparison", "Feature Analysis"]
)

st.title("📈 Time Series Forecasting Platform")

if page == "Input & Setup":
    page_input.render()

elif page == "Model Comparison":
    page_comparison.render()

elif page == "Feature Analysis":
    page_features.render()

st.markdown("---")
st.caption("Built for scalable multi-model time series forecasting 🚀")