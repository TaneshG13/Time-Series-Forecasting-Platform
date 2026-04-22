import streamlit as st

def init_state():

    defaults = {
        "preview": None,
        "raw_df": None,
        "file_name": None,

        "date_col": None,
        "target_col": None,
        "freq": "Weekly",
        "selected_models": [],
        "future_periods": 26,

        "results": None,
        "df": None,

        "is_trained": False
    }

    for key, value in defaults.items():
        if key not in st.session_state:
            st.session_state[key] = value

def set_data(preview_df, raw_df, file_name):
    st.session_state.preview = preview_df
    st.session_state.raw_df = raw_df
    st.session_state.file_name = file_name


def set_config(date_col, target_col, freq, models, future_periods):
    st.session_state.date_col = date_col
    st.session_state.target_col = target_col
    st.session_state.freq = freq
    st.session_state.selected_models = models
    st.session_state.future_periods = future_periods


def set_results(results, df):
    st.session_state.results = results
    st.session_state.df = df
    st.session_state.is_trained = True

def get_preview():
    return st.session_state.get("preview")


def get_raw_df():
    return st.session_state.get("raw_df")


def get_results():
    return st.session_state.get("results")


def get_config():
    return {
        "date_col": st.session_state.get("date_col"),
        "target_col": st.session_state.get("target_col"),
        "freq": st.session_state.get("freq"),
        "models": st.session_state.get("selected_models"),
        "future_periods": st.session_state.get("future_periods"),
    }


def is_trained():
    return st.session_state.get("is_trained", False)

def reset_results():
    st.session_state.results = None
    st.session_state.df = None
    st.session_state.is_trained = False


def reset_all():
    for key in list(st.session_state.keys()):
        del st.session_state[key]