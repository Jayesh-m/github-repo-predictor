# app.py
# simple UI to test the trained models

import pickle

import numpy as np
import pandas as pd
import streamlit as st

# page setup (must be first)
st.set_page_config(
    page_title="GitHub Repo Success Predictor",
    page_icon="⭐",
    layout="centered",
)


# load models once and cache them
@st.cache_resource
def load_models():
    clf = pickle.load(open("decision_tree_model.pkl", "rb"))
    reg = pickle.load(open("linear_regression_model.pkl", "rb"))
    feature_cols = pickle.load(open("feature_columns.pkl", "rb"))
    return clf, reg, feature_cols


clf, reg, feature_cols = load_models()


# sidebar info
with st.sidebar:
    st.header("About")
    st.write(
        "Predicts whether a GitHub repo will be popular and estimates its star count."
    )

    st.markdown("---")

    st.subheader("Models used")
    st.write("- Random Forest (classification)")
    st.write("- Linear Regression (regression)")

    st.markdown("---")

    st.caption("Mini Project")


# title
st.title("GitHub Repository Success Predictor")

st.write("Enter repository details below to predict popularity and star count.")

st.markdown("---")


# input section
st.subheader("Enter Repository Details")

col1, col2 = st.columns(2)

with col1:
    forks = st.number_input("Forks", 0, 100000, 1000, step=100)
    watchers = st.number_input("Watchers", 0, 500000, 50000, step=1000)
    issues = st.number_input("Open Issues", 0, 5000, 100, step=10)
    age = st.number_input("Repo Age (days)", 1, 10000, 1000)

with col2:
    readme = st.number_input("README Length", 0, 500000, 10000)
    language = st.selectbox(
        "Language",
        [
            "Python",
            "JavaScript",
            "TypeScript",
            "Java",
            "Go",
            "Rust",
            "C++",
            "Ruby",
            "PHP",
            "Swift",
            "Other",
        ],
    )
    wiki = st.selectbox("Has Wiki?", [1, 0], format_func=lambda x: "Yes" if x else "No")
    proj = st.selectbox(
        "Has Projects?", [1, 0], format_func=lambda x: "Yes" if x else "No"
    )


st.markdown("---")


# prediction
if st.button("Predict", use_container_width=True):
    # build input
    input_data = {
        "forks_count": forks,
        "watchers_count": watchers,
        "open_issues_count": issues,
        "repo_age_days": age,
        "readme_length": readme,
        "has_wiki": int(wiki),
        "has_projects": int(proj),
    }

    # handle language encoding
    lang_col = f"lang_{language}"
    if lang_col in feature_cols:
        input_data[lang_col] = 1

    # match training columns
    input_df = pd.DataFrame([input_data])
    input_df = input_df.reindex(columns=feature_cols, fill_value=0)

    # predictions
    cls_pred = clf.predict(input_df)[0]

    proba = clf.predict_proba(input_df)
    if proba.shape[1] == 1:
        prob_popular = float(clf.classes_[0])
    else:
        prob_popular = proba[0][1]

    star_pred = float(np.maximum(0, reg.predict(input_df)[0]))

    # results
    st.subheader("Results")

    col1, col2 = st.columns(2)

    with col1:
        if cls_pred == 1:
            st.success("Popular")
        else:
            st.warning("Not Popular")

        st.metric("Confidence", f"{prob_popular:.1%}")

    with col2:
        st.metric("Estimated Stars", f"{int(star_pred):,}")

    # progress bar
    st.progress(float(prob_popular))

    # show inputs (useful in demo)
    with st.expander("See input summary"):
        st.dataframe(input_df, use_container_width=True)


st.markdown("---")
st.caption("GitHub Repo Predictor")
