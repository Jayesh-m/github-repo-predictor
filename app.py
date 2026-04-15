import pickle
import numpy as np
import pandas as pd
import streamlit as st

# Page setup
st.set_page_config(
    page_title="GitHub Repo Success Predictor",
    page_icon="⭐",
    layout="centered",
)

# Load models
@st.cache_resource
def load_models():
    with open("randomforest_classifier_model.pkl", "rb") as f:
        clf = pickle.load(f)
    with open("randomforest_regressor_model.pkl", "rb") as f:
        reg = pickle.load(f)
    with open("feature_columns.pkl", "rb") as f:
        feature_cols = pickle.load(f)
    return clf, reg, feature_cols

try:
    clf, reg, feature_cols = load_models()
except FileNotFoundError as e:
    st.error(f"❌ Model file not found: {e}")
    st.info("Please run `train_and_evaluate.py` first to generate the .pkl files.")
    st.stop()

# Sidebar
with st.sidebar:
    st.header("About")
    st.write("Predicts whether a GitHub repo will be popular and estimates its star count.")
    st.markdown("---")
    st.subheader("Models Used")
    st.write("- **Random Forest Classifier** (Popularity)")
    st.write("- **Random Forest Regressor** (Star Count)")
    st.markdown("---")
    st.caption("Mini Project • Built with Streamlit")

# Title
st.title("GitHub Repository Success Predictor")
st.write("Enter repository details below to predict popularity and estimated star count.")
st.markdown("---")

# Inputs
st.subheader("Enter Repository Details")

col1, col2 = st.columns(2)

with col1:
    forks = st.number_input("Forks Count", 0, 500000, 1000, step=100)
    # ✅ REMOVED watchers input: not used in training features
    issues = st.number_input("Open Issues Count", 0, 10000, 100, step=10)
    age = st.number_input("Repo Age (days)", 0, 10000, 365)  # Allow 0 for new repos

with col2:
    readme = st.number_input("README Length (chars)", 0, 100000, 1000, step=100)
    language = st.selectbox(
        "Primary Language",
        ["Python", "JavaScript", "TypeScript", "Java", "Go", "Rust", "C++", "Ruby", "PHP", "Swift", "Other"],
    )
    wiki = st.selectbox("Has Wiki?", [1, 0], format_func=lambda x: "Yes" if x else "No")
    proj = st.selectbox("Has Projects?", [1, 0], format_func=lambda x: "Yes" if x else "No")

st.markdown("---")

# Prediction Button
if st.button("🔮 Predict Success", use_container_width=True, type="primary"):
    
    # 1. Build input dictionary (matching training features EXACTLY)
    input_data = {
        "forks_count": forks,
        "open_issues_count": issues,
        "repo_age_days": age,
        "readme_length": readme,
        "has_wiki": int(wiki),
        "has_projects": int(proj),
    }

    # 2. Language one-hot encoding (matching training pipeline)
    # Training used: pd.get_dummies(..., prefix="lang")
    lang_col = f"lang_{language}"
    for col in feature_cols:
        if col.startswith("lang_"):
            input_data[col] = 1 if col == lang_col else 0

    # 3. Create DataFrame and align with training columns
    input_df = pd.DataFrame([input_data])
    input_df = input_df.reindex(columns=feature_cols, fill_value=0)

    # 4. Apply IQR clipping (same logic as training) to handle extreme inputs
    # Prevents a user entering 1M forks from breaking the model
    for col in ["forks_count", "open_issues_count"]:
        if col in input_df.columns:
            # Approximate upper bounds from training (adjust if your IQR values differ)
            upper_bounds = {"forks_count": 15000, "open_issues_count": 2000}
            input_df[col] = input_df[col].clip(upper=upper_bounds.get(col, np.inf))

    # 5. Make predictions
    # Classification
    cls_pred = clf.predict(input_df)[0]
    proba = clf.predict_proba(input_df)[0]
    prob_popular = float(proba[1]) if len(proba) > 1 else float(proba[0])

    log_star_pred = reg.predict(input_df)[0]
    star_pred = np.expm1(log_star_pred)  # Convert log(1+x) back to x
    star_pred = max(0, star_pred)  # Safety clamp

    # Display Results
    st.subheader("Prediction Results")
    
    col_res1, col_res2 = st.columns(2)
    
    with col_res1:
        st.markdown("### Popularity Classification")
        if prob_popular < 0.5:
            st.warning(f"**Not Popular**")
        else:
            st.success(f"**Popular**")
        
        # Confidence bar with color coding
        conf_color = "green" if prob_popular >= 0.7 else "orange" if prob_popular >= 0.5 else "red"
        st.markdown(f"**Confidence**: <span style='color:{conf_color}'>{prob_popular:.1%}</span>", unsafe_allow_html=True)
        st.progress(prob_popular)

    with col_res2:
        st.markdown("### Estimated Star Count")
        st.metric("Predicted Stars", f"{int(star_pred):,}")
        st.caption("Based on current features • Actual growth may vary")

    # Debug/Transparency Section
    with st.expander("See Technical Details"):
        st.write("**Input Features (Processed)**")
        st.dataframe(input_df.T, use_container_width=True)
        st.write(f"**Model Outputs:**")
        st.json({
            "classification_class": int(cls_pred),
            "probability_popular": round(prob_popular, 4),
            "log_star_prediction": round(log_star_pred, 4),
            "final_star_prediction": int(star_pred)
        })

st.markdown("---")
st.caption("Built for educational purposes • Predictions are estimates, not guarantees")