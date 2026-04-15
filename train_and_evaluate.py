import pickle
import warnings

warnings.filterwarnings("ignore")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.metrics import (
    ConfusionMatrixDisplay,
    accuracy_score,
    classification_report,
    confusion_matrix,
    mean_squared_error,
    r2_score,
)
from sklearn.model_selection import cross_val_score, train_test_split
from sklearn.tree import DecisionTreeClassifier, plot_tree

sns.set_theme(style="whitegrid", palette="muted")
plt.rcParams["figure.dpi"] = 120

STAR_THRESHOLD = 10000
TOP_N_LANGUAGES = 10
INPUT_CSV = "repos.csv"

# =========================
# STEP 1 — LOAD DATA
# =========================
print("=" * 60)
print("STEP 1 — Loading dataset")
print("=" * 60)

df_raw = pd.read_csv(INPUT_CSV)
print(f"Rows loaded : {len(df_raw)}")
print(f"Columns     : {list(df_raw.columns)}")
print(df_raw.head())

# =========================
# STEP 2 — DATA PREPROCESSING
# =========================
print("\n" + "=" * 60)
print("STEP 2 — Data Preprocessing")
print("=" * 60)

df = df_raw.copy()
df = df.dropna()
df = df.drop_duplicates(subset="name")

if "watchers_count" in df.columns:
    df = df.drop(columns=["watchers_count"])

df["popular"] = (df["stargazers_count"] >= STAR_THRESHOLD).astype(int)

top_langs = df["language"].value_counts().nlargest(TOP_N_LANGUAGES).index.tolist()
df["language"] = df["language"].apply(lambda x: x if x in top_langs else "Other")

df_enc = pd.get_dummies(df, columns=["language"], prefix="lang", dtype=int)
df_enc["log_stars"] = np.log1p(df_enc["stargazers_count"])

# IQR OUTLIER HANDLING
for col in ["stargazers_count", "forks_count", "open_issues_count"]:
    Q1 = df[col].quantile(0.25)
    Q3 = df[col].quantile(0.75)
    IQR = Q3 - Q1
    upper = Q3 + 3 * IQR
    df[col] = df[col].clip(upper=upper)

# =========================
# STEP 3 — FEATURES
# =========================
DROP_COLS = ["name", "stargazers_count", "popular", "log_stars"]
X = df_enc.drop(columns=DROP_COLS)
y_cls = df_enc["popular"]
y_reg = df_enc["log_stars"]
print(f"\nFeature shape: {X.shape}")

# =========================
# STEP 4 — SPLIT
# =========================
X_train, X_test, yc_train, yc_test, yr_train, yr_test = train_test_split(
    X, y_cls, y_reg, test_size=0.2, random_state=42, stratify=y_cls
)

# =========================
# STEP 5 — CLASSIFICATION
# =========================
print("\n" + "=" * 60)
print("STEP 5 — Classification (Random Forest)")
print("=" * 60)

clf = RandomForestClassifier(
    n_estimators=200,
    max_depth=8,
    class_weight="balanced",
    min_samples_leaf=3,
    random_state=42,
    n_jobs=-1,
)
clf.fit(X_train, yc_train)
yc_pred = clf.predict(X_test)

acc = accuracy_score(yc_test, yc_pred)
print(f"Accuracy: {acc:.3f}")
print("\nClassification Report:")
print(classification_report(yc_test, yc_pred, zero_division=0))

# =========================
# STEP 6 — REGRESSION (UPDATED)
# =========================
print("\n" + "=" * 60)
print("STEP 6 — Regression (Random Forest Regressor)")
print("=" * 60)

reg = RandomForestRegressor(
    n_estimators=200,
    max_depth=10,  # Slightly deeper often helps regression
    min_samples_leaf=2,  # Prevents overfitting to noise
    random_state=42,
    n_jobs=-1,
)

reg.fit(X_train, yr_train)

# Predict in log space, then inverse transform
yr_pred_log = reg.predict(X_test)
yr_pred = np.expm1(yr_pred_log)
yr_true = np.expm1(yr_test)
yr_pred = np.maximum(0, yr_pred)  # Stars can't be negative

rmse = np.sqrt(mean_squared_error(yr_true, yr_pred))
mae = np.mean(np.abs(yr_true - yr_pred))
r2 = r2_score(yr_true, yr_pred)

print(f"RMSE: {rmse:.2f}")
print(f"MAE : {mae:.2f}")
print(f"R2  : {r2:.3f}")

# =========================
# STEP 7 — SAVE MODELS
# =========================
print("\nSaving models...")

with open("randomforest_classifier_model.pkl", "wb") as f:
    pickle.dump(clf, f)

with open("randomforest_regressor_model.pkl", "wb") as f:
    pickle.dump(reg, f)

with open("feature_columns.pkl", "wb") as f:
    pickle.dump(X.columns.tolist(), f)

print("Models saved successfully.")

# =========================
# STEP 8 — VISUALIZATIONS
# =========================
print("\n" + "=" * 60)
print("STEP 8 — Visualizations")
print("=" * 60)

fig, axes = plt.subplots(2, 3, figsize=(18, 10))

cm = confusion_matrix(yc_test, yc_pred)
disp = ConfusionMatrixDisplay(cm, display_labels=["Not Popular", "Popular"])
disp.plot(ax=axes[0, 0], colorbar=False)
axes[0, 0].set_title("Confusion Matrix")

importances = pd.Series(clf.feature_importances_, index=X.columns)
importances.nlargest(10).sort_values().plot(kind="barh", ax=axes[0, 1])
axes[0, 1].set_title("Top 10 Feature Importances")

axes[0, 2].hist(df["stargazers_count"], bins=50)
axes[0, 2].set_title("Star Distribution")
axes[0, 2].set_xlabel("Stars")
axes[0, 2].set_ylabel("Frequency")

num_cols = [
    "stargazers_count",
    "forks_count",
    "open_issues_count",
    "repo_age_days",
    "readme_length",
]
corr = df[num_cols].corr()
sns.heatmap(corr, annot=True, fmt=".2f", ax=axes[1, 0], cbar=True)
axes[1, 0].set_title("Correlation Heatmap")

dt = DecisionTreeClassifier(max_depth=3, random_state=42)
dt.fit(X_train, yc_train)
plot_tree(
    dt,
    feature_names=X.columns,
    class_names=["Not Popular", "Popular"],
    filled=True,
    rounded=True,
    fontsize=6,
    ax=axes[1, 1],
)
axes[1, 1].set_title("Decision Tree (Top 3 Levels)")

axes[1, 2].axis("off")
plt.tight_layout()
plt.savefig("visualizations.png", dpi=150)
plt.show()
print("Saved → visualizations.png")
