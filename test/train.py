    

import pickle
import warnings
warnings.filterwarnings('ignore')

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.tree import DecisionTreeClassifier, plot_tree
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LinearRegression
from sklearn.metrics import (
    accuracy_score, classification_report,
    confusion_matrix, ConfusionMatrixDisplay,
    mean_squared_error, r2_score,
)

sns.set_theme(style='whitegrid', palette='muted')
plt.rcParams['figure.dpi'] = 120

STAR_THRESHOLD  = None    # set to an int to fix the threshold, or None to auto-compute
TOP_N_LANGUAGES = 10      # group languages outside top-10 as 'Other'
INPUT_CSV = 'repos.csv'

# STEP 1 — LOAD DATA

print('=' * 60)
print('STEP 1 — Loading dataset')
print('=' * 60)

df_raw = pd.read_csv(INPUT_CSV)
print(f'Rows loaded : {len(df_raw)}')
print(f'Columns     : {list(df_raw.columns)}')
print(df_raw.head())


# STEP 2 — DATA PREPROCESSING
print('\n' + '=' * 60)
print('STEP 2 — Data Preprocessing (Syllabus-aligned)')
print('=' * 60)

# ── 2a. DATA CLEANING ───────────────────────────────────────────────────────
print('\n[2a] DATA CLEANING')
df = df_raw.copy()

# --- Handling Missing Values ---
missing_before = df.isnull().sum().sum()
print(f'  Missing values before cleaning : {missing_before}')
# Strategy: drop rows with any missing value (small datasets)
# Alternative strategy for large datasets: fill numeric cols with median
NUM_COLS = ['stargazers_count','forks_count','watchers_count',
            'open_issues_count','readme_length','repo_age_days']
for col in NUM_COLS:
    if df[col].isnull().any():
        median_val = df[col].median()
        df[col].fillna(median_val, inplace=True)
        print(f'  Filled missing {col} with median={median_val:.0f}')
df = df.dropna()  
missing_after = df.isnull().sum().sum()
print(f'  Missing values after cleaning  : {missing_after}')

# --- Tuple Duplication (Remove Duplicate Repos) ---
rows_before = len(df)
df = df.drop_duplicates(subset='name')
rows_after  = len(df)
print(f'  Duplicate rows removed : {rows_before - rows_after}')
print(f'  Rows remaining         : {rows_after}')

# --- Noisy Data: cap extreme outliers using IQR method ---
# Repos with astronomically high open_issues may be noise (e.g. 50,000 issues)
print('  Outlier capping (IQR method) on open_issues_count:')
Q1  = df['open_issues_count'].quantile(0.25)
Q3  = df['open_issues_count'].quantile(0.75)
IQR = Q3 - Q1
upper_cap = Q3 + 3 * IQR   # 3×IQR is a mild cap — keeps most data
outliers  = (df['open_issues_count'] > upper_cap).sum()
df['open_issues_count'] = df['open_issues_count'].clip(upper=upper_cap)
print(f'    Q1={Q1:.0f}  Q3={Q3:.0f}  IQR={IQR:.0f}  Cap={upper_cap:.0f}  Capped={outliers} rows')

df = df.reset_index(drop=True)

# ── 2b. DATA INTEGRATION ─────────────────────────────────────────────────────
# Syllabus: "Data Integration: Entity Identification Problem,
#            Redundancy and Correlation Analysis"
print('\n[2b] DATA INTEGRATION — Redundancy & Correlation Analysis')
num_cols_corr = ['stargazers_count','forks_count','watchers_count',
                 'open_issues_count','readme_length','repo_age_days']
corr_matrix = df[num_cols_corr].corr()
print('  Pearson correlation with stargazers_count:')
star_corr = corr_matrix['stargazers_count'].drop('stargazers_count').sort_values(ascending=False)
for col, val in star_corr.items():
    bar = '█' * int(abs(val) * 20)
    print(f'    {col:25s} r={val:+.3f}  {bar}')

# Identify highly correlated pairs (redundancy check)
print('  Highly correlated feature pairs (|r| > 0.85):')
found_redundant = False
for i in range(len(num_cols_corr)):
    for j in range(i+1, len(num_cols_corr)):
        c1, c2 = num_cols_corr[i], num_cols_corr[j]
        r = corr_matrix.loc[c1, c2]
        if abs(r) > 0.85:
            print(f'    {c1} ↔ {c2}  r={r:.3f}  → potential redundancy')
            found_redundant = True
if not found_redundant:
    print('    None found above threshold.')

# ── 2c. DATA REDUCTION ───────────────────────────────────────────────────────
# Syllabus: "Data Reduction: Attribute Subset Selection, Histograms, Sampling"
print('\n[2c] DATA REDUCTION — Attribute Subset Selection')

# Attribute Subset Selection: drop 'name' (identifier, not predictive)
# watchers_count ≈ stargazers_count in GitHub API (near-perfect correlation)
# We keep watchers as a feature since app.py uses it, but flag the redundancy
print('  Attributes dropped (non-predictive identifiers):')
print('    name           → string identifier, no predictive value')
print('  Attributes kept despite correlation:')
print('    watchers_count → kept because app.py exposes it to users')
print(f'  Final attribute count for modelling: {len(num_cols_corr) - 1 + 1} numeric + language (encoded)')

# Sampling note: we already collected a sample (1000 repos) from the population
print(f'  Dataset is a sample of n={len(df)} repos from GitHub population')
print(f'  Sampling strategy: stratified query (stars:>10 stars:<50000, sorted desc)')

# ── 2d. DATA TRANSFORMATION ──────────────────────────────────────────────────
# Syllabus: "Data Transformation, Data Discretization"
print('\n[2d] DATA TRANSFORMATION')

# --- Discretization: continuous stars → binary popular label ---
# This is explicit discretization of a continuous attribute
if STAR_THRESHOLD is None:
    STAR_THRESHOLD = int(df['stargazers_count'].median())
print(f'  Discretization: stargazers_count → popular (binary)')
print(f'  Threshold = median = {STAR_THRESHOLD} stars')
print(f'  Method: equal-frequency binning (median guarantees ~50% per bin)')

# Guard against single-class edge case
if (df['stargazers_count'] > STAR_THRESHOLD).all():
    STAR_THRESHOLD = int(df['stargazers_count'].quantile(0.5))
elif (df['stargazers_count'] <= STAR_THRESHOLD).all():
    STAR_THRESHOLD = int(df['stargazers_count'].quantile(0.5))

df['popular'] = (df['stargazers_count'] > STAR_THRESHOLD).astype(int)
pop   = df['popular'].sum()
total = len(df)
print(f'  Popular (label=1) : {pop}  ({pop/total*100:.1f}%)')
print(f'  Not popular (label=0) : {total-pop}  ({(total-pop)/total*100:.1f}%)')

# --- Normalization (Min-Max): scale numeric features to [0,1] ---
# Stored separately for reporting; models use raw features (tree-based are scale-invariant)
print('  Min-Max normalization (for reporting — trees are scale invariant):')
from sklearn.preprocessing import MinMaxScaler
scaler = MinMaxScaler()
scaled_preview = scaler.fit_transform(df[NUM_COLS])
print(f'    After scaling: min={scaled_preview.min():.2f}  max={scaled_preview.max():.2f}')
print('    (Raw features used for training — tree models do not require scaling)')

# --- Data Transformation: one-hot encode nominal attribute 'language' ---
print('  Nominal → binary encoding (one-hot) for language attribute:')
df_viz = df.copy()   # keep pre-encoded copy for visualizations
top_langs = df['language'].value_counts().nlargest(TOP_N_LANGUAGES).index.tolist()
df['language'] = df['language'].apply(lambda x: x if x in top_langs else 'Other')
df_enc = pd.get_dummies(df, columns=['language'], prefix='lang', dtype=int)
lang_cols = [c for c in df_enc.columns if c.startswith('lang_')]
print(f'  Language columns created: {lang_cols}')

# ── 2e. DEFINE FEATURE MATRIX ────────────────────────────────────────────────
DROP_COLS = ['name', 'stargazers_count', 'popular']
X     = df_enc.drop(columns=DROP_COLS)
y_cls = df_enc['popular']
y_reg = df_enc['stargazers_count']

print(f'\nFeature matrix shape : {X.shape}')
print(f'Features             : {list(X.columns)}')
print(f'y_cls range          : {y_cls.min()} – {y_cls.max()}')
print(f'y_reg range          : {y_reg.min()} – {y_reg.max()}')

# ── 2f. STRATIFIED TRAIN/TEST SPLIT ─────────────────────────────────────────
n_classes = y_cls.nunique()
X_train, X_test, yc_train, yc_test, yr_train, yr_test = train_test_split(
    X, y_cls, y_reg,
    test_size=0.2,
    random_state=42,
    stratify=y_cls if n_classes > 1 else None,
)
print(f'\nTrain samples  : {len(X_train)}  (80%)')
print(f'Test samples   : {len(X_test)}   (20%)')
print(f'Train class dist:\n{yc_train.value_counts().to_string()}')
print(f'Test class dist:\n{yc_test.value_counts().to_string()}')


# ═════════════════════════════════════════════════════════════════════════════
# STEP 3 — CLASSIFICATION: DECISION TREE
# ═════════════════════════════════════════════════════════════════════════════
print('\n' + '=' * 60)
print('STEP 3 — Random Forest Classifier (+ Decision Tree for visualization)')
print('=' * 60)

# Random Forest: builds 200 trees and averages their votes.
# This directly solves the 0%/100% problem because:
#   - Each tree sees a random subset of data and features
#   - Trees disagree with each other on borderline cases
#   - predict_proba = fraction of trees that voted popular
#   - Result: smooth probabilities like 34%, 67%, 81%
clf = RandomForestClassifier(
    n_estimators=200,         # 200 trees — more trees = smoother probabilities
    max_depth=8,              # each tree can go deeper since they are averaged
    class_weight='balanced',  # compensates for class imbalance
    min_samples_leaf=3,       # each leaf needs at least 3 samples
    random_state=42,
    n_jobs=-1,                # use all CPU cores for speed
)
clf.fit(X_train, yc_train)
yc_pred   = clf.predict(X_test)
# predict_proba always returns 2 columns for Random Forest (never single-class issue)
proba_raw = clf.predict_proba(X_test)
if proba_raw.shape[1] == 1:
    only_class = int(clf.classes_[0])
    yc_pred_proba = np.ones(len(X_test)) * float(only_class)
    print('[!] Warning: only one class in training data.')
else:
    yc_pred_proba = proba_raw[:, 1]   # fraction of trees that voted popular=1

# Also keep a single Decision Tree for the visualization diagram (chart 6)
dt_viz = DecisionTreeClassifier(max_depth=3, random_state=42)
dt_viz.fit(X_train, yc_train)

acc = accuracy_score(yc_test, yc_pred)
print(f'Test Accuracy : {acc:.4f}  ({acc*100:.1f}%)')

# 5-fold cross-validation — more reliable than a single split
cv_scores = cross_val_score(clf, X, y_cls, cv=5, scoring='accuracy')
print(f'CV Accuracy   : {cv_scores.mean():.4f} ± {cv_scores.std():.4f}  '
      f'(scores: {[round(s,3) for s in cv_scores]})')

print('\nClassification Report:')
print(classification_report(
    yc_test, yc_pred,
    target_names=['Not Popular (0)', 'Popular (1)'],
    zero_division=0,
))

top_feature = pd.Series(clf.feature_importances_, index=X.columns).idxmax()
print(f'Most important feature : {top_feature}')


# ═════════════════════════════════════════════════════════════════════════════
# STEP 4 — REGRESSION: LINEAR REGRESSION
# ═════════════════════════════════════════════════════════════════════════════
print('=' * 60)
print('STEP 4 — Linear Regression (Star Count Predictor)')
print('=' * 60)

reg = LinearRegression()
reg.fit(X_train, yr_train)
yr_pred = np.maximum(0, reg.predict(X_test))   # stars cannot be negative

rmse = np.sqrt(mean_squared_error(yr_test, yr_pred))
mae  = np.mean(np.abs(yr_test.values - yr_pred))
r2   = r2_score(yr_test, yr_pred)

print(f'RMSE     : {rmse:.2f}  (avg prediction error in stars)')
print(f'MAE      : {mae:.2f}  (mean absolute error in stars)')
print(f'R2 Score : {r2:.4f}  (1.0 = perfect, 0.0 = no better than mean)')

print('\nTop 10 Feature Coefficients (impact per unit increase):')
coef_df = pd.Series(reg.coef_, index=X.columns).sort_values(ascending=False)
print(coef_df.head(10).to_string())


# ═════════════════════════════════════════════════════════════════════════════
# STEP 5 — VISUALIZATIONS (6 charts in one figure)
# ═════════════════════════════════════════════════════════════════════════════
print('\n' + '=' * 60)
print('STEP 5 — Generating visualizations')
print('=' * 60)

fig, axes = plt.subplots(2, 3, figsize=(20, 12))
fig.suptitle(
    'GitHub Repository Success Predictor — Results Dashboard',
    fontsize=16, fontweight='bold', y=1.01,
)

# ── Chart 1: Confusion Matrix ─────────────────────────────────────────────
cm   = confusion_matrix(yc_test, yc_pred)
disp = ConfusionMatrixDisplay(cm, display_labels=['Not Popular', 'Popular'])
disp.plot(cmap='Blues', ax=axes[0, 0], colorbar=False)
axes[0, 0].set_title(f'1. Confusion Matrix\n(Accuracy: {acc*100:.1f}%)',
                     fontweight='bold')

# ── Chart 2: Feature Importance (top 10) ─────────────────────────────────
fi = pd.Series(clf.feature_importances_, index=X.columns)
fi.nlargest(10).sort_values().plot(
    kind='barh', ax=axes[0, 1], color='steelblue', edgecolor='white',
)
axes[0, 1].set_title('2. Top 10 Feature Importances\n(Random Forest — avg Gini)',
                     fontweight='bold')
axes[0, 1].set_xlabel('Importance Score')

# ── Chart 3: Star Count Distribution ─────────────────────────────────────
axes[0, 2].hist(
    df_viz['stargazers_count'], bins=50, log=True,
    color='coral', edgecolor='black', linewidth=0.4,
)
axes[0, 2].axvline(
    STAR_THRESHOLD, color='navy', linestyle='--', linewidth=1.5,
    label=f'Threshold = {STAR_THRESHOLD}',
)
axes[0, 2].set_title('3. Star Count Distribution (log scale)', fontweight='bold')
axes[0, 2].set_xlabel('Stars')
axes[0, 2].set_ylabel('Frequency (log)')
axes[0, 2].legend()

# ── Chart 4: Top 10 Programming Languages ────────────────────────────────
top_lang = df_viz['language'].value_counts().head(10)
bars = axes[1, 0].barh(
    top_lang.index[::-1], top_lang.values[::-1],
    color='teal', edgecolor='white',
)
axes[1, 0].bar_label(bars, padding=3, fontsize=9)
axes[1, 0].set_title('4. Top 10 Programming Languages', fontweight='bold')
axes[1, 0].set_xlabel('Number of Repositories')

# ── Chart 5: Correlation Heatmap ─────────────────────────────────────────
num_cols = ['stargazers_count', 'forks_count', 'watchers_count',
            'open_issues_count', 'readme_length', 'repo_age_days']
num_cols = [c for c in num_cols if c in df_viz.columns]
corr = df_viz[num_cols].corr()
sns.heatmap(
    corr, annot=True, fmt='.2f', cmap='coolwarm',
    ax=axes[1, 1], square=True, linewidths=0.5,
    cbar_kws={'shrink': 0.8},
)
axes[1, 1].set_title('5. Feature Correlation Heatmap', fontweight='bold')

# ── Chart 6: Decision Tree Diagram (depth 3) ─────────────────────────────
plot_tree(
    dt_viz,
    max_depth=3,
    feature_names=X.columns.tolist(),
    class_names=['Not Popular', 'Popular'],
    filled=True,
    rounded=True,
    fontsize=6,
    ax=axes[1, 2],
)
axes[1, 2].set_title('6. Decision Tree Structure (top 3 levels)',
                     fontweight='bold')
# Note: chart 6 shows a single Decision Tree for illustration.
# The actual classifier (clf) is a Random Forest of 200 trees.

plt.tight_layout()
plt.savefig('all_visualizations.png', dpi=150, bbox_inches='tight')
print('Saved --> all_visualizations.png')
plt.show()


# ═════════════════════════════════════════════════════════════════════════════
# STEP 6 — SAVE MODELS
# ═════════════════════════════════════════════════════════════════════════════
print('\n' + '=' * 60)
print('STEP 6 — Saving models')
print('=' * 60)

with open('decision_tree_model.pkl', 'wb') as f:
    pickle.dump(clf, f)
print('Saved --> decision_tree_model.pkl')

with open('linear_regression_model.pkl', 'wb') as f:
    pickle.dump(reg, f)
print('Saved --> linear_regression_model.pkl')

with open('feature_columns.pkl', 'wb') as f:
    pickle.dump(X.columns.tolist(), f)
print('Saved --> feature_columns.pkl')

# Quick round-trip check
clf2 = pickle.load(open('decision_tree_model.pkl', 'rb'))
assert list(clf2.predict(X_test[:3])) == list(clf.predict(X_test[:3])), \
    'Model reload check failed!'
print('\nModel reload check PASSED')


# ═════════════════════════════════════════════════════════════════════════════
# STEP 7 — SUMMARY
# ═════════════════════════════════════════════════════════════════════════════
print('\n' + '=' * 60)
print('  PROJECT SUMMARY')
print('=' * 60)
print(f'  Dataset          : {len(df)} repositories, {X.shape[1]} features')
print(f'  Train / Test     : {len(X_train)} / {len(X_test)} samples')
print(f'  -- CLASSIFICATION --')
print(f'  Test Accuracy    : {acc*100:.1f}%')
print(f'  CV Accuracy      : {cv_scores.mean()*100:.1f}% '
      f'(+/- {cv_scores.std()*100:.1f}%)')
print(f'  Top feature      : {top_feature}')
print(f'  -- REGRESSION --')
print(f'  R2 Score         : {r2:.4f}')
print(f'  RMSE             : {rmse:.1f} stars')
print(f'  MAE              : {mae:.1f} stars')
print('=' * 60)
print('\nAll done! Next step:  streamlit run app.py')