# 🏏 IPL Score Prediction

This project focuses on predicting the **final first-innings score in an IPL (Indian Premier League) match** using machine learning techniques. It simulates live-match conditions and forecasts total runs based on current match state like overs, wickets, momentum, and team details.

---

## 📌 Objective

The goal is to estimate the final score at any given point during the first innings (e.g., 5, 10, or 15 overs completed) based on historical ball-by-ball data. This allows analysts, fans, or broadcasters to **anticipate match trajectories** using real-time statistics.

---

## 🗂️ Dataset & Environment

- **Data Source**: Historical IPL ball-by-ball and aggregated match data.
- **Granularity**: Includes per-ball statistics like runs, wickets, over, extras, and player actions.
- **Environment**: Python 3.x in Jupyter Notebook.
- **Key Libraries**:
  - `pandas` – Data handling
  - `numpy` – Numerical operations
  - `matplotlib & seaborn` – Visualizations
  - `scikit-learn` – Machine learning models, preprocessing, evaluation

---

## 🔍 1. Data Loading & Initial Exploration

- Imported data using `pd.read_csv()`.
- Assessed structure with `.shape`, `.info()`, `.head()`, and `.isna()`.
- Reviewed important features like `match_id`, `batting_team`, `bowling_team`, `over`, `runs`, `venue`, etc.

---

## 🧹 2. Data Cleaning & Feature Engineering

### a) Handling Nulls
- Dropped rows with <5% missing data.
- Imputed key fields like `runs_last_5`, `overs`, or `wickets` using median/mode.

### b) Text Normalization
- Unified entries like `Delhi Daredevils` and `Delhi Capitals`.
- Fixed inconsistent venue names for reliable encoding.

### c) Snapshot Aggregation
Converted per-ball logs into structured game snapshots (e.g., at 5 or 10 overs):
- Total runs so far
- Balls remaining
- Wickets in hand
- Momentum (runs in last 5 overs)

### d) Feature Engineering
Created domain-aware features:
- `run_rate = runs / overs`
- `balls_left = 120 - balls_bowled`
- `wickets_left = 10 - wickets_lost`
- `momentum = runs in last 5 overs`
- Interaction terms like `run_rate * wickets_left`

### e) Categorical Encoding
- One-hot encoded: `batting_team`, `bowling_team`, `venue`
- Rare venues grouped as `"Other"` to reduce dimensionality

---

## 📊 3. Exploratory Data Analysis (EDA)

### Visualizations:
- **Scatter Plot**: Runs vs Overs and Runs vs Wickets
- **Heatmap**: Pearson correlation among numerical features

### Insights:
- Run rate drops significantly after 4–5 wickets.
- Certain venues (e.g., Bengaluru) produce consistently higher scores.
- Teams like Mumbai Indians consistently contribute to high-scoring matches.

---

## ✂️ 4. Dataset Splitting

### Standard Split:
- `train_test_split(X, y, test_size=0.2, random_state=42)`
- 80% training, 20% testing ensures balance and reproducibility.

### Time-Based Split (optional):
- Training on seasons 2010–2018, testing on 2019 simulates real-world deployment and prevents future-data leakage.

---

## ⚙️ 5. Modeling

We experimented with several regression models:

### 🔹 Linear Regression (Baseline)
- Simple, interpretable, but limited with non-linear data.
- Fits a straight line minimizing squared error.

### 🌲 Decision Tree Regressor
- Splits data recursively based on feature thresholds.
- Captures non-linear patterns like performance dips after losing wickets.

### 🌳 Random Forest Regressor
- Ensemble of decision trees.
- Uses bootstrapping (Bagging) for variance reduction.
- Handles missing data, outliers, and overfitting better than a single tree.

### 🚀 AdaBoost Regressor
- Boosting model that builds sequential learners.
- Focuses on hard-to-predict samples.
- Final prediction is weighted average of all weak learners.

---

## 📈 6. Model Evaluation

### Metrics Used:
- **MAE** (Mean Absolute Error): Average absolute difference.
- **MSE / RMSE** (Root Mean Squared Error): Penalizes larger deviations.
- **R² Score**: Measures variance explained by model.

### Visual Evaluation:
- **Scatter Plot**: Predicted vs Actual Scores
- **Residual Plot**: Highlights systematic errors or variance issues.

> Final model (AdaBoost) achieved:
> - **RMSE ~6**
> - **R² ~0.88**

---

## 🔧 7. Hyperparameter Tuning

Used `GridSearchCV` and `RandomizedSearchCV` with 5-fold cross-validation:
- `n_estimators`, `max_depth`, `learning_rate` fine-tuned
- Improved MAE/MSE by ~5% without overfitting

---

## 🧠 8. Final Model & Prediction Example

- **Best Model**: `AdaBoostRegressor`
- **Serialization**: `joblib.dump(model, 'ipl_score_predictor.pkl')`
- **Demo**:
```python
input = {
  'batting_team': 'Mumbai Indians',
  'bowling_team': 'CSK',
  'overs': 10.2,
  'wickets': 3,
  'runs_last_5': 45,
  'venue': 'Bengaluru'
}
