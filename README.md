# 🚗 Used Car Price Predictor

Predicting the resale price of used cars from Craigslist listings using a tuned Random Forest model, deployed as an interactive Gradio web app.

This is my **first end-to-end Machine Learning project** — covering data cleaning, EDA, feature encoding, hyperparameter tuning, experiment tracking, and deployment.

## 📌 Project Overview

Used car pricing depends on a mix of numeric and categorical factors — brand, mileage, year, condition, and more. This project builds a regression pipeline that takes these details and predicts a fair market price, then wraps the trained model in a simple web interface anyone can use.

## 📊 Dataset

- **Source:** [Craigslist Cars/Trucks Data](https://www.kaggle.com/datasets/austinreese/craigslist-carstrucks-data) (Kaggle, by Austin Reese)
- ~400K+ vehicle listings scraped from Craigslist across the US
- Features include region, manufacturer, model, year, condition, cylinders, fuel type, odometer, title status, transmission, drive, vehicle type, paint color, state, and posting date

## 🧹 Data Cleaning & Preprocessing

- Dropped irrelevant/high-cardinality columns not useful for modeling: `id`, `url`, `region_url`, `VIN`, `size`, `image_url`, `county`, `description`, `lat`, `long`
- Handled missing values:
  - Categorical columns (`condition`, `cylinders`, `drive`, `type`, `paint_color`) → filled with `"Unknown"`
  - Other categorical columns → filled with mode
  - `odometer` → filled with mean
- Removed outliers using the IQR method on `price`, `odometer`, and `year`
- Final filters: `price` between 500–150,000, `odometer` between 0–250,000, `year` between 1997–2021

## 📈 Exploratory Data Analysis

- Distribution plots and boxplots for numeric columns to spot skew and outliers
- Correlation heatmap across numeric features
- Value counts across categorical columns to understand cardinality (e.g. `region` and `model` have thousands of unique values)

## 🔧 Feature Engineering

Used a `ColumnTransformer` with three different encoding strategies based on column cardinality:

| Encoding | Columns |
|---|---|
| **Target Encoding** | `region`, `model`, `posting_date` (high cardinality) |
| **Ordinal Encoding** | `manufacturer`, `state` |
| **One-Hot Encoding** | `condition`, `cylinders`, `paint_color`, `type`, `fuel`, `drive`, `transmission`, `title_status` |

## 🤖 Model & Training

- **Algorithm:** Random Forest Regressor
- **Hyperparameter Tuning:** [Optuna](https://optuna.org/) — tuned `n_estimators`, `max_depth`, `min_samples_split`, `min_samples_leaf`, `max_features` using 5-fold cross-validation (RMSE as scoring metric)
- **Experiment Tracking:** MLflow — logged parameters, metrics, and the trained model

**Best Parameters found:**
```
n_estimators: 324
max_depth: 35
min_samples_split: 6
min_samples_leaf: 5
max_features: sqrt
```

**Test Set Performance:**
| Metric | Value |
|---|---|
| RMSE | ~8,605 |
| MAE | ~5,203 |
| R² Score | ~0.66 |

## 🖥️ Deployment

The trained model and preprocessor are wrapped in a **Gradio** app (`app.py`) with an interactive UI:

- Dropdowns for categorical fields (manufacturer, condition, fuel type, transmission, drive, vehicle type, paint color, title status, state)
- Sliders/number inputs for year and odometer
- Free-text inputs for region and model (since these are high-cardinality — the Target Encoder falls back to the global mean for unseen values)
- One-click price prediction

### Run it locally
```bash
pip install gradio joblib pandas scikit-learn category_encoders
python app.py
```

## 🛠️ Tech Stack

`Python` · `Pandas` · `NumPy` · `Scikit-learn` · `Category Encoders` · `Optuna` · `MLflow` · `Gradio` · `Matplotlib` · `Seaborn`

## 📂 Files

- `notebook.ipynb` — full EDA, preprocessing, and model training pipeline
- `app.py` — Gradio web app for inference
- `best_random_forest.pkl` — trained model
- `car_price_processor.pkl` — fitted preprocessing pipeline

## 🚀 Future Improvements

- Try gradient boosting models (XGBoost/LightGBM) for comparison
- Add SHAP-based feature importance/explainability
- Expand hyperparameter search (more Optuna trials)
- Deploy on Hugging Face Spaces for public access

---
*This project was built as a first hands-on introduction to the complete ML workflow — from raw data to a deployed, usable app.*
