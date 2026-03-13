# Predicting Heart Disease

End-to-end machine learning project for predicting heart disease risk from clinical features, with a focus on model interpretability and recall-oriented decision making for healthcare screening scenarios.

## Why This Project Matters

In healthcare triage, missing a high-risk patient (false negative) is often more costly than flagging an extra low-risk patient (false positive). This project builds a robust and explainable classification pipeline, then tunes the decision threshold to prioritize recall while preserving useful precision.

## Results Snapshot

- Final model: Logistic Regression pipeline with feature engineering and preprocessing.
- Test ROC-AUC: 0.9538.
- Baseline CV ROC-AUC: 0.9527 +/- 0.0008.
- Feature-engineered Logistic CV ROC-AUC: 0.9528 +/- 0.0008.
- Threshold strategy: selected threshold 0.3796 to enforce recall >= 0.90.
- Threshold config saved in `artifacts/models/02_modelling_threshold_config.yaml`.

## Project Workflow

1. Exploratory Data Analysis (`notebooks/01_exploratory_analysis.ipynb`)
- Data quality checks, type handling, and target distribution analysis.
- Statistical exploration of numerical and categorical predictors.
- Creation of cleaned dataset used for modelling.

2. Modelling and Evaluation (`notebooks/02_modelling.ipynb`)
- Baseline and feature-engineered pipelines.
- Candidate model comparison (Logistic Regression, Random Forest, XGBoost).
- Experiment tracking with MLflow.
- Hyperparameter tuning and final model selection.
- Threshold optimization for recall-sensitive use cases.

3. Model Interpretation
- Permutation feature importance exported to:
	- `artifacts/models/best_model_feature_importance.csv`
- Top contributors include chest pain type, thallium scan outcomes, number of vessels, exercise angina, and max heart rate.

## Dataset

- Source files:
	- Raw: `notebooks/data/heart_disease.csv`
	- Cleaned: `notebooks/data/heart_disease_cleaned.csv`
- Size: approximately 630,000 rows (synthetically expanded clinical-style dataset used for large-scale experimentation).
- Target:
	- `Heart Disease` (Presence vs Absence)

## Tech Stack

- Python 3.12+
- pandas, numpy, scipy
- scikit-learn
- xgboost, catboost
- matplotlib, seaborn, statsmodels
- MLflow for experiment tracking
- Jupyter Notebook
- uv for environment/dependency management

## Repository Structure

```text
.
|- notebooks/
|  |- 01_exploratory_analysis.ipynb
|  |- 02_modelling.ipynb
|  |- data/
|     |- heart_disease.csv
|     |- heart_disease_cleaned.csv
|- artifacts/
|  |- models/
|     |- 02_modelling_final_pipeline.joblib
|     |- 02_modelling_best_estimator.joblib
|     |- 02_modelling_threshold_config.yaml
|     |- best_model_feature_importance.csv
|  |- plots/
|- mlruns/
|- pyproject.toml
|- README.md
```

## Getting Started

### 1) Install dependencies

```bash
uv sync
```

### 2) Launch notebooks

```bash
uv run jupyter notebook
```

### 3) Reproduce the pipeline

- Run `notebooks/01_exploratory_analysis.ipynb` to generate/verify cleaned data.
- Run `notebooks/02_modelling.ipynb` for model training, evaluation, threshold tuning, and artifact export.

## Portfolio Highlights

- Built a production-style ML workflow from EDA to thresholded model output.
- Balanced model quality and clinical risk priorities through custom threshold selection.
- Used MLflow to make experiments traceable and reproducible.
- Delivered exportable artifacts ready for downstream integration.

## Notes

- This project is intended for educational and portfolio demonstration purposes.
- It does not provide clinical diagnosis and should not be used as a medical decision system.

## License

MIT License.
