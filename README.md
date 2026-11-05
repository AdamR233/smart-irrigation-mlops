# Smart Irrigation MLOps

MLOps pipeline for classifying irrigation need (**Low / Medium / High**) from soil, weather and crop data.
Built on the *Smart Irrigation AI* project
([AdamR233/smart-irrigation-ai](https://github.com/AdamR233/smart-irrigation-ai)).

Planned stack: Python, scikit-learn, CatBoost, Docker, GitHub Actions, Jenkins, MLflow, Streamlit, pytest.

## Dataset

[Irrigation Water Requirement Prediction Dataset](https://www.kaggle.com/datasets/miadul/irrigation-water-requirement-prediction-dataset)
(Kaggle, author: miadul). 10 000 rows, 19 features, target `Irrigation_Need`.
The CSV is not stored in this repository; it is downloaded from Kaggle into `data/raw/`.

## Repository structure

```
src/config.py             paths, dataset schema, experiment configurations
src/data.py               download (pinned Kaggle version), validation, stratified split
src/train.py              training, 5-fold CV, test evaluation, saving models
src/evaluate.py           metrics (overall and per class), confusion matrix
src/predict.py            predictions for new data from a saved model
data/raw/                 dataset (downloaded, not in Git)
models/                   trained models and metrics (generated, not in Git)
notebooks/                original EDA and model comparison notebooks
reports/original_study/   results of the original study (CV, ablation, SHAP)
```

## Quick start (Windows, PowerShell, Python 3.14)

```powershell
py -3.14 -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -r requirements.txt

python -m src.data                              # download + validate data
python -m src.train --experiment catboost       # train one experiment
python -m src.train --experiment all            # train all experiments
python -m src.predict --model models/catboost/model.joblib --input data/raw/irrigation_prediction.csv
```

Available experiments: `logistic_regression`, `logistic_regression_balanced`, `decision_tree`,
`random_forest`, `catboost`, `catboost_balanced` (see `src/config.py`).
