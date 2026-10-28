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
data/raw/                 dataset (downloaded, not in Git)
notebooks/                original EDA and model comparison notebooks
reports/original_study/   results of the original study (CV, ablation, SHAP)
```

Full installation and usage instructions will be added as the project progresses.
