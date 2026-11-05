# Smart Irrigation MLOps

[![CI](https://github.com/AdamR233/smart-irrigation-mlops/actions/workflows/ci.yml/badge.svg)](https://github.com/AdamR233/smart-irrigation-mlops/actions/workflows/ci.yml)

MLOps pipeline for classifying irrigation need (**Low / Medium / High**) from soil, weather and crop data.
Built on the *Smart Irrigation AI* project
([AdamR233/smart-irrigation-ai](https://github.com/AdamR233/smart-irrigation-ai)).

Stack: Python, scikit-learn, CatBoost, Docker, GitHub Actions, Jenkins, MLflow, Streamlit, pytest.

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
tests/                    pytest unit tests (synthetic data, no download)
.github/workflows/ci.yml  GitHub Actions: lint, tests, Docker build
pyproject.toml            pytest and ruff configuration
data/raw/                 dataset (downloaded, not in Git)
models/                   trained models and metrics (generated, not in Git)
notebooks/                original EDA and model comparison notebooks
reports/original_study/   results of the original study (CV, ablation, SHAP)
Dockerfile                training environment image (Python 3.14 + dependencies + src/)
docker-compose.yml        services; volumes keep data/ and models/ on the host
Jenkinsfile               Jenkins pipeline: build, validate data, train, metrics
jenkins/Dockerfile        Jenkins image with Docker CLI
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

## Docker

Requirements: [Docker Desktop](https://www.docker.com/products/docker-desktop/) running (on Windows with WSL 2).
No local Python is needed. The dataset is downloaded automatically on the first run.

```powershell
docker compose build                                    # build the image
docker compose run --rm train                           # train the main model (CatBoost)
```

### Command cheat sheet

| Task | Command |
|---|---|
| Build the image | `docker compose build` |
| Rebuild from scratch (no cache) | `docker compose build --no-cache` |
| Train CatBoost (default) | `docker compose run --rm train` |
| Train all experiments | `docker compose run --rm train python -m src.train --experiment all` |
| Train one experiment | `docker compose run --rm train python -m src.train --experiment decision_tree` |
| Train without cross-validation (fast) | `docker compose run --rm train python -m src.train --experiment all --cv-folds 0` |
| Download and validate data | `docker compose run --rm train python -m src.data --download` |
| Predict with a saved model | `docker compose run --rm train python -m src.predict --model models/catboost/model.joblib --input data/raw/irrigation_prediction.csv --output models/predictions.csv` |
| Open a shell in the container | `docker compose run --rm train bash` |
| List images | `docker image ls` |
| List containers (also stopped) | `docker ps -a` |
| Remove containers and network | `docker compose down` |
| Remove the image | `docker image rm smart-irrigation-mlops:latest` |

`--rm` removes the container after it finishes. Results are kept in `models/` thanks to the volume.
After changing `requirements.txt`, run `docker compose build` again; after changing only `src/`,
a rebuild takes a few seconds (dependencies are cached).

## Tests and Continuous Integration

```powershell
ruff check .        # code quality (style, unused imports, import order)
pytest -v           # unit tests on a small synthetic dataset
```

GitHub Actions (`.github/workflows/ci.yml`) runs on every push to `main` and on every pull request:

1. **Lint and tests**: installs `requirements.txt` on Python 3.14, runs `ruff check .` and `pytest`.
2. **Docker image build**: builds the image and checks that it imports the training code.

Full training on the Kaggle data is not part of CI; it runs in the Jenkins pipeline.
Results are visible in the repository's **Actions** tab and in the badge above.

## Jenkins

Jenkins runs in Docker and starts the training inside the project image
(through the host Docker engine). The dataset is kept in the `smart-irrigation-data` volume.

```powershell
docker compose up -d --build jenkins     # start Jenkins at http://localhost:8080
docker compose logs jenkins              # logs
docker compose stop jenkins              # stop (configuration is kept)
```

Initial admin password:

```powershell
docker compose exec jenkins cat /var/jenkins_home/secrets/initialAdminPassword
```

Job setup: **New Item** → *Pipeline* → **Pipeline script from SCM** → Git,
repository `https://github.com/AdamR233/smart-irrigation-mlops.git`, branch `*/main`,
script path `Jenkinsfile`.

Pipeline stages: **Checkout → Build environment → Validate data → Train → Metrics**.
Run it with **Build with Parameters** (experiment and number of CV folds).
`metrics.json` and the confusion matrix of each run are archived as build artifacts.
