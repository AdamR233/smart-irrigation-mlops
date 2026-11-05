# Training environment for Smart Irrigation MLOps
FROM python:3.14-slim

WORKDIR /app

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1

# Install dependencies first, so this layer is cached when only the code changes
COPY requirements.txt .
RUN pip install -r requirements.txt

COPY src/ src/

# Default: train the main model; override the command to run other experiments
CMD ["python", "-m", "src.train", "--experiment", "catboost"]
