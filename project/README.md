# Spaceship Titanic – model + API

Predicts whether a passenger was transported to another dimension, served via a FastAPI endpoint.
Gradient Boosting model, ~81% hold-out accuracy (ROC-AUC 0.91).

## Data
The dataset is not stored in this repo. Download `spaceship-titanic.zip` from
[Kaggle](https://www.kaggle.com/competitions/spaceship-titanic/data) and unzip it into `data/`
(only needed to re-train; the trained model is in `artifacts/`).

## Local development
    pip install -r requirements-dev.txt
    python train.py                       # (re)creates artifacts/model.joblib
    uvicorn app.main:app --reload         # dev server with auto-reload
    pytest -q                             # API tests
    ./curl_examples.sh                    # manual requests (or open /docs)

## Production
    pip install -r requirements.txt
    ./run_prod.sh                         # = gunicorn app.main:app -c gunicorn.conf.py

Environment variables (all optional):
| Variable          | Default            | Meaning                              |
|-------------------|--------------------|--------------------------------------|
| PORT              | 8000               | Port to listen on                    |
| WEB_CONCURRENCY   | min(2*CPUs, 4)     | Number of worker processes           |
| TIMEOUT           | 60                 | Seconds before a stuck worker is killed |
| LOG_LEVEL         | info               | Gunicorn log level                   |

Always start from the project root: the saved model imports `app.features`.
`Procfile` is included for Heroku/Render/Railway-style platforms.

## Deploy to Vercel
    npm i -g vercel
    vercel login
    vercel            # preview deployment
    vercel --prod     # production
Or import the GitHub repo at vercel.com/new. No build settings are needed: Vercel detects
the FastAPI `app` in `app/main.py`, installs `requirements.txt` and uses `.python-version` (3.12).
The first request after idle time is slow (cold start while scikit-learn/pandas load).

## Layout
    app/features.py     feature engineering (shared by training and API)
    app/predictor.py    loads the saved pipeline and predicts
    app/main.py         FastAPI endpoints, pydantic validation, error handling
    train.py            model comparison, evaluation, export
    artifacts/          model.joblib, metadata.json, submission.csv
    requirements.txt    runtime deps (ML libs pinned exactly)
    requirements-dev.txt  + test tools
    gunicorn.conf.py    production server settings
