# Telco Customer Churn Deployment

This project is a clean churn prediction pipeline for the Telco customer dataset.
It starts with exploration, moves into preprocessing and modeling, and ends with a prediction service that can be dockerized and tested.

## Project elements

- `notebooks/01_eda.ipynb`: explore the data, check balance, inspect numeric and categorical patterns, and collect the main signals.
- `notebooks/modeling.ipynb`: preprocess data, build features, train models, compare logistic regression and XGBoost, tune threshold, and run SHAP explainability.
- `api/main.py`: FastAPI app that will load the trained model and expose a `/predict` endpoint.
- `Dockerfile` and `requirements.txt`: containerize the app and lock the environment.
- `tests/`: basic pytest coverage for the prediction flow and any support functions.
- `data/telco_churn.csv`: source dataset for the project.

## Current task list

- Finish the Python module files for preprocessing, prediction, and any reusable helpers.
- Build the FastAPI app around the trained model and expose `/predict`.
- Dockerize the app with the existing `Dockerfile` and dependencies.
- Add basic pytest tests to validate the API and prediction pipeline.
- Push to GitHub with a clean README and the final project structure.

## What is left

The notebooks are done. The remaining work is:

- [ ] build the `.py` files for preprocessing, model loading, and inference
- [ ] wire up the FastAPI app
- [ ] finalize Docker packaging
- [ ] run tests and make sure the app works end to end

