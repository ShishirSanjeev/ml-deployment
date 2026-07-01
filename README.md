# Customer Churn Prediction API

An end-to-end machine learning project that predicts customer churn using XGBoost, served via a FastAPI REST API and containerized with Docker.

---

## Project Structure

```
churn-prediction-api/
├── data/
│   └── telco_churn.csv          # Kaggle dataset
├── notebooks/
│   ├── 01_eda.ipynb             # Exploratory Data Analysis
│   └── 02_modeling.ipynb        # Preprocessing, Training, SHAP
├── src/
│   └── (future modular scripts)
├── api/
│   └── main.py                  # FastAPI app
├── models/
│   ├── churn_model.joblib       # Trained pipeline
│   ├── best_threshold.joblib    # Optimal decision threshold
│   └── feature_names.joblib     # Expected input columns
├── tests/
│   └── test_predict.py          # Unit tests
├── Dockerfile
├── requirements.txt
└── README.md
```

---

## Quickstart (Local)

```bash
# 1. Clone the repo
git clone https://github.com/YOUR_USERNAME/churn-prediction-api.git
cd churn-prediction-api

# 2. Install dependencies
pip install -r requirements.txt

# 3. Run the notebooks to train and save the model
jupyter notebook notebooks/02_modeling.ipynb

# 4. Start the API
uvicorn api.main:app --reload --port 8000
```

Visit: http://localhost:8000/docs for the interactive Swagger UI.

---

## Run with Docker

```bash
# Build
docker build -t churn-api .

# Run
docker run -p 8000:8000 churn-api
```

---

## API Endpoints

| Method | Endpoint | Description |
|--------|----------|-------------|
| GET | `/` | Health check |
| GET | `/health` | Model status |
| POST | `/predict` | Single prediction |
| POST | `/predict/batch` | Batch predictions (max 100) |

---

## Example Request

```bash
curl -X POST http://localhost:8000/predict \
  -H "Content-Type: application/json" \
  -d '{
    "gender": "Male",
    "SeniorCitizen": 0,
    "Partner": "Yes",
    "Dependents": "No",
    "tenure": 12,
    "Contract": "Month-to-month",
    "PaperlessBilling": "Yes",
    "PaymentMethod": "Electronic check",
    "MonthlyCharges": 65.5,
    "TotalCharges": 786.0,
    "PhoneService": "Yes",
    "MultipleLines": "No",
    "InternetService": "Fiber optic",
    "OnlineSecurity": "No",
    "OnlineBackup": "No",
    "DeviceProtection": "No",
    "TechSupport": "No",
    "StreamingTV": "Yes",
    "StreamingMovies": "Yes"
  }'
```

## Example Response

```json
{
  "churn_prediction": 1,
  "churn_probability": 0.8241,
  "risk_level": "High",
  "threshold_used": 0.42,
  "model_version": "1.0.0"
}
```

---

## Model Performance

| Model | ROC-AUC | F1 Score |
|-------|---------|----------|
| Logistic Regression (baseline) | ~0.84 | ~0.60 |
| XGBoost | ~0.86 | ~0.63 |
| XGBoost (tuned threshold) | ~0.86 | ~0.65 |

---

## Dataset

[Telco Customer Churn — IBM / Kaggle](https://www.kaggle.com/datasets/blastchar/telco-customer-churn)