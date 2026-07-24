# 📞 Customer Churn Prediction

End-to-end Machine Learning project to predict telecom customer churn and identify customers at risk of leaving.

The project includes a trained ML model, a REST API, an interactive dashboard, containerization and cloud deployment.

---

## 🚀 Live Demo

🌐 **Streamlit Dashboard**  
https://churn-custormer-pred.streamlit.app/


📖 **API Documentation (Swagger)**  
https://churn-prediction-l95e.onrender.com/docs

---

## 🎯 Project Goal

Customer churn is a major business challenge in the telecommunications industry.

The company loses **25% of its customers every year**, while acquiring a new customer costs **5 times more than retaining an existing one**.

The objective of this project is to build a Machine Learning solution capable of identifying customers at high risk of churn **up to 3 months before they leave**, enabling targeted retention campaigns.

By proactively detecting at-risk customers, the expected business impact is:

- 📉 Reduce churn rate from **25% to 18%** (**-28% reduction**)
- 💰 Generate approximately **€2M annual savings**
- 📈 Achieve a **5:1 ROI** on retention campaigns

---

## 🤖 Machine Learning

### Model

- XGBoost Classifier

### Pipeline

- Data Exploration
- Data preprocessing
- Missing values handling
- Feature engineering
- Model training
- Hyperparameter tuning
- Prediction explainability with SHAP

### Performance

The model achieved:

- **Recall > 80%**

Recall was prioritized because missing a customer who is likely to churn represents a higher business cost.

---

## 🏗️ Architecture

```
                 GitHub
                    |
                    ▼
          GitHub Actions (CI)
          - Unit Tests
          - Docker Build
                    |
        ┌───────────┴───────────┐
        ▼                       ▼
   Render API            Streamlit Cloud
   (FastAPI)              (Dashboard)
        │                       │
        └────────── API ────────┘
```

---

## ✨ Features

### API

- Customer churn prediction
- Churn probability score
- Health monitoring endpoint
- Swagger documentation

### Dashboard

- Customer information input
- Prediction visualization
- Model explanation using SHAP feature importance

---

## 🛠️ Tech Stack

- **Python**
- **Scikit-learn**
- **XGBoost**
- **SHAP**
- **FastAPI**
- **Streamlit**
- **Plotly**
- **Docker**
- **GitHub Actions**
- **Render**
- **Streamlit Cloud**

---

## 📂 Project Structure

```
churn_prediction/
│
├── api/              # FastAPI application
├── dashboard/        # Streamlit application
├── src/              # ML pipeline and utilities
├── models/           # Trained models
├── tests/            # Unit tests
├── notebooks/        # Exploratory analysis
├── Dockerfile
├── docker-compose.yml
└── README.md
```

---

## 🧪 Testing

Unit tests are executed through GitHub Actions.

Run locally:

```bash
pytest
```

---

## 🐳 Run Locally

Clone the repository:

```bash
git clone https://github.com/ellakonan41/churn_prediction.git
```

Build and run with Docker:

```bash
docker compose up --build
```

---

## 🔮 Future Improvements

- Model monitoring
- Automated retraining pipeline
- Model versioning
- Full CD pipeline with GitHub Actions

---

## 👤 Author

**Your Name**

GitHub: https://github.com/ellakonan41

LinkedIn: https://www.linkedin.com/in/ella-ange-konan-556412213/

---

⭐ If you found this project interesting, feel free to give it a star!