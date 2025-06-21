from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import pandas as pd
import joblib
import os
import numpy as np
from joblib import load
from sklearn.feature_extraction.text import TfidfVectorizer
from typing import Optional, Dict, Any

app = FastAPI(title="News Cluster Prediction API")

# Настройка CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Загрузка данных о кластерах
data_path = os.path.join(os.path.dirname(__file__), "../data/lenta-ru-news_1.csv")
news_df = pd.read_csv(data_path)
cluster_names = news_df['topic'].value_counts().to_dict()

# Загрузка лейбл-энкодера
try:
    label_encoder = load('app/label_encoder.pkl')
except Exception as e:
    print(f"Ошибка загрузки LabelEncoder: {e}")
    label_encoder = None

class NewsRequest(BaseModel):
    text: str


class ClusterPrediction(BaseModel):
    cluster: int
    cluster_name: str
    model_used: str
    confidence: Optional[float] = None
    probabilities: Optional[Dict[str, float]] = None
    error: Optional[str] = None
    warning: Optional[str] = None


def load_model(model_path: str):
    try:
        return joblib.load(model_path)
    except Exception as e:
        print(f"Error loading model from {model_path}: {str(e)}")
        return None


def safe_predict(model, vectorizer, text: str):
    try:
        vectorized = vectorizer.transform([text])

        prediction = model.predict(vectorized)[0]

        if hasattr(model, 'predict_proba'):
            proba = model.predict_proba(vectorized)[0]
            confidence = np.max(proba)
            probabilities = {f"{i}": float(p) for i, p in enumerate(proba)}
        else:
            confidence = 0.7
            probabilities = {f"{prediction}": 1.0}

        return {
            "cluster": int(prediction),
            "confidence": float(confidence),
            "probabilities": probabilities,
            "error": None
        }
    except Exception as e:
        return {
            "cluster": -1,
            "confidence": 0.0,
            "probabilities": {},
            "error": str(e)
        }


# Загрузка моделей и векторайзера
models_dir = os.path.join(os.path.dirname(__file__))
vectorizer = load_model(os.path.join(models_dir, "vectorizer.pkl"))

models = {
    "logistic_regression": load_model(os.path.join(models_dir, "logistic_regression.pkl")),
    "decision_tree": load_model(os.path.join(models_dir, "dt.pkl")),
    "knn": load_model(os.path.join(models_dir, "knn.pkl"))
}

@app.post("/predict", response_model=ClusterPrediction)
async def predict_cluster(request: NewsRequest):
    if not request.text or not isinstance(request.text, str):
        raise HTTPException(status_code=400, detail="Текст не может быть пустым")

    if not vectorizer:
        raise HTTPException(status_code=500, detail="Векторизатор не загружен")

    if not models:
        raise HTTPException(status_code=500, detail="Нет доступных моделей для предсказания")

    predictions = []
    for model_name, model in models.items():
        result = safe_predict(model, vectorizer, request.text)
        if result["error"]:
            print(f"Model {model_name} error: {result['error']}")
            continue
        predictions.append({
            **result,
            "model_name": model_name
        })

    if not predictions:
        return {
            "cluster": -1,
            "cluster_name": "Unknown",
            "model_used": "none",
            "error": "All models failed to predict",
            "warning": "No working models available"
        }

    best_prediction = max(predictions, key=lambda x: x["confidence"])

    # Получаем имя категории по числу
    if label_encoder is not None:
        cluster_name = label_encoder.inverse_transform([best_prediction["cluster"]])[0]
    else:
        cluster_name = "Unknown"

    response = {
        "cluster": best_prediction["cluster"],
        "cluster_name": cluster_name,
        "model_used": best_prediction["model_name"],
        "confidence": best_prediction["confidence"],
        "probabilities": best_prediction["probabilities"]
    }

    if best_prediction["confidence"] < 0.5:
        response["warning"] = "Low confidence prediction"

    return response


@app.get("/clusters")
async def get_clusters():
    return {"topics": list(cluster_names.keys())}


@app.get("/health")
async def health_check():
    return {
        "status": "OK" if models else "ERROR",
        "models_loaded": list(models.keys()),
        "vectorizer_loaded": vectorizer is not None
    }


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="localhost", port=8000)