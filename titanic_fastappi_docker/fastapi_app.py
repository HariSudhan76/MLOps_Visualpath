from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from typing import Optional
import joblib
import pandas as pd
import os

app = FastAPI(title="Titanic FAST API Demo for MLOPs Class")

MODEL = None

class Passenger(BaseModel):
    pclass: int
    sex: str
    age: Optional[float] = None
    sibsp: int
    parch: int
    fare: float
    embarked: str

@app.on_event("startup")
def startup():
    global MODEL
    path = "artifacts/titanic_model.pkl"
    if not os.path.exists(path):
        raise RuntimeError("Model not found")
    MODEL = joblib.load(path)

@app.get("/health")
def health():
    return {"status": "ok"}

@app.get("/msg")
def msg():
    return {"message":"CD/CD pipeline working"}

@app.post("/predict")
def predict(p: Passenger):
    df = pd.DataFrame([p.dict()])
    df["age"] = df["age"].fillna(df["age"].median())
    df["embarked"] = df["embarked"].fillna("S")

    pred = MODEL.predict(df)[0]
    prob = MODEL.predict_proba(df)[0][1]

    return {
        "prediction": int(pred),
        "probability": round(float(prob), 4)
    }

