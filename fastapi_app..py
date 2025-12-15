##fastapi file for deployment
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field
from typing import Optional
import joblib
import pandas as pd 
import logging
import uvicorn
import os 

## -------- Logging ----------
os.makedirs("logs",exist_ok=True)
logging.basicConfig(
    filename="logs/fastapi_requests.log",
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s"
)

## --------- Loading the model
MODEL_PATH = "artifacts/titanic_model.pkl"

model = joblib.load(MODEL_PATH)

## --- FastAPI APP --------

app = FastAPI(title="Titanic Survival Prediction API", version="1.0")

## -----------Pydantic input model (matches features used in training)-----
class Passenger(BaseModel):
    pclass: int = Field(...,example=3,description="Passenger class (1,2,3)")
    sex: str = Field(...,example="male",description="male or female")
    age: Optional[float] = Field(None,example=32.0, description="Age in years (use median if unknown)")
    sibsp: int = Field(0,example=1,description="Number of Siblings/Spouse abroad")
    parch: int = Field(0,examples=1,description="Number of parents/Children abroad")
    fare: float = Field(10.0, example=10.0, description="Fare Paid")
    embarked: str = Field("S", example="S", description="Port of Embarkation: S, C or Q")

    class Config:
        schema_extra= {
            "example":{
                "pclass":3,
                "sex":"male",
                "age":32.0,
                "sibsp":1,
                "parch":1,
                "fare":10.0,
                "embarked":"S"

            }
        }

## ---------Health endpoint--------
@app.get("/health")
def health():
    return {"status":"ok","model_loaded":True}

##_------------- Predict endpoint (JSON)
@app.post("/predict")
def predit(passenger:Passenger):
    try:
        #Convert incoming pydantic model to DataFrame (Single-row)
        input_df = pd.DataFrame([passenger.dict()])

        ## Some minimal cleaning consistent with training
        if input_df["age"].isnull().any():
            input_df["age"] = input_df["age"].fillna(input_df["age"].median())
        
        if input_df["embarked"].isnull().any():
            input_df["embarked"] = input_df["embarked"].fillna("S")
        
        preds = model.predict(input_df)
        probs = model.predict_proba(input_df)[:,1]

        pred_label = int(preds[0])
        pred_proba = float(probs[0])

        ## logging
        logging.info({
            "input": passenger.dict(),
            "prediction":pred_label,
            "probability": round(pred_proba,4)
        })

        return{
            "prediction": pred_label,
            "survived": bool(pred_label),
            "probability_of_survival": round(pred_proba,4)
        }
    except Exception as e:
        logging.exception("error during prediction")
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/predict_bulk")

