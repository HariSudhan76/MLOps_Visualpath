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

@app.on_event("startup")
def startup():
    global MODEL
## --------- Loading the model
    MODEL_PATH = "artifacts/titanic_model.pkl"

    MODEL = joblib.load(MODEL_PATH)

    ## --- FastAPI APP --------

    

@app.get("/")
def read_root():
    return {"message": "Hello World"}

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
        
        preds = MODEL.predict(input_df)
        probs = MODEL.predict_proba(input_df)[:,1]

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
def predict_bulk(passengers: list[Passenger]):
    try:
        rows = [p.dict() for p in passengers]
        input_df = pd.DataFrame(rows)

        #Basic cleaning
        input_df["age"] = input_df["age"].fillna(input_df["age"].median())
        input_df["embarked"] = input_df["embarked"].fillna("S")

        preds = model.predict(input_df)
        probs = model.predict_proba(input_df)[:,1]

        results = []
        for i in range(len(preds)):
            results.append({
                "input": rows[i],
                "prediction":int(preds[i]),
                "probability_of_surviaval": float(round(probs[i],4))
            })
        
        logging.info({"bulk_size":len(results)})
        return {"results":results}
    
    except Exception as e:
        logging.exception("Error during bulk prediction")
        raise HTTPException(status_code=500, detail=str(e))
    

# ----------------------------- Run with uvicorn 

if __name__ =="__main__":
    uvicorn.run("fastapi_app:app",host="0.0.0.0",port=8000,reload=True)


