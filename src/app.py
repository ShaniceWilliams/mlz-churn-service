from typing import Any, Dict
from fastapi import FastAPI
import uvicorn
from src.pipeline.predict import predict_single
from src.schemas.models import Customer, PredictResponse


app = FastAPI(title="ChurnPredict")


@app.post("/predict")
def predict(customer: Customer) -> PredictResponse:
    churn_prob = predict_single(customer)
    return PredictResponse(
        churn_probability=churn_prob,
        churn=bool(churn_prob >= 0.5)
    )
        

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=9696)