from fastapi import APIRouter, HTTPException, Request

from app.schemas import PredictionRequest, AllPredictionsRequest
from app.services.prediction_service import predict_one, predict_all

router = APIRouter()


@router.post("/predict")
def predict(payload: PredictionRequest, request: Request):
    try:
        return predict_one(request.app.state.resources, payload.model, payload.complaint)
    except ValueError as exc:
        raise HTTPException(status_code=422, detail=str(exc))


@router.post("/predict/all")
def predict_every_model(payload: AllPredictionsRequest, request: Request):
    try:
        return predict_all(request.app.state.resources, payload.complaint)
    except ValueError as exc:
        raise HTTPException(status_code=422, detail=str(exc))
