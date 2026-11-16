from pydantic import BaseModel, Field
from typing import Literal

ModelName = Literal["simple_rnn", "lstm", "gru", "transformer"]

class PredictionRequest(BaseModel):
    complaint: str = Field(min_length=3, max_length=20000)
    model: ModelName = "gru"

class AllPredictionsRequest(BaseModel):
    complaint: str = Field(min_length=3, max_length=20000)
