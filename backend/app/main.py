from contextlib import asynccontextmanager
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from app.services.model_loader import load_resources
from app.routes import prediction, models, metrics


@asynccontextmanager
async def lifespan(app: FastAPI):
    app.state.resources = load_resources()
    yield
    app.state.resources = None


app = FastAPI(
    title="Consumer Complaint Classification API",
    version="1.0.0",
    description="Predict complaint categories using SimpleRNN, LSTM, GRU, and TinyBERT.",
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:5173", "http://127.0.0.1:5173"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(prediction.router, prefix="/api", tags=["Prediction"])
app.include_router(models.router, prefix="/api", tags=["Models"])
app.include_router(metrics.router, prefix="/api", tags=["Metrics"])


@app.get("/api/health")
def health():
    return {"status": "ok"}
