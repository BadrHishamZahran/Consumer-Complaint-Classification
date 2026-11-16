from fastapi import APIRouter, Request
from app.config import DISPLAY_NAMES

router = APIRouter()

@router.get("/models")
def list_models(request: Request):
    resources = request.app.state.resources
    return {
        "models": [
            {"id": key, "name": DISPLAY_NAMES[key], "loaded": True}
            for key in ("simple_rnn", "lstm", "gru", "transformer")
        ],
        "class_names": resources["class_names"],
        "best_model_by_macro_f1": resources["project_config"].get("best_model_by_macro_f1"),
    }
