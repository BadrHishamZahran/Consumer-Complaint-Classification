import json
import pickle
from pathlib import Path

import tensorflow as tf
import torch
from transformers import AutoConfig, AutoModelForSequenceClassification, AutoTokenizer

from app.config import EXPORTED_MODELS_DIR, RNN_MODEL_NAMES


def _require(path: Path):
    if not path.exists():
        raise FileNotFoundError(
            f"Required exported file not found: {path}\n"
            "Copy your extracted Kaggle exported_models contents into the project/exported_models folder."
        )


def load_resources():
    base = EXPORTED_MODELS_DIR
    _require(base / "project_config.json")
    _require(base / "keras_tokenizer.pkl")
    _require(base / "label_encoder.pkl")

    with open(base / "project_config.json", encoding="utf-8") as f:
        project_config = json.load(f)

    with open(base / "keras_tokenizer.pkl", "rb") as f:
        keras_tokenizer = pickle.load(f)
    with open(base / "label_encoder.pkl", "rb") as f:
        label_encoder = pickle.load(f)

    class_names = list(label_encoder.classes_)
    rnn_config = project_config.get("rnn", {})
    max_len = int(rnn_config.get("max_len", 200))

    rnn_models = {}
    for name in RNN_MODEL_NAMES:
        model_path = EXPORTED_MODELS_DIR / name / "model_compat.keras"
        _require(model_path)
        rnn_models[name] = tf.keras.models.load_model(model_path, compile=False)

    
    transformer_dir = base / "transformer"
    hf_config_dir = transformer_dir / "hf_config"

    _require(hf_config_dir / "config.json")
    _require(transformer_dir / "model.safetensors")
    _require(transformer_dir / "tokenizer.json")

    # Load the original TinyBERT architecture configuration.
    hf_config = AutoConfig.from_pretrained(
        str(hf_config_dir),
        local_files_only=True
    )

    # Match the classification head to the five project classes.
    hf_config.num_labels = len(class_names)
    hf_config.id2label = {
        i: name for i, name in enumerate(class_names)
    }
    hf_config.label2id = {
        name: i for i, name in enumerate(class_names)
    }

    device = torch.device(
        "cuda" if torch.cuda.is_available() else "cpu"
    )

    hf_tokenizer = AutoTokenizer.from_pretrained(
        str(transformer_dir),
        local_files_only=True
    )

    # Load the existing fine-tuned weights using the restored config.
    hf_model = AutoModelForSequenceClassification.from_pretrained(
        str(transformer_dir),
        config=hf_config,
        local_files_only=True,
        use_safetensors=True
    )

    hf_model.to(device)
    hf_model.eval()


    return {
        "project_config": project_config,
        "class_names": class_names,
        "keras_tokenizer": keras_tokenizer,
        "max_len": max_len,
        "rnn_models": rnn_models,
        "transformer_config": {
            "max_length": 128,
            "num_classes": len(class_names),
            "base_model": "huawei-noah/TinyBERT_General_4L_312D"
        },
        "hf_tokenizer": hf_tokenizer,
        "hf_model": hf_model,
        "device": device,
    }
