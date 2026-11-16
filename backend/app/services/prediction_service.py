import numpy as np
import torch
from tensorflow.keras.preprocessing.sequence import pad_sequences

from app.services.preprocessing import clean_text
from app.config import DISPLAY_NAMES


def _format_result(model_key, probabilities, class_names):
    probabilities = np.asarray(probabilities, dtype=float)
    index = int(np.argmax(probabilities))
    return {
        "model": DISPLAY_NAMES[model_key],
        "model_key": model_key,
        "predicted_category": class_names[index],
        "confidence": float(probabilities[index]),
        "probabilities": {
            class_names[i]: float(probabilities[i])
            for i in range(len(class_names))
        },
    }


def predict_one(resources, model_key: str, complaint: str):
    cleaned = clean_text(complaint)
    if not cleaned:
        raise ValueError("The complaint contains no usable English words after preprocessing.")

    class_names = resources["class_names"]

    if model_key in resources["rnn_models"]:
        sequence = resources["keras_tokenizer"].texts_to_sequences([cleaned])
        padded = pad_sequences(
            sequence,
            maxlen=resources["max_len"],
            padding="post",
            truncating="post",
        )
        probs = resources["rnn_models"][model_key].predict(padded, verbose=0)[0]
        return _format_result(model_key, probs, class_names)

    if model_key == "transformer":
        max_length = int(resources["transformer_config"].get("max_length", 128))
        inputs = resources["hf_tokenizer"](
            cleaned,
            return_tensors="pt",
            truncation=True,
            max_length=max_length,
        )
        inputs = {key: value.to(resources["device"]) for key, value in inputs.items()}
        with torch.inference_mode():
            logits = resources["hf_model"](**inputs).logits
            probs = torch.softmax(logits, dim=1)[0].cpu().numpy()
        return _format_result("transformer", probs, class_names)

    raise ValueError(f"Unknown model: {model_key}")


def predict_all(resources, complaint: str):
    keys = ["simple_rnn", "lstm", "gru", "transformer"]
    results = [predict_one(resources, key, complaint) for key in keys]
    return {
        "predictions": results,
        "majority_vote": _majority_vote(results),
    }


def _majority_vote(results):
    from collections import Counter
    counts = Counter(item["predicted_category"] for item in results)
    category, votes = counts.most_common(1)[0]
    return {"predicted_category": category, "votes": votes, "total_models": len(results)}
