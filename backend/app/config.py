from pathlib import Path

BACKEND_DIR = Path(__file__).resolve().parents[1]
EXPORTED_MODELS_DIR = BACKEND_DIR / "exported_models"

RNN_MODEL_NAMES = ("simple_rnn", "lstm", "gru")
DISPLAY_NAMES = {
    "simple_rnn": "SimpleRNN",
    "lstm": "LSTM",
    "gru": "GRU",
    "transformer": "TinyBERT",
}
