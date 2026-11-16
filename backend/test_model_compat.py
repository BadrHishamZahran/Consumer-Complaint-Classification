
import json
import zipfile
from pathlib import Path

import tensorflow as tf

MODEL_NAMES = ["simple_rnn", "lstm", "gru"]

def remove_null_quantization_config(obj):
    if isinstance(obj, dict):
        if obj.get("quantization_config", "MISSING") is None:
            del obj["quantization_config"]

        for value in obj.values():
            remove_null_quantization_config(value)

    elif isinstance(obj, list):
        for value in obj:
            remove_null_quantization_config(value)


for name in MODEL_NAMES:
    print(f"\n{'=' * 40}")
    print(f"Testing {name}...")

    source = Path(f"exported_models/{name}/model.keras")
    target = Path(f"exported_models/{name}/model_compat.keras")

    if not source.exists():
        print(f"ERROR: Original model not found: {source}")
        continue

    with zipfile.ZipFile(source, "r") as original:
        with zipfile.ZipFile(target, "w", zipfile.ZIP_DEFLATED) as patched:
            for item in original.infolist():
                data = original.read(item.filename)

                if item.filename == "config.json":
                    config = json.loads(data)
                    remove_null_quantization_config(config)
                    data = json.dumps(config).encode("utf-8")

                patched.writestr(item, data)

    try:
        model = tf.keras.models.load_model(
            target,
            compile=False
        )

        print("SUCCESS!")
        print("Input shape:", model.input_shape)
        print("Output shape:", model.output_shape)

    except Exception as error:
        print("FAILED:", repr(error))

print("\nFinished testing all RNN models.")
