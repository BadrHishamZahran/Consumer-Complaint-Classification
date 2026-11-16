# Consumer Complaint Classification

Local deployment of four trained text-classification models (SimpleRNN, LSTM, GRU, and TinyBERT) using **FastAPI + React**.

This repository contains the application source code. The trained model artifacts are downloaded separately from the Kaggle notebook and stored locally under `backend/exported_models/`. They are intentionally excluded from GitHub because they are large.

## 1. Download the trained artifacts from Kaggle

Open the project notebook:

**[Consumer Complaint Classification – Kaggle Notebook](https://www.kaggle.com/code/badrhishamzahran/consumer-complaint-classification)**

Download the exported models archive (ZIP) from the notebook's model-export step and extract it. You do **not** need to train the models again if you already have the exported artifacts.

Keep the ZIP as a backup. The model weights and tokenizer artifacts are not included in this GitHub repository.

## 2. Place the exported files in the expected directory

Copy the **contents** of the extracted `exported_models` directory into:

```text
Consumer-Complaint-Classification/
└── backend/
    └── exported_models/
```

Do not copy the ZIP itself, and do not create an extra nested `exported_models/exported_models` directory.

The deployment expects a structure similar to the following. Some optional CSV files may or may not be present in the Kaggle export.

```text
Consumer-Complaint-Classification/
├── .gitignore
├── README.md
├── backend/
│   ├── requirements.txt
│   ├── exported_models/
│   │   ├── keras_tokenizer.pkl
│   │   ├── label_encoder.pkl
│   │   ├── project_config.json
│   │   ├── simple_rnn/
│   │   │   ├── model.keras
│   │   │   └── model_compat.keras
│   │   ├── lstm/
│   │   │   ├── model.keras
│   │   │   └── model_compat.keras
│   │   ├── gru/
│   │   │   ├── model.keras
│   │   │   └── model_compat.keras
│   │   └── transformer/
│   │       ├── config.json
│   │       ├── hf_config/
│   │       │   └── config.json
│   │       ├── model.safetensors
│   │       ├── tokenizer.json
│   │       ├── tokenizer_config.json
│   │       └── training_args.bin
│   ├── app/
│   │   ├── __init__.py
│   │   ├── config.py
│   │   ├── main.py
│   │   ├── schemas.py
│   │   ├── routes/
│   │   │   ├── __init__.py
│   │   │   ├── prediction.py
│   │   │   ├── models.py
│   │   │   └── metrics.py
│   │   └── services/
│   │       ├── __init__.py
│   │       ├── preprocessing.py
│   │       ├── model_loader.py
│   │       └── prediction_service.py
│   └── test_model_compat.py
└── frontend/
    ├── package.json
    ├── index.html
    ├── vite.config.js
    └── src/
        ├── main.jsx
        ├── App.jsx
        └── style.css
```

**Important model-file notes:**

- The RNN loaders use `model_compat.keras`. These are compatibility copies of the original `.keras` exports; the original `model.keras` files can be kept as backups. The compatibility conversion does not retrain the models or change their learned weights.
- `transformer/config.json` contains project-level Transformer settings. It is **not** the original Hugging Face architecture configuration.
- `transformer/hf_config/config.json` contains the Hugging Face TinyBERT architecture configuration used to instantiate the model before loading the local fine-tuned weights from `model.safetensors`.
- `tokenizer.json` and `tokenizer_config.json` are used to load the local Transformer tokenizer.
- The application resolves the model directory through `backend/app/config.py`, which points to `backend/exported_models/`.

### Restore the TinyBERT architecture configuration (if `hf_config/` is missing)

The Kaggle export may not contain the additional `hf_config/` directory. After installing the backend requirements, open a terminal in `backend` with its virtual environment activated and run this command once while connected to the internet:

```bat
python -c "from transformers import AutoConfig; from pathlib import Path; name='huawei-noah/TinyBERT_General_4L_312D'; c=AutoConfig.from_pretrained(name); p=Path('exported_models/transformer/hf_config'); p.mkdir(parents=True, exist_ok=True); c.save_pretrained(p); print('Saved TinyBERT config:', c.model_type)"
```

This downloads and saves the small architecture configuration only. It does **not** download model weights or retrain TinyBERT. The application loads the tokenizer and fine-tuned weights from the local `transformer/` directory with local-only loading enabled.

## 3. Backend setup (Windows CMD or PowerShell)

The project was run successfully in a **Python 3.10** virtual environment. Create and activate the environment from the project root:

```bat
cd backend
py -3.10 -m venv .venv
.venv\Scripts\activate
python -m pip install --upgrade pip
pip install -r requirements.txt
```

If `py -3.10` is not available, install Python 3.10 and ensure the Python Launcher can find it. Use the Python version and package versions specified for this project rather than mixing packages from another Python installation.

Start the API from the `backend` directory:

```bat
uvicorn app.main:app --reload
```

When startup succeeds, the terminal displays:

```text
Application startup complete.
```

Backend API documentation (while the server is running): <http://127.0.0.1:8000/docs>

All four models are loaded once when the backend starts. Initial startup can take some time and requires sufficient RAM.

## 4. Frontend setup

Keep the backend terminal running. Open a **second terminal** in the project root:

```bat
cd frontend
npm install
npm run dev
```

Open the local URL printed by Vite (usually <http://localhost:5173>).

## 5. Test the prediction API

With the backend running, send a sample request from another CMD terminal:

```bat
curl -X POST http://127.0.0.1:8000/api/predict/all ^
  -H "Content-Type: application/json" ^
  -d "{\"complaint\":\"I was charged twice for my credit card payment.\"}"
```

The request body uses the `complaint` field. The endpoint returns predictions from the available models according to the API implementation.

## 6. Troubleshooting

### Keras `quantization_config` loading error

If loading an RNN model fails with an error similar to:

```text
Unrecognized keyword arguments passed to Embedding: {'quantization_config': None}
```

the exported `.keras` archive may contain a configuration field unsupported by the locally installed Keras version. Use the project's compatibility conversion script to create the `model_compat.keras` files from the original exports. Do not retrain the models. Keep the original `model.keras` files as backups.

### TinyBERT `model_type` configuration error

If Transformers reports that the model configuration has no `model_type`, check that the original Hugging Face architecture configuration exists at:

```text
backend/exported_models/transformer/hf_config/config.json
```

Do not replace the project-level `transformer/config.json` with the Hugging Face configuration. If `hf_config/` is missing, follow the restoration command in Section 2.

### scikit-learn `InconsistentVersionWarning`

The exported `label_encoder.pkl` was created with scikit-learn **1.6.1**. The requirements pin this version to reduce pickle-version mismatch warnings. If the environment already has a different scikit-learn version installed, reinstall the requirements inside the project's virtual environment.

### TensorFlow GPU warning on native Windows

TensorFlow may warn that native Windows GPU support is unavailable. This warning does not necessarily prevent startup; TensorFlow can run the RNN models on the CPU. PyTorch may use CUDA separately if a compatible PyTorch/CUDA installation is available.

### oneDNN messages

TensorFlow may print informational oneDNN messages. They are not, by themselves, startup errors.

## 7. Upload the source code to GitHub (without model artifacts)

The root `.gitignore` excludes `/backend/exported_models/`, model weight files, virtual environments, caches, and frontend build/dependency directories. The trained models remain on your computer and are not uploaded to GitHub.

From the project root, run:

```bat
git init
git add .
git status
```

Before committing, inspect the staged files. Nothing under `backend/exported_models/` should appear. Verify that the model directory is ignored:

```bat
git check-ignore -v backend/exported_models/keras_tokenizer.pkl
```

If Git prints the matching `.gitignore` rule and file path, the ignore rule is active.

Create an empty repository on GitHub, then connect and push (replace the URL with your repository URL):

```bat
git commit -m "Initial project setup"
git branch -M main
git remote add origin https://github.com/YOUR-USERNAME/YOUR-REPOSITORY.git
git push -u origin main
```

If `origin` is already configured, skip `git remote add origin`.

### If model artifacts were already tracked by Git

Adding `.gitignore` does not automatically untrack files Git already tracks. To remove the model directory from Git's index **without deleting the local files**, run this from the project root:

```bat
git rm -r --cached backend/exported_models
git add .gitignore
git commit -m "Stop tracking local model artifacts"
git push
```

After cloning the repository on another computer, download the exported artifacts from the Kaggle notebook and place them in that computer's `backend/exported_models/` directory. Restore `hf_config/` as described in Section 2 if it is not included in the archive. GitHub contains the application source code, not the trained model artifacts.
