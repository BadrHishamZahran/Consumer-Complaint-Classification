Run commands from this directory:
1. `py -3.11 -m venv .venv`
2. `.venv\Scripts\activate`
3. `pip install -r requirements.txt`
4. `uvicorn app.main:app --reload`

The model files are expected in `backend/exported_models/`. This directory is excluded by the root `.gitignore`; copy the exported model artifacts there locally.
