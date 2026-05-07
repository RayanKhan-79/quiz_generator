# RACE Reading Comprehension Quiz Generator

TF-IDF based reading comprehension and quiz generation system with classical ML models, a FastAPI backend, and a Vite React + Tailwind frontend.

## Setup

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -r requirements.txt
```

If this machine uses `pyenv-win`, set a Python version first, for example:

```powershell
pyenv install 3.11.9
pyenv local 3.11.9
```

## Data

Place RACE files under `data/raw/` as either Parquet or CSV:

- `train.parquet` or `train.csv`
- `validation.parquet`, `validation.csv`, or `val.csv`
- `test.parquet` or `test.csv`

The loader normalizes common RACE schemas into:

`id`, `article`, `question`, `A`, `B`, `C`, `D`, `answer`

## Pipeline

```powershell
python -m src.preprocessing
python -m src.model_a_train
python -m src.model_b_train
python -m src.evaluate
```

Processed datasets are written to `data/processed/`. Model artifacts are written to `models/model_a/` and `models/model_b/`. The loader supports Parquet `options` values stored as array-like objects and expands them into populated `A/B/C/D` fields.

Model A reports option-level Logistic Regression, calibrated Linear SVM, the soft-voting ensemble, and auxiliary direct multiclass baselines. After fixing option extraction and retraining, the validation exact-match answer accuracy for option-level Logistic Regression is about 41.8%, and the current saved test split evaluation is about 37.8%.

## Backend

```powershell
uvicorn backend.main:app --reload --port 8000
```

Endpoints:

- `GET /health`
- `GET /sample`
- `POST /generate`
- `POST /verify`
- `GET /metrics`
- `GET /logs/export`

`POST /generate` returns at least five AI-generated cloze-style quiz questions for each submitted article. Each generated item is grounded in a source sentence and includes its own A/B/C/D options, predicted answer, distractors, graduated hints, confidence, and latency fields.

## Frontend

```powershell
cd ui
npm install
npm run dev
```

The UI uses Tailwind utility classes, supports pasted passages and `.txt` article uploads, and expects the backend at `http://localhost:8000` by default. Set `VITE_API_BASE_URL` to override it. The frontend build is pinned to stable Vite 5 packages.
