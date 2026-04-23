# GrowForMe Credit Scoring Service

Production-minded credit scoring solution for Ghanaian smallholder farmers with two scoring approaches:
- Rule-based scorecard model
- XGBoost regression model with plain-language explanation

The API supports:
- Single-farmer scoring
- Batch scoring
- Direct CSV upload for batch rule-based scoring

## Tech Stack

- Python 3.10
- FastAPI
- XGBoost
- scikit-learn
- pandas
- pytest
- GitHub Actions (CI)

## Project Structure

- `scoring_api.py`: FastAPI service and request/response contracts
- `rule_based_model.py`: Rule-based scoring logic and component-level reasoning
- `ml_model.py`: ML training/evaluation and artifact export
- `ml_data_prep.py`: Raw-to-ML feature transformation and schema checks
- `credit_scoring_model.py`: End-to-end orchestration script
- `artifacts/`: Saved model artifacts and metadata
- `tests/`: Unit and API tests
- `.github/workflows/ci.yml`: CI pipeline

## Local Setup

1. Create and activate a virtual environment.
2. Install dependencies:

```bash
pip install -r requirements.txt
```

## Run the API

```bash
uvicorn scoring_api:app --host 0.0.0.0 --port 8000
```

Health check:

```bash
curl http://127.0.0.1:8000/health
```

## Run Tests

```bash
pytest -q
```

## API Endpoints

- `GET /health`
- `POST /score/rule-based`
- `POST /score/xgboost`
- `POST /score/both`
- `POST /score/batch/rule-based`
- `POST /score/batch/rule-based/csv`
- `POST /score/batch/both` (backward-compatible alias, returns rule-based batch schema)

## CI/CD

GitHub Actions workflow is configured to run automatically on push and pull requests:
1. Install dependencies from `requirements.txt`
2. Execute test suite with `pytest`

This provides automated quality checks for submission and supports a clean engineering workflow.
