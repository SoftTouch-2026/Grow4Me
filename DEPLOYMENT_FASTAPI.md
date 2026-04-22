# GrowForMe FastAPI Deployment (Rule-Based + XGBoost)

This service accepts one **raw farmer payload**, then:

- computes the **rule-based score** with plain-text reasoning
- internally preprocesses the same payload for **XGBoost**
- returns XGBoost score with plain-text reasoning

## Files

- `scoring_api.py` - dual-model FastAPI service
- `artifacts/model_metadata.json` - feature order + artifact references
- `artifacts/xgboost_regressor.json` - trained XGBoost model
- `rule_based_model.py` - rule scoring logic

## Install

```bash
pip install fastapi uvicorn pydantic pandas xgboost
```

## Run

```bash
uvicorn scoring_api:app --host 0.0.0.0 --port 8000 --reload
```

## Endpoints

- `GET /health`
- `POST /score/rule-based`
- `POST /score/xgboost`
- `POST /score/both`

## Example request (raw input)

```json
{
  "farmer_id": 1001,
  "farmer_name": "Demo Farmer",
  "gender": "female",
  "region": "Ashanti",
  "drought_flood_index": 22,
  "savings_ghs": 8250,
  "payment_frequency": 14,
  "crop_types": "staple,cash_crop",
  "is_association_member": true,
  "has_motorbike": false,
  "acres": 3.5,
  "satellite_verified": true,
  "repayment_rate": 87,
  "yield_data": "112,104,120",
  "endorsements": 4,
  "irrigation_type": "canal",
  "irrigation_scheme": false,
  "market_access_index": 73,
  "training_sessions": 3,
  "livestock_value_ghs": 2900,
  "alternative_income_ghs": 1300,
  "insurance_type": "crop",
  "insurance_subscription": true,
  "digital_score": 68,
  "soil_health_index": 71,
  "farmer_budget_ghs": 6200
}
```

## Example response (`/score/both`)

```json
{
  "rule_based": {
    "score": 70.43,
    "band": "Good",
    "reasoning": "Top positive contributors: ..."
  },
  "xgboost": {
    "score": 68.52,
    "band": "Good",
    "reasoning": "Top XGBoost drivers: ..."
  }
}
```

## Notes for integration

- Send **one raw payload** from frontend/backend.
- XGBoost preprocessing is handled **inside** `scoring_api.py`.
- Currency fields support either USD or GHS (`*_usd` preferred, otherwise `*_ghs` converted at 15 GHS/USD).
- Rule-based accepts optional `farmer_budget_score`; if not provided, fallback logic is used.
