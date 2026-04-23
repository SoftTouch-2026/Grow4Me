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
- `POST /score/batch/rule-based` (JSON batch, rule-based only)
- `POST /score/batch/rule-based/csv` (CSV upload batch, rule-based only)
- `POST /score/batch/both` (legacy alias now returns rule-based-only for compatibility)

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

## Example batch request (`/score/batch/rule-based`)

```json
{
  "farmers": [
    {
      "farmer_id": 1001,
      "farmer_name": "Demo Farmer A",
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
    },
    {
      "farmer_id": 1002,
      "farmer_name": "Demo Farmer B",
      "gender": "male",
      "region": "Ahafo",
      "drought_flood_index": 45,
      "savings_ghs": 2200,
      "payment_frequency": 6,
      "crop_types": "staple,vegetable",
      "is_association_member": false,
      "has_motorbike": false,
      "acres": 1.2,
      "satellite_verified": false,
      "repayment_rate": 64,
      "yield_data": "88,91,93",
      "endorsements": 2,
      "irrigation_type": "none",
      "irrigation_scheme": false,
      "market_access_index": 42,
      "training_sessions": 1,
      "livestock_value_ghs": 1300,
      "alternative_income_ghs": 900,
      "insurance_type": "none",
      "insurance_subscription": false,
      "digital_score": 41,
      "soil_health_index": 56,
      "farmer_budget_ghs": 2500
    }
  ]
}
```

Batch response shape:

```json
{
  "count": 2,
  "results": [
    { "row_index": 0, "farmer_id": 1001, "score": 70.43, "band": "Good", "reasoning": "..." },
    { "row_index": 1, "farmer_id": 1002, "score": 55.11, "band": "Fair", "reasoning": "..." }
  ]
}
```

## Example CSV upload (`/score/batch/rule-based/csv`)

Upload a CSV file (`multipart/form-data`) with raw input columns matching `RawFarmerInput` keys.

Example curl:

```bash
curl -X POST "http://localhost:8000/score/batch/rule-based/csv" \
  -H "accept: application/json" \
  -H "Content-Type: multipart/form-data" \
  -F "file=@farmers.csv"
```

## Notes for integration

- Send **one raw payload** from frontend/backend.
- XGBoost preprocessing is handled **inside** `scoring_api.py`.
- Currency fields support either USD or GHS (`*_usd` preferred, otherwise `*_ghs` converted at 15 GHS/USD).
- Rule-based accepts optional `farmer_budget_score`; if not provided, fallback logic is used.
