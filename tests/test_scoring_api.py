import asyncio
from io import BytesIO

import scoring_api
from scoring_api import BatchRuleScoreItem, RawFarmerInput, ScoreResponse
from starlette.datastructures import UploadFile


class DummyService:
    def score_rule_based(self, payload):
        return ScoreResponse(score=72.5, band="Good", reasoning="Rule reasoning")

    def score_xgboost(self, payload):
        return ScoreResponse(score=75.0, band="Good", reasoning="XGB reasoning")

    def score_both(self, payload):
        return {
            "rule_based": self.score_rule_based(payload),
            "xgboost": self.score_xgboost(payload),
        }

    def score_rule_based_batch(self, payloads):
        return [
            BatchRuleScoreItem(
                row_index=i,
                farmer_id=getattr(p, "farmer_id", None),
                score=70.0 + i,
                band="Good",
                reasoning="Batch reasoning",
            )
            for i, p in enumerate(payloads)
        ]


VALID_PAYLOAD = {
    "farmer_id": 1,
    "farmer_name": "A",
    "gender": "female",
    "region": "Ashanti",
    "drought_flood_index": 22,
    "savings_ghs": 8250,
    "payment_frequency": 14,
    "crop_types": "staple,cash_crop",
    "is_association_member": True,
    "has_motorbike": False,
    "acres": 3.5,
    "satellite_verified": True,
    "repayment_rate": 87,
    "yield_data": "112,104,120",
    "endorsements": 4,
    "irrigation_type": "canal",
    "irrigation_scheme": False,
    "market_access_index": 73,
    "training_sessions": 3,
    "livestock_value_ghs": 2900,
    "alternative_income_ghs": 1300,
    "insurance_type": "crop",
    "insurance_subscription": True,
    "digital_score": 68,
    "soil_health_index": 71,
    "farmer_budget_ghs": 6200,
}


def test_health_ok_when_no_startup_error() -> None:
    scoring_api.startup_error = None
    out = scoring_api.health()
    assert out["status"] == "ok"


def test_score_rule_based_route(monkeypatch) -> None:
    monkeypatch.setattr(scoring_api, "get_service", lambda: DummyService())
    payload = RawFarmerInput(**VALID_PAYLOAD)
    out = scoring_api.score_rule_based_route(payload)
    assert out.score == 72.5
    assert out.band == "Good"


def test_score_rule_based_batch_csv(monkeypatch) -> None:
    monkeypatch.setattr(scoring_api, "get_service", lambda: DummyService())
    csv_text = (
        "farmer_id,farmer_name,gender,region,drought_flood_index,savings_ghs,payment_frequency,crop_types,is_association_member,has_motorbike,acres,satellite_verified,repayment_rate,yield_data,endorsements,irrigation_type,irrigation_scheme,market_access_index,training_sessions,livestock_value_ghs,alternative_income_ghs,insurance_type,insurance_subscription,digital_score,soil_health_index,farmer_budget_ghs\n"
        "1,A,female,Ashanti,22,8250,14,\"staple,cash_crop\",true,false,3.5,true,87,\"112,104,120\",4,canal,false,73,3,2900,1300,crop,true,68,71,6200\n"
    )
    upload = UploadFile(file=BytesIO(csv_text.encode("utf-8")), filename="farmers.csv")
    body = asyncio.run(scoring_api.score_rule_based_batch_csv(upload)).model_dump()
    assert body["count"] == 1
    assert body["results"][0]["row_index"] == 0
    assert body["results"][0]["score"] == 70.0


def test_score_rule_based_batch_csv_rejects_non_csv(monkeypatch) -> None:
    monkeypatch.setattr(scoring_api, "get_service", lambda: DummyService())
    upload = UploadFile(file=BytesIO(b"x"), filename="farmers.txt")
    try:
        asyncio.run(scoring_api.score_rule_based_batch_csv(upload))
        assert False, "Expected exception for non-CSV upload"
    except Exception as exc:
        assert "CSV" in str(exc)
