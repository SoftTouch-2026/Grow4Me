import json
import io
from pathlib import Path
from typing import Any

import pandas as pd
import xgboost as xgb
from fastapi import FastAPI, File, HTTPException, UploadFile
from pydantic import BaseModel, Field
from xgboost import XGBRegressor

from constants import COMPONENT_LABELS, FEATURE_PLAIN_ENGLISH
from rule_based_model import credit_band, score_farmer_rule_based, top_helping_and_dragging, what_if_suggestion


FX_RATE_GHS_PER_USD = 15.0


class RawFarmerInput(BaseModel):
    farmer_id: int | None = None
    farmer_name: str | None = None
    gender: str
    region: str | None = None

    drought_flood_index: float = Field(ge=0, le=100)
    savings_usd: float | None = Field(default=None, ge=0)
    savings_ghs: float | None = Field(default=None, ge=0)
    payment_frequency: int = Field(ge=0)

    crop_types: str
    is_association_member: bool
    has_motorbike: bool
    acres: float = Field(ge=0)
    satellite_verified: bool

    repayment_rate: float = Field(ge=0, le=100)
    yield_data: str
    endorsements: int = Field(ge=0)

    irrigation_type: str
    irrigation_scheme: bool
    market_access_index: float = Field(ge=0, le=100)
    training_sessions: int = Field(ge=0)

    livestock_value_usd: float | None = Field(default=None, ge=0)
    livestock_value_ghs: float | None = Field(default=None, ge=0)
    alternative_income_usd: float | None = Field(default=None, ge=0)
    alternative_income_ghs: float | None = Field(default=None, ge=0)

    insurance_type: str
    insurance_subscription: bool
    digital_score: float = Field(ge=0, le=100)
    soil_health_index: float = Field(ge=0, le=100)

    farmer_budget_ghs: float
    farmer_budget_score: float | None = Field(default=None, ge=0, le=100)


class ScoreResponse(BaseModel):
    score: float
    band: str
    reasoning: str


class CombinedScoreResponse(BaseModel):
    rule_based: ScoreResponse
    xgboost: ScoreResponse


class BatchScoreRequest(BaseModel):
    farmers: list[RawFarmerInput]


class BatchRuleScoreItem(BaseModel):
    row_index: int
    farmer_id: int | None = None
    score: float
    band: str
    reasoning: str


class BatchRuleScoreResponse(BaseModel):
    count: int
    results: list[BatchRuleScoreItem]


def _to_usd(usd_value: float | None, ghs_value: float | None, label: str) -> float:
    if usd_value is not None:
        return float(usd_value)
    if ghs_value is not None:
        return float(ghs_value) / FX_RATE_GHS_PER_USD
    raise HTTPException(status_code=400, detail=f"Missing both USD and GHS value for {label}.")


def _parse_yield_avg(yield_data: str) -> float:
    values = []
    for part in str(yield_data).split(","):
        part = part.strip()
        if not part:
            continue
        values.append(float(part))
    return float(sum(values) / len(values)) if values else 0.0


def _crop_flags(crop_types: str) -> dict[str, int]:
    parts = {c.strip().lower() for c in str(crop_types).split(",") if c.strip()}
    return {
        "has_staple": int("staple" in parts),
        "has_cash_crop": int("cash_crop" in parts),
        "has_vegetable": int("vegetable" in parts),
        "has_other": int("other" in parts),
    }


def _build_rule_row(payload: RawFarmerInput) -> pd.Series:
    return pd.Series(
        {
            "Farmer Id": payload.farmer_id or 0,
            "Farmer Name": payload.farmer_name or "Unknown",
            "Gender": payload.gender,
            "Region": payload.region or "Unknown",
            "Drought Flood Index": payload.drought_flood_index,
            "Savings Usd": _to_usd(payload.savings_usd, payload.savings_ghs, "savings"),
            "Payment Frequency": payload.payment_frequency,
            "Crop Types": payload.crop_types,
            "Is Association Member": payload.is_association_member,
            "Has Motorbike": payload.has_motorbike,
            "Acres": payload.acres,
            "Satellite Verified": payload.satellite_verified,
            "Repayment Rate": payload.repayment_rate,
            "Yield Data": payload.yield_data,
            "Endorsements": payload.endorsements,
            "Irrigation Type": payload.irrigation_type,
            "Irrigation Scheme": payload.irrigation_scheme,
            "Market Access Index": payload.market_access_index,
            "Training Sessions": payload.training_sessions,
            "Livestock Value Usd": _to_usd(payload.livestock_value_usd, payload.livestock_value_ghs, "livestock value"),
            "Alternative Income Usd": _to_usd(payload.alternative_income_usd, payload.alternative_income_ghs, "alternative income"),
            "Insurance Type": payload.insurance_type,
            "Insurance Subscription": payload.insurance_subscription,
            "Digital Score": payload.digital_score,
            "Soil Health Index": payload.soil_health_index,
            "Farmer Budget Ghs": payload.farmer_budget_ghs,
        }
    )


def _build_xgb_features(payload: RawFarmerInput, feature_columns: list[str]) -> pd.DataFrame:
    irrigation = payload.irrigation_type.strip().lower()
    insurance = payload.insurance_type.strip().lower()
    crops = _crop_flags(payload.crop_types)

    row = {
        "drought_flood_index": payload.drought_flood_index,
        "savings_usd": _to_usd(payload.savings_usd, payload.savings_ghs, "savings"),
        "payment_frequency": payload.payment_frequency,
        "acres": payload.acres,
        "repayment_rate": payload.repayment_rate,
        "yield_avg": _parse_yield_avg(payload.yield_data),
        "farmer_budget_ghs": payload.farmer_budget_ghs,
        "endorsements": payload.endorsements,
        "market_access_index": payload.market_access_index,
        "training_sessions": payload.training_sessions,
        "livestock_value_usd": _to_usd(payload.livestock_value_usd, payload.livestock_value_ghs, "livestock value"),
        "alternative_income_usd": _to_usd(payload.alternative_income_usd, payload.alternative_income_ghs, "alternative income"),
        "digital_score": payload.digital_score,
        "soil_health_index": payload.soil_health_index,
        "gender_female": int(payload.gender.strip().lower() == "female"),
        "is_association_member": int(payload.is_association_member),
        "has_motorbike": int(payload.has_motorbike),
        "satellite_verified": int(payload.satellite_verified),
        "irrigation_scheme": int(payload.irrigation_scheme),
        "insurance_subscription": int(payload.insurance_subscription),
        "irr_none": int(irrigation == "none"),
        "irr_canal": int(irrigation == "canal"),
        "irr_drip": int(irrigation == "drip"),
        "ins_none": int(insurance == "none"),
        "ins_crop": int(insurance == "crop"),
        "ins_livestock": int(insurance == "livestock"),
        "ins_both": int(insurance == "both"),
        **crops,
    }

    ordered = {col: row.get(col, 0) for col in feature_columns}
    return pd.DataFrame([ordered], columns=feature_columns)


def _normalize_columns(df: pd.DataFrame) -> pd.DataFrame:
    normalized = df.copy()
    normalized.columns = [str(c).strip().lower().replace(" ", "_") for c in normalized.columns]
    return normalized


def _rows_to_raw_inputs(df: pd.DataFrame) -> list[RawFarmerInput]:
    out: list[RawFarmerInput] = []
    for _, row in df.iterrows():
        payload_dict = {
            k: (None if pd.isna(v) else v)
            for k, v in row.to_dict().items()
        }
        out.append(RawFarmerInput(**payload_dict))
    return out


class ModelService:
    def __init__(self, metadata_path: str = "artifacts/model_metadata.json") -> None:
        self.metadata_path = Path(metadata_path)
        self.metadata = self._load_metadata()
        self.feature_columns = self.metadata["feature_columns"]
        self.model = self._load_xgb_model()

    def _load_metadata(self) -> dict[str, Any]:
        if not self.metadata_path.exists():
            raise FileNotFoundError(f"Metadata file not found: {self.metadata_path}")
        return json.loads(self.metadata_path.read_text(encoding="utf-8"))

    def _resolve_artifact_path(self, rel_path: str) -> Path:
        normalized = rel_path.replace("\\", "/")
        return self.metadata_path.parent.parent / Path(normalized)

    def _load_xgb_model(self) -> XGBRegressor:
        xgb_path = self._resolve_artifact_path(self.metadata["artifacts"]["xgboost_regressor"])
        if not xgb_path.exists():
            raise FileNotFoundError(f"XGBoost model artifact not found: {xgb_path}")
        model = XGBRegressor()
        model.load_model(str(xgb_path))
        return model

    def score_rule_based(self, payload: RawFarmerInput) -> ScoreResponse:
        row = _build_rule_row(payload)
        result = score_farmer_rule_based(row, farmer_budget_score=payload.farmer_budget_score)
        helping, dragging = top_helping_and_dragging(result, n=3)
        suggestion, _ = what_if_suggestion(result)

        helping_text = "; ".join(
            [f"{COMPONENT_LABELS[k]} (+{v:.2f})" for k, v in helping]
        )
        dragging_text = "; ".join(
            [f"{COMPONENT_LABELS[k]} (-opportunity {v:.2f})" for k, v in dragging]
        )

        reasoning = (
            f"Top positive contributors: {helping_text}. "
            f"Biggest drags: {dragging_text}. "
            f"What-if: {suggestion}"
        )

        return ScoreResponse(score=round(result.score, 2), band=result.band, reasoning=reasoning)

    def score_xgboost(self, payload: RawFarmerInput) -> ScoreResponse:
        X = _build_xgb_features(payload, self.feature_columns)
        score = float(self.model.predict(X)[0])
        band = credit_band(score)

        booster = self.model.get_booster()
        contribs = booster.predict(xgb.DMatrix(X, feature_names=self.feature_columns), pred_contribs=True)[0]
        feature_contribs = contribs[:-1]

        top_idx = sorted(range(len(feature_contribs)), key=lambda i: abs(feature_contribs[i]), reverse=True)[:3]
        explanations = []
        for idx in top_idx:
            f_name = self.feature_columns[idx]
            f_label = FEATURE_PLAIN_ENGLISH.get(f_name, f_name.replace("_", " "))
            delta = feature_contribs[idx]
            direction = "increased" if delta >= 0 else "decreased"
            explanations.append(f"{f_label} ({X.iloc[0, idx]}) {direction} score by {delta:+.2f}")

        reasoning = "Top XGBoost drivers: " + "; ".join(explanations) + "."
        return ScoreResponse(score=round(score, 2), band=band, reasoning=reasoning)

    def score_both(self, payload: RawFarmerInput) -> CombinedScoreResponse:
        return CombinedScoreResponse(
            rule_based=self.score_rule_based(payload),
            xgboost=self.score_xgboost(payload),
        )

    def score_rule_based_batch(self, payloads: list[RawFarmerInput]) -> list[BatchRuleScoreItem]:
        if not payloads:
            return []

        results: list[BatchRuleScoreItem] = []
        for i, p in enumerate(payloads):
            rb = self.score_rule_based(p)
            results.append(
                BatchRuleScoreItem(
                    row_index=i,
                    farmer_id=p.farmer_id,
                    score=rb.score,
                    band=rb.band,
                    reasoning=rb.reasoning,
                )
            )

        return results


app = FastAPI(title="GrowForMe Dual Scoring API", version="1.0.0")
service: ModelService | None = None
startup_error: str | None = None


def get_service() -> ModelService:
    global service
    if service is None:
        service = ModelService()
    return service


@app.on_event("startup")
def initialize_service() -> None:
    global startup_error
    try:
        get_service()
        startup_error = None
    except Exception as exc:
        startup_error = str(exc)
        # Keep the API process alive so /health can report actionable diagnostics.
        print(f"[STARTUP WARNING] Model service failed to initialize: {startup_error}")


@app.get("/health")
def health() -> dict[str, str]:
    if startup_error:
        return {
            "status": "degraded",
            "reason": startup_error,
        }
    return {"status": "ok"}


@app.post("/score/rule-based", response_model=ScoreResponse)
def score_rule_based_route(payload: RawFarmerInput) -> ScoreResponse:
    return get_service().score_rule_based(payload)


@app.post("/score/xgboost", response_model=ScoreResponse)
def score_xgboost_route(payload: RawFarmerInput) -> ScoreResponse:
    return get_service().score_xgboost(payload)


@app.post("/score/both", response_model=CombinedScoreResponse)
def score_both_route(payload: RawFarmerInput) -> CombinedScoreResponse:
    return get_service().score_both(payload)


@app.post("/score/batch/both", response_model=BatchRuleScoreResponse)
def score_both_batch_route(req: BatchScoreRequest) -> BatchRuleScoreResponse:
    # Kept this route for backward compatibility. It now returns rule-based-only results.
    results = get_service().score_rule_based_batch(req.farmers)
    return BatchRuleScoreResponse(count=len(results), results=results)


@app.post("/score/batch/rule-based", response_model=BatchRuleScoreResponse)
def score_rule_based_batch_route(req: BatchScoreRequest) -> BatchRuleScoreResponse:
    results = get_service().score_rule_based_batch(req.farmers)
    return BatchRuleScoreResponse(count=len(results), results=results)


@app.post("/score/batch/rule-based/csv", response_model=BatchRuleScoreResponse)
async def score_rule_based_batch_csv(file: UploadFile = File(...)) -> BatchRuleScoreResponse:
    if not file.filename.lower().endswith(".csv"):
        raise HTTPException(status_code=400, detail="Please upload a CSV file.")

    content = await file.read()
    try:
        text = content.decode("utf-8-sig")
    except UnicodeDecodeError:
        raise HTTPException(status_code=400, detail="CSV must be UTF-8 encoded.")

    try:
        df = pd.read_csv(io.StringIO(text))
    except Exception as exc:
        raise HTTPException(status_code=400, detail=f"Unable to parse CSV: {exc}")

    if df.empty:
        return BatchRuleScoreResponse(count=0, results=[])

    df = _normalize_columns(df)
    try:
        farmers = _rows_to_raw_inputs(df)
    except Exception as exc:
        raise HTTPException(status_code=400, detail=f"CSV row validation failed: {exc}")

    results = get_service().score_rule_based_batch(farmers)
    return BatchRuleScoreResponse(count=len(results), results=results)


if __name__ == "__main__":
    demo = RawFarmerInput(
        farmer_id=1001,
        farmer_name="Demo Farmer",
        gender="female",
        region="Ashanti",
        drought_flood_index=22,
        savings_ghs=8250,
        payment_frequency=14,
        crop_types="staple,cash_crop",
        is_association_member=True,
        has_motorbike=False,
        acres=3.5,
        satellite_verified=True,
        repayment_rate=87,
        yield_data="112,104,120",
        endorsements=4,
        irrigation_type="canal",
        irrigation_scheme=False,
        market_access_index=73,
        training_sessions=3,
        livestock_value_ghs=2900,
        alternative_income_ghs=1300,
        insurance_type="crop",
        insurance_subscription=True,
        digital_score=68,
        soil_health_index=71,
        farmer_budget_ghs=6200,
    )
    print(get_service().score_both(demo).model_dump())
