from dataclasses import dataclass

import numpy as np
import pandas as pd

from constants import COMPONENT_LABELS, COMPONENT_MAX, CONTROLLABLE_COMPONENTS, WEIGHTS


@dataclass
class RuleBasedResult:
    score: float
    band: str
    components: dict
    contributions: dict
    opportunity_gap: dict


def clamp(value: float, low: float = 0.0, high: float = 100.0) -> float:
    return float(max(low, min(high, value)))


def credit_band(score: float) -> str:
    if score >= 80:
        return "Excellent"
    if score >= 60:
        return "Good"
    if score >= 40:
        return "Fair"
    return "Poor"


def parse_yield_average(yield_data: str) -> float:
    values = []
    for part in str(yield_data).split(","):
        part = part.strip()
        if not part:
            continue
        try:
            values.append(float(part))
        except ValueError:
            continue
    if not values:
        return 0.0
    return float(np.mean(values))


def calculate_component_scores(row: pd.Series, farmer_budget_score: float | None = None) -> dict:
    """Compute all component scores from the raw farmer input row.

    farmer_budget_score can be passed from the Component Scores tab when the
    budget scoring rule is custom and not explicitly defined in documentation.
    """
    crop_map = {"staple": 30, "cash_crop": 40, "vegetable": 20, "other": 10}
    irrigation_map = {"drip": 90, "canal": 70, "none": 30}

    crop_score = 0
    for crop in str(row["Crop Types"]).split(","):
        crop_score += crop_map.get(crop.strip().lower(), 0)

    savings_component = min((float(row["Savings Usd"]) / 1000.0) * 50.0, 50.0)
    freq_component = min(float(row["Payment Frequency"]) * 5.0, 50.0)

    if farmer_budget_score is None:
        # Fallback approximation only when the custom budget score is unavailable.
        farmer_budget_score = clamp(float(row["Farmer Budget Ghs"]) / 200.0)

    return {
        "cs_loan_repayment_history": clamp(float(row["Repayment Rate"])),
        "cs_gender": 75.0 if str(row["Gender"]).strip().lower() != "male" else 50.0,
        "cs_farm_location_risk": clamp(100.0 - float(row["Drought Flood Index"])),
        "cs_mobile_money_score": clamp(savings_component + freq_component),
        "cs_farm_size": clamp(min(float(row["Acres"]) * 10.0, 80.0) + (20.0 if bool(row["Satellite Verified"]) else 0.0)),
        "cs_farmer_budget": clamp(float(farmer_budget_score)),
        "cs_crop_type_score": clamp(crop_score),
        "cs_has_irrigation": float(irrigation_map.get(str(row["Irrigation Type"]).strip().lower(), 30)),
        "cs_market_access": clamp(float(row["Market Access Index"])),
        "cs_seasonal_yield_score": clamp(parse_yield_average(row["Yield Data"])),
        "cs_farmer_association": 75.0 if bool(row["Is Association Member"]) else 25.0,
        "cs_insurance_subscription": 80.0 if bool(row["Insurance Subscription"]) else 30.0,
        "cs_irrigation_scheme": 80.0 if bool(row["Irrigation Scheme"]) else 30.0,
        "cs_training_services": clamp(float(row["Training Sessions"]) * 20.0),
        "cs_livestock_ownership": clamp((float(row["Livestock Value Usd"]) / 100.0) * 50.0),
        "cs_alternative_income": clamp((float(row["Alternative Income Usd"]) / 50.0) * 50.0),
        "cs_has_insurance": 90.0 if str(row["Insurance Type"]).strip().lower() == "both" else (
            80.0 if str(row["Insurance Type"]).strip().lower() in {"crop", "livestock"} else 30.0
        ),
        "cs_peer_endorsements": clamp(float(row["Endorsements"]) * 10.0),
        "cs_soil_health": clamp(float(row["Soil Health Index"])),
        "cs_motorbike_ownership": 75.0 if bool(row["Has Motorbike"]) else 25.0,
        "cs_digital_footprint": clamp(float(row["Digital Score"])),
    }


def score_farmer_rule_based(row: pd.Series, farmer_budget_score: float | None = None) -> RuleBasedResult:
    components = calculate_component_scores(row, farmer_budget_score=farmer_budget_score)
    contributions = {k: components[k] * WEIGHTS[k] for k in WEIGHTS}
    score = round(sum(contributions.values()), 2)
    band = credit_band(score)

    opportunity_gap = {
        key: max(0.0, (COMPONENT_MAX[key] - components[key]) * WEIGHTS[key]) for key in WEIGHTS
    }
    return RuleBasedResult(
        score=score,
        band=band,
        components=components,
        contributions=contributions,
        opportunity_gap=opportunity_gap,
    )


def top_helping_and_dragging(result: RuleBasedResult, n: int = 3) -> tuple[list, list]:
    helping = sorted(result.contributions.items(), key=lambda x: x[1], reverse=True)[:n]
    dragging = sorted(result.opportunity_gap.items(), key=lambda x: x[1], reverse=True)[:n]
    return helping, dragging


def what_if_suggestion(result: RuleBasedResult) -> tuple[str, float]:
    """Pick the single highest-impact controllable component to improve."""
    ranked = sorted(
        [(k, result.opportunity_gap[k]) for k in CONTROLLABLE_COMPONENTS],
        key=lambda x: x[1],
        reverse=True,
    )
    best_key, best_gain = ranked[0]

    suggestions = {
        "cs_mobile_money_score": "Increase monthly mobile savings and transaction activity (e.g., target savings >= 1,000 USD equivalent and frequent MoMo transactions).",
        "cs_crop_type_score": "Diversify crops by adding staple + cash crop + vegetable categories to smooth revenue volatility.",
        "cs_farmer_association": "Join a recognized farmer cooperative to improve support, training, and market access.",
        "cs_motorbike_ownership": "Acquire reliable farm transport (motorbike) to improve mobility and market linkage.",
        "cs_farm_size": "Expand effective cultivated acreage and maintain satellite verification records.",
        "cs_farmer_budget": "Improve seasonal budget strength (better cost control and stronger net operating surplus).",
        "cs_loan_repayment_history": "Improve on-time repayment behavior on all existing loans to push repayment rate toward 100%.",
        "cs_seasonal_yield_score": "Improve yield consistency through better inputs and farm practices across seasons.",
        "cs_peer_endorsements": "Obtain more community/co-op endorsements from trusted peers.",
        "cs_has_irrigation": "Upgrade irrigation toward drip systems where feasible.",
        "cs_irrigation_scheme": "Enroll in a formal irrigation scheme (e.g., GIDA-supported program).",
        "cs_market_access": "Improve route-to-market reliability and access to stronger buyer channels.",
        "cs_training_services": "Attend additional agricultural extension/training sessions.",
        "cs_livestock_ownership": "Increase productive livestock assets to diversify household cash flow.",
        "cs_alternative_income": "Build stable off-farm monthly income to reduce repayment stress.",
        "cs_has_insurance": "Move to broader insurance coverage (ideally both crop and livestock).",
        "cs_insurance_subscription": "Keep insurance premiums current to maintain active subscription status.",
        "cs_digital_footprint": "Increase digital engagement (mobile usage, transaction velocity, app interactions).",
        "cs_soil_health": "Adopt soil restoration practices to improve measured soil health.",
    }

    line = (
        f"Best single what-if change: {COMPONENT_LABELS[best_key]} -> {suggestions[best_key]} "
        f"Estimated score upside: +{best_gain:.2f} points."
    )
    return line, best_gain
