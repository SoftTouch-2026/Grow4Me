import pandas as pd

from rule_based_model import credit_band, parse_yield_average, score_farmer_rule_based


def _sample_row() -> pd.Series:
    return pd.Series(
        {
            "Farmer Id": 1,
            "Farmer Name": "Test Farmer",
            "Gender": "female",
            "Region": "Ashanti",
            "Drought Flood Index": 20,
            "Savings Usd": 600,
            "Payment Frequency": 10,
            "Crop Types": "staple,cash_crop",
            "Is Association Member": True,
            "Has Motorbike": False,
            "Acres": 3.0,
            "Satellite Verified": True,
            "Repayment Rate": 85,
            "Yield Data": "100,110,120",
            "Endorsements": 3,
            "Irrigation Type": "canal",
            "Irrigation Scheme": True,
            "Market Access Index": 70,
            "Training Sessions": 3,
            "Livestock Value Usd": 150,
            "Alternative Income Usd": 120,
            "Insurance Type": "crop",
            "Insurance Subscription": True,
            "Digital Score": 65,
            "Soil Health Index": 68,
            "Farmer Budget Ghs": 5000,
        }
    )


def test_credit_band_thresholds() -> None:
    assert credit_band(80) == "Excellent"
    assert credit_band(79.99) == "Good"
    assert credit_band(60) == "Good"
    assert credit_band(59.99) == "Fair"
    assert credit_band(40) == "Fair"
    assert credit_band(39.99) == "Poor"


def test_parse_yield_average_ignores_bad_values() -> None:
    assert parse_yield_average("100, bad, 80") == 90.0
    assert parse_yield_average("") == 0.0


def test_score_farmer_rule_based_returns_valid_structure() -> None:
    result = score_farmer_rule_based(_sample_row(), farmer_budget_score=65)

    assert 0 <= result.score <= 100
    assert result.band in {"Excellent", "Good", "Fair", "Poor"}
    assert "cs_farmer_budget" in result.components
    assert "cs_farmer_budget" in result.contributions
    assert "cs_farmer_budget" in result.opportunity_gap
