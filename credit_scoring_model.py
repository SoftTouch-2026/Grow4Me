import pandas as pd

from constants import COMPONENT_LABELS, WEIGHTS, WORKBOOK_PATH
from explainability import explain_top_shap_features, generate_shap_outputs
from ml_model import prepare_farmer_feature_row, predict_with_all_models, save_trained_models, train_regression_models
from rule_based_model import credit_band, clamp, score_farmer_rule_based, top_helping_and_dragging, what_if_suggestion


def load_data() -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Load the workbook sheets used by the rule-based and ML pipelines."""
    farmer_df = pd.read_excel(WORKBOOK_PATH, sheet_name="🌾 Farmer Data", header=2)
    ml_df = pd.read_excel(WORKBOOK_PATH, sheet_name="🤖 ML-Ready Data", header=1)
    component_df = pd.read_excel(WORKBOOK_PATH, sheet_name="🔢 Component Scores", header=2)
    return farmer_df, ml_df, component_df


def print_rule_based_section(example_farmer: pd.Series, component_df: pd.DataFrame) -> tuple[float, str, dict]:
    component_row = component_df.loc[
        component_df["Farmer Id"] == int(example_farmer["Farmer Id"])
    ].iloc[0]
    result = score_farmer_rule_based(
        example_farmer,
        farmer_budget_score=float(component_row["Farmer Budget"]),
    )
    workbook_score = float(example_farmer["Credit Score"])

    print("PART 1 - RULE-BASED SCORING ENGINE")
    print(f"Example Farmer: {example_farmer['Farmer Name']} (ID: {int(example_farmer['Farmer Id'])})")
    print(f"Rule-based Credit Score: {result.score:.2f}")
    print(f"Rule-based Category: {result.band}")
    print(f"Workbook Provided Credit Score: {workbook_score:.2f}")

    score_gap = result.score - workbook_score
    if abs(score_gap) > 0.01:
        has_budget_component = "Farmer Budget" in component_df.columns
        print(
            "Note: Formula-derived score and workbook score differ."
            f"Delta={score_gap:+.2f}."
        )
        if has_budget_component:
            print(
                "Likely reason: workbook component recipe includes 'Farmer Budget', "
                "but your provided 20-component manual does not include that term."
            )

    print("\nComponent contribution breakdown (component_score x weight):")
    for key, weight in WEIGHTS.items():
        print(
            f"- {COMPONENT_LABELS[key]:32s} score={result.components[key]:6.2f}, weight={weight:.2f}, contribution={result.contributions[key]:6.2f}"
        )

    helping, dragging = top_helping_and_dragging(result, n=3)

    print("\nTop 3 factors helping the score:")
    for key, value in helping:
        print(f"- {COMPONENT_LABELS[key]} contributed +{value:.2f} points.")

    print("\nTop 3 factors dragging the score down (largest unrealized weighted points):")
    for key, gap in dragging:
        print(f"- {COMPONENT_LABELS[key]} is leaving about {gap:.2f} points on the table.")

    suggestion_line, _ = what_if_suggestion(result)
    print("\nWhat-if improvement suggestion:")
    print(f"- {suggestion_line}")

    return result.score, result.band, result.components


def print_ml_section(ml_df: pd.DataFrame, farmer_id: int) -> tuple[float, str]:
    artifacts = train_regression_models(ml_df)
    saved = save_trained_models(artifacts)


    print("PART 2 - ML MODELING (REGRESSION)")
    print("Model evaluation on test set:")

    for model_name in ["Linear Regression", "XGBoost Regressor"]:
        print(
            f"- {model_name:20s} MAE={artifacts.metrics[model_name]['MAE']:.3f}  "
            f"RMSE={artifacts.metrics[model_name]['RMSE']:.3f}  R2={artifacts.metrics[model_name]['R2']:.3f}"
        )

    x_example = prepare_farmer_feature_row(ml_df, farmer_id, artifacts.feature_cols)
    predictions = predict_with_all_models(artifacts, x_example)

    chosen_model_name = artifacts.best_model_name
    chosen_pred = clamp(predictions[chosen_model_name])
    chosen_band = credit_band(chosen_pred)

    print(f"\nSelected model for explanation: {chosen_model_name}")
    print(f"Predicted score for same example farmer: {chosen_pred:.2f}")
    print(f"Predicted category: {chosen_band}")

    xgb_model = artifacts.models["XGBoost Regressor"]
    example_shap = generate_shap_outputs(xgb_model, artifacts.X_test, x_example)
    shap_lines = explain_top_shap_features(example_shap, x_example, top_n=3)

    print("\nSHAP outputs saved:")
    print("- shap_summary_plot.png")
    print("- shap_single_farmer_waterfall.png")

    print("\nModel artifacts saved:")
    print(f"- {saved['linear_regression']}")
    print(f"- {saved['xgboost_regressor']}")
    print(f"- {saved['metadata']}")

    print("\nTop 3 ML factors (SHAP, plain English):")
    for line in shap_lines:
        print(f"- {line}")

    return chosen_pred, chosen_band


def print_comparison_section(rule_score: float, rule_band: str, ml_score: float, ml_band: str) -> None:
    diff = ml_score - rule_score
    agreement = "agree" if rule_band == ml_band else "diverge"

    print("PART 3 - RULE VS ML COMPARISON")
    print(f"Rule-based score/category: {rule_score:.2f} / {rule_band}")
    print(f"ML-predicted score/category: {ml_score:.2f} / {ml_band}")
    print(f"Difference (ML - Rule): {diff:+.2f} points")

    if agreement == "agree":
        print("Interpretation: Rule-based and ML assessments agree on credit band for this farmer.")
    else:
        print("Interpretation: Rule-based and ML assessments diverge on credit band.")

    print(
        "Why this can happen: the rule engine uses fixed expert weights, while ML learns patterns and interactions "
        "from historical data (including non-linear effects captured by XGBoost)."
    )


def main() -> None:
    farmer_df, ml_df, component_df = load_data()
    example_farmer = farmer_df.iloc[5]
    example_farmer_id = int(example_farmer["Farmer Id"])

    rule_score, rule_band, _ = print_rule_based_section(example_farmer, component_df)
    ml_score, ml_band = print_ml_section(ml_df, example_farmer_id)
    print_comparison_section(rule_score, rule_band, ml_score, ml_band)


if __name__ == "__main__":
    main()
