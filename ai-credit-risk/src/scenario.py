"""
src/scenario.py

What-If Scenario Simulator. Deliberately contains NO separate scoring
formula - it calls src.prediction.predict_application twice (once for
the original inputs, once for the modified inputs), using exactly the
same trained model and preprocessing pipeline as every other part of the
platform.
"""

from src.prediction import predict_application
import config


def run_scenario(original_inputs: dict, modified_inputs: dict) -> dict:
    merged_after = {**original_inputs, **modified_inputs}

    before = predict_application(original_inputs, explain=False)
    after = predict_application(merged_after, explain=False)

    changed_fields = []
    for key, new_val in modified_inputs.items():
        old_val = original_inputs.get(key)
        if old_val != new_val:
            changed_fields.append({"field": key, "before": old_val, "after": new_val})

    return {
        "before": {
            "risk_score": before["risk_score"],
            "probability_of_default": before["probability_of_default"],
            "risk_category": before["risk_category"],
            "recommendation": before["recommendation"],
        },
        "after": {
            "risk_score": after["risk_score"],
            "probability_of_default": after["probability_of_default"],
            "risk_category": after["risk_category"],
            "recommendation": after["recommendation"],
        },
        "change": {
            "risk_score_delta": after["risk_score"] - before["risk_score"],
            "probability_delta": round(
                after["probability_of_default"] - before["probability_of_default"], 4
            ),
            "recommendation_changed": before["recommendation"] != after["recommendation"],
        },
        "changed_fields": changed_fields,
        "disclaimer": config.SCENARIO_DISCLAIMER,
    }
