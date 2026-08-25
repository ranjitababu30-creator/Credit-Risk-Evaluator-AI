from src.prediction import ArtifactBundle
from src.scenario import run_scenario


def base_inputs():
    bundle = ArtifactBundle.get()
    values = {}
    for col in bundle.numeric_columns:
        values[col] = 1000
    for col in bundle.categorical_columns:
        values[col] = "other"
    return values


def test_scenario_returns_before_after_and_change():
    original = base_inputs()
    modified = dict(original)
    if "income" in modified:
        modified["income"] = float(modified["income"]) * 3
    else:
        # modify the first numeric field available to still exercise the logic
        bundle = ArtifactBundle.get()
        if bundle.numeric_columns:
            first = bundle.numeric_columns[0]
            modified[first] = float(modified.get(first, 1000)) * 3

    result = run_scenario(original, modified)
    assert "before" in result and "after" in result
    assert "risk_score_delta" in result["change"]
    assert isinstance(result["changed_fields"], list)


def test_scenario_no_changes_gives_zero_delta():
    original = base_inputs()
    result = run_scenario(original, {})
    assert result["change"]["risk_score_delta"] == 0
    assert result["changed_fields"] == []
