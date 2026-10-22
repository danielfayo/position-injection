import numpy as np

from src.analysis import analyse_model
from src.simulate_analysis import SHAPES, simulate


def verdict_for(shape: str, seed: int = 1) -> str:
    rng = np.random.default_rng(seed)
    return analyse_model(simulate(SHAPES[shape], rng), margin=0.10, n_boot=300, rng=rng)["verdict"]


def test_recognises_each_simulated_shape():
    assert verdict_for("U-shaped") == "U-shaped"
    assert verdict_for("rising") == "rising"
    assert verdict_for("falling") == "falling"
    assert verdict_for("flat") == "flat"


def test_wordings_stuck_at_zero_are_excluded():
    rng = np.random.default_rng(2)
    data = simulate(SHAPES["rising"], rng)
    data.loc[data["wording"] == "A", "success"] = 0.0
    result = analyse_model(data, margin=0.10, n_boot=300, rng=rng)
    assert result["excluded_wordings"] == ["A"]
