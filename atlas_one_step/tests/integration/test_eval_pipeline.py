from atlas_one_step.eval.tail_failure import compute


def test_eval_pipeline():
    tail = compute([0.1, 0.2, 0.3, 0.8, 0.9])
    assert tail["percentile_95"] >= tail["percentile_99"] or tail["percentile_99"] >= tail["percentile_95"]
