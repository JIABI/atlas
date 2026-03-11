from atlas_one_step.selection.select_target import select_target


def test_select_target_smoke():
    best = select_target([{"pathology_score": 3.0}, {"pathology_score": 1.0}])
    assert best["pathology_score"] == 1.0
