from atlas_one_step.selection.select_target import select_target
def test_sel(): assert select_target([{"pathology_score":2},{"pathology_score":1}])["pathology_score"]==1
