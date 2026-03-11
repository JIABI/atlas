from pathlib import Path
import json


def test_atlas_build_pipeline(tmp_path):
    from atlas_one_step.atlas.aggregate import aggregate

    src = tmp_path / "sweeps"
    src.mkdir()
    (src / "a.json").write_text(json.dumps({"pathology": {"pathology_score": 1.0}, "quality": {"fid": 2.0}}))
    n = aggregate(str(src), str(tmp_path / "atlas.parquet"))
    assert n == 1
    assert (tmp_path / "atlas.jsonl").exists()
