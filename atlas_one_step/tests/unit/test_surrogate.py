import json
from pathlib import Path


def test_surrogate_fit(tmp_path):
    atlas = tmp_path / "atlas.parquet"
    jsonl = tmp_path / "atlas.jsonl"
    rows = [
        {"pathology": {"pathology_score": 1.0, "jacobian_norm": 0.5, "rho_nor": 0.2}, "quality": {"fid": 10.0}},
        {"pathology": {"pathology_score": 2.0, "jacobian_norm": 0.8, "rho_nor": 0.3}, "quality": {"fid": 20.0}},
    ]
    jsonl.write_text("\n".join(json.dumps(r) for r in rows))
    from atlas_one_step.atlas.surrogate import fit_from_atlas

    model = fit_from_atlas(atlas, tmp_path / "out")
    assert "r2" in model
