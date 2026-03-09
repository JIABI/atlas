from atlas_one_step.models.phi_map import build_phi
def test_phi(): assert build_phi("identity") is not None
