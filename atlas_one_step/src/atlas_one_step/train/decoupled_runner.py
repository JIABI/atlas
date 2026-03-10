from .base_runner import BaseRunner


class DecoupledRunner(BaseRunner):
    def __init__(self, cfg):
        super().__init__(cfg, mode="atlas_guided_decoupled")

    def run(self, exp_id: str = "EXP-B4"):
        return super().run(exp_id=exp_id)
