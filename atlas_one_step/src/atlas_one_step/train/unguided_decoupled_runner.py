from .base_runner import BaseRunner


class UnguidedDecoupledRunner(BaseRunner):
    def __init__(self, cfg):
        super().__init__(cfg, mode="unguided_decoupled")

    def run(self, exp_id: str = "EXP-B3"):
        return super().run(exp_id=exp_id)
