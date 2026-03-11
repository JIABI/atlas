from .base_runner import BaseRunner


class CoupledRunner(BaseRunner):
    def __init__(self, cfg):
        super().__init__(cfg, mode="coupled")

    def run(self, exp_id: str = "EXP-B1"):
        return super().run(exp_id=exp_id)
