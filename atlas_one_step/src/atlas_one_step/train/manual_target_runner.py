from .base_runner import BaseRunner


class ManualTargetRunner(BaseRunner):
    def __init__(self, cfg):
        super().__init__(cfg, mode="manual_target")

    def run(self, exp_id: str = "EXP-B2"):
        return super().run(exp_id=exp_id)
