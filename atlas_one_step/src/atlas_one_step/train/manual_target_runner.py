from .base_runner import BaseRunner
class ManualTargetRunner(BaseRunner):
    def run(self, exp_id="EXP-B2"):
        return super().run(exp_id=exp_id)
