from .base_runner import BaseRunner
class DecoupledRunner(BaseRunner):
    def run(self, exp_id="EXP-B4"):
        return super().run(exp_id=exp_id)
