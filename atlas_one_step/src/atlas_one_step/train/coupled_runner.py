from .base_runner import BaseRunner
class CoupledRunner(BaseRunner):
    def run(self, exp_id="EXP-B1"):
        return super().run(exp_id=exp_id)
