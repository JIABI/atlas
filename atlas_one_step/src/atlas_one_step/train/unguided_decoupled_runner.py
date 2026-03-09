from .base_runner import BaseRunner
class UnguidedDecoupledRunner(BaseRunner):
    def run(self, exp_id="EXP-B3"):
        return super().run(exp_id=exp_id)
