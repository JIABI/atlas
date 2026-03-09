from ._common import cfg
from ..train.coupled_runner import CoupledRunner
from ..train.decoupled_runner import DecoupledRunner

def main():
 c=cfg(); n=c.train.name
 r=DecoupledRunner(c) if "decoupled" in n else CoupledRunner(c)
 print(r.run())
if __name__=="__main__": main()
