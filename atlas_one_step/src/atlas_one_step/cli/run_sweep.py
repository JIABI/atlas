from ._common import cfg
from ..atlas.line_sweep import run_line_sweep
from ..atlas.simplex_sweep import run_simplex_sweep
from ..atlas.schedule_sweep import run_schedule_sweep

def main():
 c=cfg(); n=c.atlas.name
 if "simplex" in n: run_simplex_sweep(family=c.target.name)
 elif "schedule" in n: run_schedule_sweep(family=c.target.name)
 else: run_line_sweep(family=c.target.name)

if __name__=="__main__": main()
