from pathlib import Path
from ..viz.paper_figures import make_plot
def main():
 Path("paper_artifacts/figures").mkdir(parents=True,exist_ok=True); make_plot(out="paper_artifacts/figures/main.png")
if __name__=="__main__": main()
