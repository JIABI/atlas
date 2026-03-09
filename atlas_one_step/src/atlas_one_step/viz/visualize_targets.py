
import matplotlib.pyplot as plt

def make_plot(values=None, out='outputs/plot.png'):
    values = values or [0,1,2]
    plt.figure(); plt.plot(values); plt.tight_layout(); plt.savefig(out); plt.close()
