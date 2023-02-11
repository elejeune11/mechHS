from matplotlib.patches import Circle
from matplotlib.patches import Rectangle
import matplotlib.pyplot as plt
import numpy as np
import os
from pathlib import Path
import reproduce_figure_fcns as rff



def create_folder(folder_path: Path, new_folder_name: str) -> Path:
    """Given a path to a directory and a folder name. Will create a directory in the given directory."""
    new_path = folder_path.joinpath(new_folder_name).resolve()
    if new_path.exists() is False:
        os.mkdir(new_path)
    return new_path


mypath = Path(__file__).resolve().parent
save_path = create_folder(mypath, "analysis_results")
fea_path = create_folder(mypath, "FEA_results_summarized")
ss_path = create_folder(mypath, "SS_results_summarized")
fig_path = create_folder(mypath, "figures")

x0 = 0
y0 = 0
x1 = 10
y1 = 0
h_mean = 2.5
h_range = 1
num_pts = 5


for seed in [1, 2, 3, 4, 5, 6]:
    plt.figure()
    ax = plt.gca()
    rff.plot_load(ax, x0, y0, x1, y1, h_mean, h_range, num_pts, seed)
    ax.set_aspect("equal")
    ax.axis("off")
    plt.savefig(str(fig_path) + "/load%i.png" % (seed))
    plt.savefig(str(fig_path) + "/load%i.pdf" % (seed))

aa = 44
