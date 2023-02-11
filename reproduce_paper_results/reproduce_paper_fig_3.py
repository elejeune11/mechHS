import matplotlib.pyplot as plt
import numpy as np
import os
from pathlib import Path
import reproduce_figure_fcns as rff

plt.rcParams['text.usetex'] = True


#############################################################################################
# import relevant data
#############################################################################################
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

all_loads = np.load(str(save_path) + "/all_loads.npy")

# normalize all_loads
val_list = []
for kk in range(0, len(all_loads)):
    for jj in range(0, all_loads[kk].shape[0]):
        val_list.append(np.sqrt(np.sum(all_loads[kk][jj, :] ** 2.0)))
        all_loads[kk][jj, :] = all_loads[kk][jj, :] / np.sqrt(np.sum(all_loads[kk][jj, :] ** 2.0))

#############################################################################################
# visualize all loads
#############################################################################################
fig, axs = plt.subplots(4, 5, figsize=(10, 5))
num_examples = 20
for kk in range(0, num_examples):
    for jj in range(0, 20):
        row_ix = int(np.floor(jj / 5))
        col_ix = jj % 5
        axs[row_ix, col_ix].plot(all_loads[kk][jj, :], linewidth=0.5, color=plt.cm.coolwarm(kk / 19))
        axs[row_ix, col_ix].set_ylim((-0.085, 0.03))
        if kk == 0:
            axs[row_ix, col_ix].text(0, .02, "%i" % (jj + 1), bbox=dict(boxstyle="circle", fc="lightgrey"))
            axs[row_ix, col_ix].axis("off")


plt.savefig(str(fig_path) + "/all_loads.png")
plt.savefig(str(fig_path) + "/all_loads.pdf")


#############################################################################################
# visualize all devices
#############################################################################################

#############################################################################################
# simply supported (3b)
#############################################################################################
x0 = 0
y0 = 0
s = 1
r = 1
x1 = 11
y1 = y0
fig, axs = plt.subplots(2, 5, figsize=(10, 1.8))
for ix in range(0, 9):
    num_segs = ix + 1
    row_ix = int(np.floor(ix / 5.0))
    col_ix = ix % 5
    ax = axs[row_ix, col_ix]
    rff.composite_beam(ax, x0, y0, x1, s, num_segs)
    ax.text(-2.5, 0, "S%i" % (num_segs + 1), bbox=dict(boxstyle="circle", fc="lightgrey"))
    ax.set_aspect("equal")
    ax.axis("off")
axs[1, 4].axis("off")

plt.savefig(str(fig_path) + "/all_ss.png")
plt.savefig(str(fig_path) + "/all_ss.pdf")

#############################################################################################
# rectangular (rectangle 3c)
#############################################################################################
plt.figure()
ax = plt.gca()
space = 3
x_ll = 0
y_ll = 0
wid = 10
y_d1 = y_ll
dep_list = [1, 2.5, 5.0, 10.0, 20.0]
for dep_ix in range(0, len(dep_list)):
    dep = dep_list[dep_ix]
    x_d1 = x_ll
    y_d1 = y_d1 - dep - space - 3
    for num_sensors in [2, 3, 4, 5]:
        lw_rect = 2
        lc = plt.cm.coolwarm(dep_ix / (len(dep_list) - 1))
        val = (num_sensors) / 5.0
        fc = (val, val, val)
        x_new, y_new = rff.rectangular_device(ax, x_d1, y_d1, wid, dep, num_sensors, lw_rect, lc, fc)
        x_d1 = x_new + space

ax.set_aspect("equal")
ax.axis("off")

plt.savefig(str(fig_path) + "/all_rectangle.png")
plt.savefig(str(fig_path) + "/all_rectangle.pdf")

#############################################################################################
# note: lattice + custom will be created in inskscape by filling in pngs (3d)
#############################################################################################


twaa = 44