import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1 import make_axes_locatable
import numpy as np
import os
from pathlib import Path

plt.rcParams['text.usetex'] = True
plt.rcParams.update({'font.size': 12})


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

#############################################################################################

spearmanr_rect = np.loadtxt(str(save_path) + "/spearmanr_rect_L2_norm.txt")
spearmanr_lattice = np.loadtxt(str(save_path) + "/spearmanr_lattice_L2_norm.txt")
spearmanr_custom = np.loadtxt(str(save_path) + "/spearmanr_custom_L2_norm.txt")

accuracy_rect = np.loadtxt(str(save_path) + "/accuracy_rect_L2_norm.txt")
accuracy_lattice = np.loadtxt(str(save_path) + "/accuracy_lattice_L2_norm.txt")
accuracy_custom = np.loadtxt(str(save_path) + "/accuracy_custom_L2_norm.txt")

#############################################################################################
#############################################################################################

#############################################################################################


# create a matrix representation of the spearman's rho for rectangular domains
spearmanr_mat = np.zeros((5, 4))
accuracy_mat = np.zeros((5, 4))
ix = 0
for fix_whole_bottom in [True, False]:
    for depth_num in [0, 1, 2, 3, 4]:
        for sensor_num in [0, 1, 2, 3]:
            if fix_whole_bottom is True:
                spearmanr_mat[depth_num, sensor_num] = spearmanr_rect[ix]
                accuracy_mat[depth_num, sensor_num] = accuracy_rect[ix]
            ix += 1

# scale to make it a rectangular color background
dim_row = 6
dim_col = 20
mat_viz_rho = np.zeros((21 * dim_row, 4 * dim_col))
mat_viz_acc = np.zeros((21 * dim_row, 4 * dim_col))
for kk in range(0, mat_viz_rho.shape[0]):
    for jj in range(0, mat_viz_rho.shape[1]):
        kk_match = kk / dim_row
        jj_match = jj / dim_col + 1.5
        depth_ix = np.argmin(np.abs(np.asarray([1.0, 2.5, 5.0, 10.0, 20.0]) - kk_match))
        sensor_ix = np.argmin(np.abs(np.asarray([2, 3, 4, 5]) - jj_match))
        mat_viz_rho[kk, jj] = spearmanr_mat[depth_ix, sensor_ix]
        mat_viz_acc[kk, jj] = accuracy_mat[depth_ix, sensor_ix]

# organize the rectangle markers to make it easy to plot discretely
depth_rect_fix_true = []
sensors_rect_fix_true = []
spearman_r_fix_true = []
acc_r_fix_true = []
depth_rect_fix_false = []
sensors_rect_fix_false = []
spearman_r_fix_false = []
acc_r_fix_false = []


ix = 0
offset = 0.15

for fix_whole_bottom in [True, False]:
    for depth_num in [1.0, 2.5, 5.0, 10.0, 20.0]:
        for sensor_num in [2, 3, 4, 5]:
            if fix_whole_bottom:
                depth_rect_fix_true.append(depth_num * dim_row)
                sensors_rect_fix_true.append((sensor_num + offset - 1.5) * dim_col)
                spearman_r_fix_true.append(spearmanr_rect[ix])
                acc_r_fix_true.append(accuracy_rect[ix])
            else:
                depth_rect_fix_false.append(depth_num * dim_row)
                sensors_rect_fix_false.append((sensor_num - offset - 1.5) * dim_col)
                spearman_r_fix_false.append(spearmanr_rect[ix])
                acc_r_fix_false.append(accuracy_rect[ix])
            ix += 1


#############################################################################################
#############################################################################################
fig, axs = plt.subplots(3, 2, figsize=(7.5, 10))
#############################################################################################
#############################################################################################


# plot the background and add the black lines
for rho_choose in [True, False]:
    if rho_choose is True:
        mat_viz = mat_viz_rho
        choose_cmap = plt.cm.viridis
        fix_false_color = spearman_r_fix_false
        fix_true_color = spearman_r_fix_true
        fname = "rho"
        ax_row = 0
    else:
        mat_viz = mat_viz_acc
        choose_cmap = plt.cm.inferno
        fix_false_color = acc_r_fix_false
        fix_true_color = acc_r_fix_true
        fname = "acc"
        ax_row = 1

    axs[ax_row, 1].imshow(mat_viz, cmap=choose_cmap, vmin=np.min(mat_viz), vmax=np.max(mat_viz))

    for jj in [2, 3, 4, 5]:
        ix_mat = int((jj - 1.5) * dim_col)
        axs[ax_row, 1].plot([ix_mat, ix_mat], [0, mat_viz.shape[0] - 1], "k-")

    for kk in [1.0, 2.5, 5.0, 10.0, 20.0]:
        ix_mat = int(kk * dim_row)
        axs[ax_row, 1].plot([0, mat_viz.shape[1] - 1], [ix_mat, ix_mat], "k-")

    axs[ax_row, 1].set_xlim((0, mat_viz.shape[1]))
    axs[ax_row, 1].set_ylim((0, mat_viz.shape[0]))
    axs[ax_row, 1].set_xlabel("number of sensors")
    axs[ax_row, 1].set_xticks([0.5 * dim_col, 1.5 * dim_col, 2.5 * dim_col, 3.5 * dim_col])
    axs[ax_row, 1].set_yticks([1.0 * dim_row, 2.5 * dim_row, 5.0 * dim_row, 10.0 * dim_row, 20.0 * dim_row])
    axs[ax_row, 1].set_xticklabels(["2", "3", "4", "5"])
    axs[ax_row, 1].set_yticklabels(["1.0", "2.5", "5.0", "10.0", "20.0"])
    axs[ax_row, 1].set_ylabel("depth")
    axs[ax_row, 1].invert_yaxis()
    #############################################################################################
    im = axs[ax_row, 0].imshow(mat_viz, cmap=choose_cmap, vmin=np.min(mat_viz), vmax=np.max(mat_viz))

    for jj in [2, 3, 4, 5]:
        ix_mat = int((jj - 1.5) * dim_col)
        axs[ax_row, 0].plot([ix_mat, ix_mat], [0, mat_viz.shape[0] - 1], "k-", zorder=1)

    for kk in [1.0, 2.5, 5.0, 10.0, 20.0]:
        ix_mat = int(kk * dim_row)
        axs[ax_row, 0].plot([0, mat_viz.shape[1] - 1], [ix_mat, ix_mat], "k-", zorder=2)

    axs[ax_row, 0].scatter(sensors_rect_fix_false, depth_rect_fix_false, s=50, marker="s", c=fix_false_color, cmap=choose_cmap, edgecolors=(0.8, 0.8, 0.8), linewidths=2, vmin=np.min(mat_viz), vmax=np.max(mat_viz), zorder=100)
    axs[ax_row, 0].scatter(sensors_rect_fix_true, depth_rect_fix_true, s=50, marker="D", c=fix_true_color, cmap=choose_cmap, edgecolors=(0.8, 0.8, 0.8), linewidths=2, vmin=np.min(mat_viz), vmax=np.max(mat_viz), zorder=100)

    axs[ax_row, 0].set_xlim((0, mat_viz.shape[1]))
    axs[ax_row, 0].set_ylim((0, mat_viz.shape[0]))
    axs[ax_row, 0].set_xlabel("number of sensors")
    axs[ax_row, 0].set_xticks([0.5 * dim_col, 1.5 * dim_col, 2.5 * dim_col, 3.5 * dim_col])
    axs[ax_row, 0].set_yticks([1.0 * dim_row, 2.5 * dim_row, 5.0 * dim_row, 10.0 * dim_row, 20.0 * dim_row])
    axs[ax_row, 0].set_xticklabels(["2", "3", "4", "5"])
    axs[ax_row, 0].set_yticklabels(["1.0", "2.5", "5.0", "10.0", "20.0"])
    axs[ax_row, 0].set_ylabel("depth")
    axs[ax_row, 0].invert_yaxis()
    divider = make_axes_locatable(axs[ax_row, 0])
    cax = divider.append_axes("left", size="5%", pad=1.25)
    cb = fig.colorbar(im, cax=cax, orientation="vertical")
    if rho_choose:
        cb.set_label(r"Spearman's $\rho$ correlation", rotation=90, labelpad=-50)
    else:
        cb.set_label(r"classification accuracy", rotation=90, labelpad=-45)


plt.tight_layout()
axs[2, 0].axis("off")
axs[2, 1].axis("off")

#################
# set up legend
#################
mst = 12
msr = 12
axs[2, 0].plot([100], [100], "ws", markeredgewidth="2", markeredgecolor=(0, 0, 0), label="rectangle free btm")
axs[2, 0].plot([100], [100], "wD", markeredgewidth="2", markeredgecolor=(0, 0, 0), label="rectangle fixed btm")
axs[2, 0].scatter(100, 100, marker="$ $", s=mst * 5, color=(1, 0.25, 0.25), label=" ")

axs[2, 0].set_xlim((0, 1))
axs[2, 0].set_ylim((0, 1))
axs[2, 0].scatter(100, 100, marker="$L3$", s=mst * 5, color=(1, 0.25, 0.25), label="lattice 3")
axs[2, 0].scatter(100, 100, marker="$L4$", s=mst * 5, color=(1, 0.25, 0.25), label="lattice 4")
axs[2, 0].scatter(100, 100, marker="$L5$", s=mst * 5, color=(1, 0.25, 0.25), label="lattice 5")
axs[2, 0].scatter(100, 100, marker="$C1$", s=mst * 5, color=(0.5, 0.25, 0.25), label="custom 1")
axs[2, 0].scatter(100, 100, marker="$C2$", s=mst * 5, color=(0.5, 0.25, 0.25), label="custom 2")
axs[2, 0].scatter(100, 100, marker="$C3$", s=mst * 5, color=(0.5, 0.25, 0.25), label="custom 3")

plt.figlegend(ncols=3, loc="lower center", bbox_to_anchor=(0.0, 0.22, 1.0, 0.375))

plt.savefig(str(fig_path) + "/figure_6_L2_norm.png")
plt.savefig(str(fig_path) + "/figure_6_L2_norm.eps")
plt.savefig(str(fig_path) + "/figure_6_L2_norm.pdf")


#############################################################################################

#############################################################################################
# create legends + bespoke system color markers
#############################################################################################

plt.figure()

plt.scatter([1, 2, 3], [1, 2, 3], c=spearmanr_lattice, s=100, cmap=plt.cm.viridis, vmin=np.min(mat_viz_rho), vmax=np.max(mat_viz_rho))
plt.scatter([4, 5, 6], [4, 5, 6], c=spearmanr_custom, s=100, cmap=plt.cm.viridis, vmin=np.min(mat_viz_rho), vmax=np.max(mat_viz_rho))

plt.scatter([10, 11, 12], [10, 11, 12], c=accuracy_lattice, s=100, cmap=plt.cm.inferno, vmin=np.min(mat_viz_acc), vmax=np.max(mat_viz_acc))
plt.scatter([13, 14, 15], [13, 14, 15], c=accuracy_custom, s=100, cmap=plt.cm.inferno, vmin=np.min(mat_viz_acc), vmax=np.max(mat_viz_acc))

plt.savefig(str(fig_path) + "/figure_6_legend_L2_norm.png")
plt.savefig(str(fig_path) + "/figure_6_legend_L2_norm.eps")
plt.savefig(str(fig_path) + "/figure_6_legend_L2_norm.pdf")


aa = 44
