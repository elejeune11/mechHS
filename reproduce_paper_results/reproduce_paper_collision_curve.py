import matplotlib.pyplot as plt
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

middle_dist_rect = np.load(str(save_path) + "/middle_dist_rect.npy")
match_ratio_rect = np.load(str(save_path) + "/match_ratio_rect.npy")

middle_dist_ss_2 = np.loadtxt(str(save_path) + "/middle_distance_ss_2.txt")
match_ratio_ss_2 = np.loadtxt(str(save_path) + "/match_ratio_ss_2.txt")
middle_dist_ss_3_ensemble = np.loadtxt(str(save_path) + "/middle_dist_ss_3_ensemble.txt")
match_ratio_ss_3_ensemble = np.loadtxt(str(save_path) + "/match_ratio_ss_3_ensemble.txt")
#############################################################################################
#############################################################################################


fig = plt.figure(figsize=(7.5, 5))
ax = plt.subplot(111)


def plot_collisions(ax):
    ax.plot(middle_dist_ss_2, match_ratio_ss_2, "o", color=(0.75, 0.75, 0.75), markersize=20, markeredgecolor="k", zorder=500)
    ax.plot(middle_dist_ss_2, match_ratio_ss_2, linestyle="None", marker="$SS$", color=(0, 0, 0), markersize=11, label="simply supported", zorder=600)
    ax.plot(middle_dist_ss_2, match_ratio_ss_2, "k-", markersize=10, zorder=0)
    middle_dist_all = []
    match_ratio_all = []
    ix = 0
    for fix_whole_bottom in [True, False]:
        for depth_num in [1, 2, 3, 4, 5]:
            for sensor_num in [2, 3, 4, 5]:
                val_x = middle_dist_rect[ix]
                val_y = match_ratio_rect[ix]
                if fix_whole_bottom:
                    ax.plot(val_x, val_y, "-", color=plt.cm.coolwarm((depth_num - 1) / 4), linewidth=.5, zorder=0)
                    # ax.plot(val_x, val_y, marker="s", linestyle="None", color=plt.cm.plasma(0.5 + (sensor_num - 2) / 6), markersize=8 + depth_num, markeredgecolor="k", markeredgewidth=1.5, zorder=100)
                    # ax.plot(val_x, val_y, marker="$R%i$" % (sensor_num), linestyle="None", color=(0, 0, 0), markersize=6 + depth_num, zorder=200)
                    ax.plot(val_x, val_y, "D", color=(sensor_num / 5.0, sensor_num / 5.0, sensor_num / 5.0), linestyle="None", markersize=10, markeredgecolor=plt.cm.coolwarm((depth_num - 1) / 4), markeredgewidth=3, zorder=100)
                    ax.plot(val_x, val_y, marker="$R%i$" % (sensor_num), linestyle="None", color=(0, 0, 0), markersize=8, zorder=200)
                else:
                    ax.plot(val_x, val_y, "-", color=plt.cm.coolwarm((depth_num - 1) / 4), linewidth=.5, zorder=0)
                    # ax.plot(val_x, val_y, marker="s", linestyle="None", color=plt.cm.plasma(0.5 + (sensor_num - 2) / 6), markersize=8 + depth_num, zorder=100)
                    # ax.plot(val_x, val_y, marker="$R%i$" % (sensor_num), linestyle="None", color=(0, 0, 0), markersize=6 + depth_num, zorder=200)
                    ax.plot(val_x, val_y, "s", color=(sensor_num / 5.0, sensor_num / 5.0, sensor_num / 5.0), linestyle="None", markersize=10, markeredgecolor=plt.cm.coolwarm((depth_num - 1) / 4), markeredgewidth=3, zorder=100)
                    ax.plot(val_x, val_y, marker="$R%i$" % (sensor_num), color=(0, 0, 0), linestyle="None", markersize=8, zorder=200)
                middle_dist_all.append(val_x)
                match_ratio_all.append(val_y)
                ix += 1
    ax.plot(100, 100, marker="$R2$", linestyle="None", color=(0, 0, 0), markersize=6 + depth_num, zorder=200, label="2 sensors")
    ax.plot(100, 100, marker="$R3$", linestyle="None", color=(0, 0, 0), markersize=6 + depth_num, zorder=200, label="3 sensors")
    ax.plot(100, 100, marker="$R4$", linestyle="None", color=(0, 0, 0), markersize=6 + depth_num, zorder=200, label="4 sensors")
    ax.plot(100, 100, marker="$R5$", linestyle="None", color=(0, 0, 0), markersize=6 + depth_num, zorder=200, label="5 sensors")
    ax.plot(100, 100, marker="s", linestyle="None", color=(0.75, 0.75, 0.75), markersize=10, zorder=100, label="free btm")
    ax.plot(100, 100, marker="D", linestyle="None", color=(0.75, 0.75, 0.75), markersize=10, zorder=100, label="fixed btm")
    ax.plot(100, 100, marker="s", linestyle="None", color=(0.5, 0.5, 0.5), markersize=10, zorder=100, markeredgecolor=plt.cm.coolwarm((1 - 1) / 4), markeredgewidth=3, label="depth = 1.0")
    ax.plot(100, 100, marker="s", linestyle="None", color=(0.5, 0.5, 0.5), markersize=10, zorder=100, markeredgecolor=plt.cm.coolwarm((2 - 1) / 4), markeredgewidth=3, label="depth = 2.5")
    ax.plot(100, 100, marker="s", linestyle="None", color=(0.5, 0.5, 0.5), markersize=10, zorder=100, markeredgecolor=plt.cm.coolwarm((3 - 1) / 4), markeredgewidth=3, label="depth = 5.0")
    ax.plot(100, 100, marker="s", linestyle="None", color=(0.5, 0.5, 0.5), markersize=10, zorder=100, markeredgecolor=plt.cm.coolwarm((4 - 1) / 4), markeredgewidth=3, label="depth = 10.0")
    ax.plot(100, 100, marker="s", linestyle="None", color=(0.5, 0.5, 0.5), markersize=10, zorder=100, markeredgecolor=plt.cm.coolwarm((5 - 1) / 4), markeredgewidth=3, label="depth = 20.0")

    middle_dist_all = np.asarray(middle_dist_all)
    match_ratio_all = np.asarray(match_ratio_all)
    middle_dist_mean = np.mean(middle_dist_all, axis=0)
    match_ratio_mean = np.mean(match_ratio_all, axis=0)
    # ax.plot(middle_dist_mean, match_ratio_mean, "k-o", markerfacecolor=(1.0, 0.0, 0.0), markeredgecolor="k", markersize=15, markeredgewidth=3, linewidth=3, zorder=500, label="mean")
    ax.plot(100, 100, "k-o", markerfacecolor=(1.0, 0.0, 0.0), markeredgecolor="k", markersize=15, markeredgewidth=3, linewidth=3, zorder=500, label="mean")
    return middle_dist_mean, match_ratio_mean


middle_dist_mean, match_ratio_mean = plot_collisions(ax)
inset_ax = ax.inset_axes([0.625, 0.625, 0.34, 0.34])
inset_ax.plot(middle_dist_mean, match_ratio_mean, "k-o", markerfacecolor=(1.0, 0.0, 0.0), markeredgecolor="k", markersize=10, markeredgewidth=2, linewidth=3, zorder=500, label="mean")
inset_ax.grid(True)
inset_ax.set_xticks([])
inset_ax.set_yticks([])
inset_ax.set_xlabel(r"$||w_i - w_j||_{\infty}$")
inset_ax.set_ylabel(r"$p_{collision}$")

box = ax.get_position()
ax.set_position([box.x0, box.y0, box.width * 0.8, box.height])

# Put a legend to the right of the current axis
ax.legend(loc='center left', bbox_to_anchor=(1, 0.5))

plt.ticklabel_format(style='sci', axis='x', scilimits=(0,0))
plt.grid(True)
ax.set_xlabel(r"$||w_i - w_j||_{\infty}$")
ax.set_ylabel(r"$p_{collision} = Pr \bigg[ \, ||h_i - h_j||_{\infty} < 0.01 \, \bigg]$")
plt.xlim((np.min(middle_dist_mean) - 0.01, np.max(middle_dist_mean) + 0.01))
plt.ylim((0.0, 0.4))
plt.title("probability of hash collisions")
plt.savefig(str(fig_path) + "/collision_curve.png")
plt.savefig(str(fig_path) + "/collision_curve.eps")

aa = 44