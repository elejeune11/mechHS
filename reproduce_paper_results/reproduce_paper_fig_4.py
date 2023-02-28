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

#############################################################################################

ss = np.loadtxt(str(save_path) + "/ss_performance_2.txt")
spearmanr_ss_2 = ss[1]
ss = np.loadtxt(str(save_path) + "/ss_performance_3.txt")
spearmanr_ss_3 = ss[1]
ss = np.loadtxt(str(save_path) + "/ss_performance_4.txt")
spearmanr_ss_4 = ss[1]
input_data_ss_5 = np.loadtxt(str(ss_path) + "/ssb_results_5.txt")
spearmanr_ss_5 = ss[1]
ss = np.loadtxt(str(save_path) + "/ss_performance_6.txt")
spearmanr_ss_6 = ss[1]
ss = np.loadtxt(str(save_path) + "/ss_performance_7.txt")
spearmanr_ss_7 = ss[1]
ss = np.loadtxt(str(save_path) + "/ss_performance_8.txt")
spearmanr_ss_8 = ss[1]
ss = np.loadtxt(str(save_path) + "/ss_performance_9.txt")
spearmanr_ss_9 = ss[1]
ss = np.loadtxt(str(save_path) + "/ss_performance_10.txt")
spearmanr_ss_10 = ss[1]
spearmanr_ss = [spearmanr_ss_2, spearmanr_ss_3, spearmanr_ss_4, spearmanr_ss_5, spearmanr_ss_6, spearmanr_ss_7, spearmanr_ss_8, spearmanr_ss_9, spearmanr_ss_10]

accuracy_spearmanr_ss_3_ensemble = np.loadtxt(str(save_path) + "/ss_performance_3_ensemble.txt")
accuracy_spearmanr_ss_4_ensemble = np.loadtxt(str(save_path) + "/ss_performance_4_ensemble.txt")
accuracy_spearmanr_ss_5_ensemble = np.loadtxt(str(save_path) + "/ss_performance_5_ensemble.txt")
accuracy_spearmanr_ss_6_ensemble = np.loadtxt(str(save_path) + "/ss_performance_6_ensemble.txt")
accuracy_spearmanr_ss_7_ensemble = np.loadtxt(str(save_path) + "/ss_performance_7_ensemble.txt")
accuracy_spearmanr_ss_8_ensemble = np.loadtxt(str(save_path) + "/ss_performance_8_ensemble.txt")
accuracy_spearmanr_ss_9_ensemble = np.loadtxt(str(save_path) + "/ss_performance_9_ensemble.txt")
accuracy_spearmanr_ss_10_ensemble = np.loadtxt(str(save_path) + "/ss_performance_10_ensemble.txt")
spearmanr_345678910_true_ensemble = [np.mean(accuracy_spearmanr_ss_3_ensemble[:, 1]), np.mean(accuracy_spearmanr_ss_4_ensemble[:, 1]), np.mean(accuracy_spearmanr_ss_5_ensemble[:, 1]), np.mean(accuracy_spearmanr_ss_6_ensemble[:, 1]), np.mean(accuracy_spearmanr_ss_7_ensemble[:, 1]), np.mean(accuracy_spearmanr_ss_8_ensemble[:, 1]), np.mean(accuracy_spearmanr_ss_9_ensemble[:, 1]), np.mean(accuracy_spearmanr_ss_10_ensemble[:, 1])]

spearmanr_rect = np.loadtxt(str(save_path) + "/spearmanr_rect.txt")
spearmanr_lattice = np.loadtxt(str(save_path) + "/spearmanr_lattice.txt")
spearmanr_custom = np.loadtxt(str(save_path) + "/spearmanr_custom.txt")

#############################################################################################
# create plots MUTED!! -- probability of a hash collision vs. distance between inputs
#############################################################################################
fig = plt.figure(figsize=(7.5, 5))
ax = plt.subplot(111)


# first set of plots -- probability of collision vs. distance
# simply supported

def plot_collisions_muted(ax):
    middle_dist_all = []
    match_ratio_all = []
    ix = 0
    for fix_whole_bottom in [True, False]:
        for depth_num in [1, 2, 3, 4, 5]:
            for sensor_num in [2, 3, 4, 5]:
                val_x = middle_dist_rect[ix]
                val_y = match_ratio_rect[ix]
                if fix_whole_bottom:
                    ax.plot(val_x, val_y, "-o", color=(0.5, 0.5, 0.5), markerfacecolor=(0.5, 0.5, 0.5), markeredgecolor="k",linewidth=.5, zorder=0)
                else:
                    ax.plot(val_x, val_y, "-o", color=(0.5, 0.5, 0.5), markerfacecolor=(0.5, 0.5, 0.5), markeredgecolor="k", linewidth=.5, zorder=0)
                middle_dist_all.append(val_x)
                match_ratio_all.append(val_y)
                ix += 1
    ax.plot(val_x, val_y, "-o", color=(0.5, 0.5, 0.5), markerfacecolor=(0.5, 0.5, 0.5), markeredgecolor="k",linewidth=.5, zorder=0, label="individual")
    middle_dist_all = np.asarray(middle_dist_all)
    match_ratio_all = np.asarray(match_ratio_all)
    middle_dist_mean = np.mean(middle_dist_all, axis=0)
    match_ratio_mean = np.mean(match_ratio_all, axis=0)
    ax.plot(middle_dist_mean, match_ratio_mean, "k-o", markerfacecolor=(1.0, 0.0, 0.0), markeredgecolor="k", markersize=15, markeredgewidth=3, linewidth=3, zorder=500, label="mean")
    return middle_dist_mean, match_ratio_mean


middle_dist_mean, match_ratio_mean = plot_collisions_muted(ax)
inset_ax = ax.inset_axes([0.59, 0.59, 0.38, 0.38])
inset_ax.plot(middle_dist_mean, match_ratio_mean, "k-o", markerfacecolor=(1.0, 0.0, 0.0), markeredgecolor="k", markersize=10, markeredgewidth=2, linewidth=3, zorder=500)
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
ax.set_ylabel(r"$p_{collision}$")
plt.xlim((np.min(middle_dist_mean) - 0.01, np.max(middle_dist_mean) + 0.01))
plt.ylim((0.0, 0.18))
plt.title("probability of hash collisions")
plt.savefig(str(fig_path) + "/collision_curve_muted.png")
plt.savefig(str(fig_path) + "/collision_curve_muted.eps")


#############################################################################################
# create plots -- probability of a hash collision vs. distance between inputs
#############################################################################################
fig = plt.figure(figsize=(7.5, 5))
ax = plt.subplot(111)


# first set of plots -- probability of collision vs. distance
# simply supported

def plot_collisions(ax):
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
    ax.plot(middle_dist_mean, match_ratio_mean, "k-o", markerfacecolor=(1.0, 0.0, 0.0), markeredgecolor="k", markersize=15, markeredgewidth=3, linewidth=3, zorder=500, label="mean")
    return middle_dist_mean, match_ratio_mean


middle_dist_mean, match_ratio_mean = plot_collisions(ax)
inset_ax = ax.inset_axes([0.59, 0.59, 0.38, 0.38])
inset_ax.plot(middle_dist_mean, match_ratio_mean, "k-o", markerfacecolor=(1.0, 0.0, 0.0), markeredgecolor="k", markersize=10, markeredgewidth=2, linewidth=3, zorder=500)
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
ax.set_ylabel(r"$p_{collision}$")
plt.xlim((np.min(middle_dist_mean) - 0.01, np.max(middle_dist_mean) + 0.01))
plt.ylim((0.0, 0.18))
plt.title("probability of hash collisions")
plt.savefig(str(fig_path) + "/collision_curve.png")
plt.savefig(str(fig_path) + "/collision_curve.eps")



#############################################################################################
# create plots -- accuracy vs. spearman rho
#############################################################################################
fig = plt.figure(figsize=(7.5, 5))
ax = plt.subplot(111)


def plot_rho_sensor_num(ax):
    mst = 8
    msr = 12
    sensor_num_ss = [2, 3, 4, 5, 6, 7, 8, 9, 10]
    spearmanr_ss = [spearmanr_ss_2, spearmanr_ss_3, spearmanr_ss_4, spearmanr_ss_5, spearmanr_ss_6, spearmanr_ss_7, spearmanr_ss_8, spearmanr_ss_9, spearmanr_ss_10]
    ax.plot(sensor_num_ss, spearmanr_ss, "H", markersize=msr, markeredgecolor=(0.25, 0.25, 1.0), color=(0.75, 0.75, 1.0), zorder=80)
    for kk in range(0, 9):
        ax.plot(sensor_num_ss[kk], spearmanr_ss[kk], marker="$S%i$" % (kk + 2), markersize=mst, color=(0.25, 0.25, 1.0), zorder=100)
    ax.scatter(100, 100, marker="$S$", s=mst * 5, color=(0.25, 0.25, 1.0), label="simply supported")
    sensor_num_ensemble = [3, 4, 5, 6, 7, 8, 9, 10]
    spearmanr_345678910_true_ensemble = [np.mean(accuracy_spearmanr_ss_3_ensemble[:, 1]), np.mean(accuracy_spearmanr_ss_4_ensemble[:, 1]), np.mean(accuracy_spearmanr_ss_5_ensemble[:, 1]), np.mean(accuracy_spearmanr_ss_6_ensemble[:, 1]), np.mean(accuracy_spearmanr_ss_7_ensemble[:, 1]), np.mean(accuracy_spearmanr_ss_8_ensemble[:, 1]), np.mean(accuracy_spearmanr_ss_9_ensemble[:, 1]), np.mean(accuracy_spearmanr_ss_10_ensemble[:, 1])]
    ax.plot(sensor_num_ensemble, spearmanr_345678910_true_ensemble, "H", markersize=msr, markeredgecolor=(0.25, 0.25, 0.5), color=(0.75, 0.75, 1.0), zorder=80)
    for kk in range(0, 8):
        ax.plot(sensor_num_ensemble[kk], spearmanr_345678910_true_ensemble[kk], marker="$E%i$" % (kk + 3), markersize=mst, color=(0.25, 0.25, 0.5), zorder=100)
    ax.scatter(100, 100, marker="$E$", s=mst * 5, color=(0.25, 0.25, 0.5), label="ss ensemble")
    ix = 0
    for fix_whole_bottom in [True, False]:
        for depth_num in [1, 2, 3, 4, 5]:
            for sensor_num in [2, 3, 4, 5]:
                if fix_whole_bottom:
                    ax.plot(sensor_num, spearmanr_rect[ix], "D", color=(sensor_num / 5.0, sensor_num / 5.0, sensor_num / 5.0), markersize=10, markeredgecolor=plt.cm.coolwarm((depth_num - 1) / 4), markeredgewidth=3)
                    ax.plot(sensor_num, spearmanr_rect[ix], marker="$R%i$" % (sensor_num), color=(0, 0, 0), markersize=8)
                else:
                    ax.plot(sensor_num, spearmanr_rect[ix], "s", color=(sensor_num / 5.0, sensor_num / 5.0, sensor_num / 5.0), markersize=10, markeredgecolor=plt.cm.coolwarm((depth_num - 1) / 4), markeredgewidth=3)
                    ax.plot(sensor_num, spearmanr_rect[ix], marker="$R%i$" % (sensor_num), color=(0, 0, 0), markersize=8)
                ix += 1
    ax.scatter(100, 100, marker="$R$", s=mst * 5, color=(0, 0, 0), label="rectangle")
    ax.plot(100, 100, marker="s", linestyle="None", color=(0.75, 0.75, 0.75), markersize=10, zorder=100, label="free btm")
    ax.plot(100, 100, marker="D", linestyle="None", color=(0.75, 0.75, 0.75), markersize=10, zorder=100, label="fixed btm")
    ax.plot(100, 100, marker="s", linestyle="None", color=(0.5, 0.5, 0.5), markersize=10, zorder=100, markeredgecolor=plt.cm.coolwarm((1 - 1) / 4), markeredgewidth=3, label="depth = 1.0")
    ax.plot(100, 100, marker="s", linestyle="None", color=(0.5, 0.5, 0.5), markersize=10, zorder=100, markeredgecolor=plt.cm.coolwarm((2 - 1) / 4), markeredgewidth=3, label="depth = 2.5")
    ax.plot(100, 100, marker="s", linestyle="None", color=(0.5, 0.5, 0.5), markersize=10, zorder=100, markeredgecolor=plt.cm.coolwarm((3 - 1) / 4), markeredgewidth=3, label="depth = 5.0")
    ax.plot(100, 100, marker="s", linestyle="None", color=(0.5, 0.5, 0.5), markersize=10, zorder=100, markeredgecolor=plt.cm.coolwarm((4 - 1) / 4), markeredgewidth=3, label="depth = 10.0")
    ax.plot(100, 100, marker="s", linestyle="None", color=(0.5, 0.5, 0.5), markersize=10, zorder=100, markeredgecolor=plt.cm.coolwarm((5 - 1) / 4), markeredgewidth=3, label="depth = 20.0")
    lattice_sensor_num = [3, 4, 5]
    ax.plot(lattice_sensor_num, spearmanr_lattice, "o", markersize=msr, markeredgecolor=(1, 0.25, 0.25), color=(0.75, 0.75, 0.75), zorder=80)
    for kk in range(0, 3):
        ax.plot(lattice_sensor_num[kk], spearmanr_lattice[kk], marker="$L%i$" % (kk + 3), markersize=mst, color=(1, 0.25, 0.25), zorder=100)
    ax.scatter(100, 100, marker="$L$", s=mst * 5, color=(1, 0.25, 0.25), label="lattice")
    custom_sensor_num = [3, 3, 5]
    ax.plot(custom_sensor_num, spearmanr_custom, "o", markersize=msr, markeredgecolor=(.5, 0.25, 0.25), color=(0.75, 0.75, 0.75), zorder=80)
    for kk in range(0, 3):
        ax.plot(custom_sensor_num[kk], spearmanr_custom[kk], marker="$C%i$" % (kk + 1), markersize=mst, color=(.5, 0.25, 0.25), zorder=100)
    ax.scatter(100, 100, marker="$C$", s=mst * 5, color=(0.5, 0.25, 0.25), label="custom")
    return


plot_rho_sensor_num(ax)
plt.ylabel(r"Spearman's $\rho$ correlation")
plt.xlabel("number of sensors")
plt.xlim((1.7, 10.25))
plt.ylim((0.3, 0.85))

box = ax.get_position()
ax.set_position([box.x0, box.y0, box.width * 0.8, box.height])

# Put a legend to the right of the current axis
ax.legend(loc='center left', bbox_to_anchor=(1, 0.5))
plt.grid(True)
plt.title("correlation between inputs and sensor readouts")

plt.savefig(str(fig_path) + "/rho_sensor_num_plot.png")
plt.savefig(str(fig_path) + "/rho_sensor_num_plot.eps")


aa = 44