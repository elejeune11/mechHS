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

orig = np.loadtxt(str(save_path) + "/orig_performance.txt")
accuracy_orig = orig[0]
spearmanr_orig = orig[1]

ss = np.loadtxt(str(save_path) + "/ss_performance_2.txt")
accuracy_ss_2 = ss[0]
spearmanr_ss_2 = ss[1]
ss = np.loadtxt(str(save_path) + "/ss_performance_3.txt")
accuracy_ss_3 = ss[0]
spearmanr_ss_3 = ss[1]
ss = np.loadtxt(str(save_path) + "/ss_performance_4.txt")
accuracy_ss_4 = ss[0]
spearmanr_ss_4 = ss[1]
input_data_ss_5 = np.loadtxt(str(ss_path) + "/ssb_results_5.txt")
accuracy_ss_5 = ss[0]
spearmanr_ss_5 = ss[1]
ss = np.loadtxt(str(save_path) + "/ss_performance_6.txt")
accuracy_ss_6 = ss[0]
spearmanr_ss_6 = ss[1]
ss = np.loadtxt(str(save_path) + "/ss_performance_7.txt")
accuracy_ss_7 = ss[0]
spearmanr_ss_7 = ss[1]
ss = np.loadtxt(str(save_path) + "/ss_performance_8.txt")
accuracy_ss_8 = ss[0]
spearmanr_ss_8 = ss[1]
ss = np.loadtxt(str(save_path) + "/ss_performance_9.txt")
accuracy_ss_9 = ss[0]
spearmanr_ss_9 = ss[1]
ss = np.loadtxt(str(save_path) + "/ss_performance_10.txt")
accuracy_ss_10 = ss[0]
spearmanr_ss_10 = ss[1]

accuracy_ss = [accuracy_ss_2, accuracy_ss_3, accuracy_ss_4, accuracy_ss_5, accuracy_ss_6, accuracy_ss_7, accuracy_ss_8, accuracy_ss_9, accuracy_ss_10]
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

accuracy_rect = np.loadtxt(str(save_path) + "/accuracy_rect.txt")
spearmanr_rect = np.loadtxt(str(save_path) + "/spearmanr_rect.txt")

accuracy_lattice = np.loadtxt(str(save_path) + "/accuracy_lattice.txt")
spearmanr_lattice = np.loadtxt(str(save_path) + "/spearmanr_lattice.txt")

accuracy_custom = np.loadtxt(str(save_path) + "/accuracy_custom.txt")
spearmanr_custom = np.loadtxt(str(save_path) + "/spearmanr_custom.txt")

#############################################################################################
# create plots -- accuracy vs. spearman rho
#############################################################################################
fig = plt.figure(figsize=(7.5, 5))
ax = plt.subplot(111)

plt.plot([0, 1], [accuracy_orig, accuracy_orig], "k-", label="input data")
plt.plot([0, 1], [0.05, 0.05], "k--", label="random guess")


def plot_acc_rho(ax):
    mst = 8
    msr = 12
    accuracy_ss = [accuracy_ss_2, accuracy_ss_3, accuracy_ss_4, accuracy_ss_5, accuracy_ss_6, accuracy_ss_7, accuracy_ss_8, accuracy_ss_9, accuracy_ss_10]
    spearmanr_ss = [spearmanr_ss_2, spearmanr_ss_3, spearmanr_ss_4, spearmanr_ss_5, spearmanr_ss_6, spearmanr_ss_7, spearmanr_ss_8, spearmanr_ss_9, spearmanr_ss_10]
    ax.plot(spearmanr_ss, accuracy_ss, "H", markersize=msr, markeredgecolor=(0.25, 0.25, 1.0), color=(0.75, 0.75, 1.0), zorder=80)
    for kk in range(0, 9):
        ax.plot(spearmanr_ss[kk], accuracy_ss[kk], marker="$S%i$" % (kk + 2), markersize=mst, color=(0.25, 0.25, 1.0), zorder=100)
    ax.scatter(100, 100, marker="$S$", s=mst * 5, color=(0.25, 0.25, 1.0), label="simply supported")
    accuracy_345678910_true_ensemble = np.loadtxt(str(save_path) + "/true_ensemble_accuracy.txt")
    spearmanr_345678910_true_ensemble = [np.mean(accuracy_spearmanr_ss_3_ensemble[:, 1]), np.mean(accuracy_spearmanr_ss_4_ensemble[:, 1]), np.mean(accuracy_spearmanr_ss_5_ensemble[:, 1]), np.mean(accuracy_spearmanr_ss_6_ensemble[:, 1]), np.mean(accuracy_spearmanr_ss_7_ensemble[:, 1]), np.mean(accuracy_spearmanr_ss_8_ensemble[:, 1]), np.mean(accuracy_spearmanr_ss_9_ensemble[:, 1]), np.mean(accuracy_spearmanr_ss_10_ensemble[:, 1])]
    ax.plot(spearmanr_345678910_true_ensemble, accuracy_345678910_true_ensemble , "H", markersize=msr, markeredgecolor=(0.25, 0.25, 0.5), color=(0.75, 0.75, 1.0), zorder=80)
    for kk in range(0, 8):
        ax.plot(spearmanr_345678910_true_ensemble[kk], accuracy_345678910_true_ensemble[kk], marker="$E%i$" % (kk + 3), markersize=mst, color=(0.25, 0.25, 0.5), zorder=100)
    ax.scatter(100, 100, marker="$E$", s=mst * 5, color=(0.25, 0.25, 0.5), label="ss ensemble")
    ix = 0
    for fix_whole_bottom in [True, False]:
        for depth_num in [1, 2, 3, 4, 5]:
            for sensor_num in [2, 3, 4, 5]:
                if fix_whole_bottom:
                    # ax.plot(spearmanr_rect[ix], accuracy_rect[ix], "s", color=plt.cm.plasma(0.5 + (sensor_num - 2) / 6), markersize=8 + depth_num, markeredgecolor="k", markeredgewidth=1.5)
                    # ax.plot(spearmanr_rect[ix], accuracy_rect[ix], marker="$R%i$" % (sensor_num), color=(0, 0, 0), markersize=6 + depth_num)
                    ax.plot(spearmanr_rect[ix], accuracy_rect[ix], "D", color=(sensor_num / 5.0, sensor_num / 5.0, sensor_num / 5.0), linestyle="None", markersize=10, markeredgecolor=plt.cm.coolwarm((depth_num - 1) / 4), markeredgewidth=3, zorder=100)
                    ax.plot(spearmanr_rect[ix], accuracy_rect[ix], marker="$R%i$" % (sensor_num), linestyle="None", color=(0, 0, 0), markersize=8, zorder=200)
                else:
                    # ax.plot(spearmanr_rect[ix], accuracy_rect[ix], "s", color=plt.cm.plasma(0.5 + (sensor_num - 2) / 6), markersize=8 + depth_num)
                    # ax.plot(spearmanr_rect[ix], accuracy_rect[ix], marker="$R%i$" % (sensor_num), color=(0, 0, 0), markersize=6 + depth_num)
                    ax.plot(spearmanr_rect[ix], accuracy_rect[ix], "s", color=(sensor_num / 5.0, sensor_num / 5.0, sensor_num / 5.0), linestyle="None", markersize=10, markeredgecolor=plt.cm.coolwarm((depth_num - 1) / 4), markeredgewidth=3, zorder=100)
                    ax.plot(spearmanr_rect[ix], accuracy_rect[ix], marker="$R%i$" % (sensor_num), color=(0, 0, 0), linestyle="None", markersize=8, zorder=200)
                ix += 1
    ax.scatter(100, 100, marker="$R$", s=mst * 5, color=(0, 0, 0), label="rectangle")
    ax.plot(100, 100, marker="s", linestyle="None", color=(0.75, 0.75, 0.75), markersize=10, zorder=100, label="free btm")
    ax.plot(100, 100, marker="D", linestyle="None", color=(0.75, 0.75, 0.75), markersize=10, zorder=100, label="fixed btm")
    ax.plot(100, 100, marker="s", linestyle="None", color=(0.5, 0.5, 0.5), markersize=10, zorder=100, markeredgecolor=plt.cm.coolwarm((1 - 1) / 4), markeredgewidth=3, label="depth = 1.0")
    ax.plot(100, 100, marker="s", linestyle="None", color=(0.5, 0.5, 0.5), markersize=10, zorder=100, markeredgecolor=plt.cm.coolwarm((2 - 1) / 4), markeredgewidth=3, label="depth = 2.5")
    ax.plot(100, 100, marker="s", linestyle="None", color=(0.5, 0.5, 0.5), markersize=10, zorder=100, markeredgecolor=plt.cm.coolwarm((3 - 1) / 4), markeredgewidth=3, label="depth = 5.0")
    ax.plot(100, 100, marker="s", linestyle="None", color=(0.5, 0.5, 0.5), markersize=10, zorder=100, markeredgecolor=plt.cm.coolwarm((4 - 1) / 4), markeredgewidth=3, label="depth = 10.0")
    ax.plot(100, 100, marker="s", linestyle="None", color=(0.5, 0.5, 0.5), markersize=10, zorder=100, markeredgecolor=plt.cm.coolwarm((5 - 1) / 4), markeredgewidth=3, label="depth = 20.0")
    ax.plot(spearmanr_lattice, accuracy_lattice, "o", markersize=msr, markeredgecolor=(1, 0.25, 0.25), color=(0.75, 0.75, 0.75), zorder=80)
    for kk in range(0, 3):
        ax.plot(spearmanr_lattice[kk], accuracy_lattice[kk], marker="$L%i$" % (kk + 3), markersize=mst, color=(1, 0.25, 0.25), zorder=100)
    ax.scatter(100, 100, marker="$L$", s=mst * 5, color=(1, 0.25, 0.25), label="lattice")
    ax.plot(spearmanr_custom, accuracy_custom, "o", markersize=msr, markeredgecolor=(.5, 0.25, 0.25), color=(0.75, 0.75, 0.75), zorder=80)
    for kk in range(0, 3):
        ax.plot(spearmanr_custom[kk], accuracy_custom[kk], marker="$C%i$" % (kk + 1), markersize=mst, color=(.5, 0.25, 0.25), zorder=100)
    ax.scatter(100, 100, marker="$C$", s=mst * 5, color=(0.5, 0.25, 0.25), label="custom")
    return


plot_acc_rho(ax)

inset_ax = ax.inset_axes([0.625, 0.025, 0.325, 0.4])
plot_acc_rho(inset_ax)
inset_ax.set_xlim((0.33, 0.39))
inset_ax.set_ylim((0.135, 0.215))
inset_ax.grid(True)
inset_ax.set_xticks([])
inset_ax.set_yticks([])
ax.indicate_inset_zoom(inset_ax)


plt.xlabel(r"Spearman's $\rho$ correlation")
plt.ylabel("accuracy")
plt.xlim((0.2, 0.9))
plt.ylim((0.0, 1.0))

box = ax.get_position()
ax.set_position([box.x0, box.y0, box.width * 0.8, box.height])

# Put a legend to the right of the current axis
ax.legend(loc='center left', bbox_to_anchor=(1, 0.5))
plt.grid(True)
plt.title("functional performance")

plt.savefig(str(fig_path) + "/accuracy_rho_plot.png")
plt.savefig(str(fig_path) + "/accuracy_rho_plot.eps")

#############################################################################################
# create plots -- accuracy vs. depth
#############################################################################################
fig = plt.figure(figsize=(7.5, 5))
ax = plt.subplot(111)

plt.plot([0, 21], [accuracy_orig, accuracy_orig], "k-", label="input data")
plt.plot([0, 21], [0.05, 0.05], "k--", label="random guess")


def all_pts(ax):
    mst = 8
    msr = 12
    ix = 0
    for fix_whole_bottom in [True, False]:
        for depth_num in [1, 2, 3, 4, 5]:
            for sensor_num in [2, 3, 4, 5]:
                if depth_num == 1:
                    depth = 1
                elif depth_num == 2:
                    depth = 2.5
                elif depth_num == 3:
                    depth = 5.0
                elif depth_num == 4:
                    depth = 10.0
                else:
                    depth = 20.0
                if fix_whole_bottom:
                    ax.plot(depth, accuracy_rect[ix], "D", color=(sensor_num / 5.0, sensor_num / 5.0, sensor_num / 5.0), linestyle="None", markersize=10, markeredgecolor=plt.cm.coolwarm((depth_num - 1) / 4), markeredgewidth=3, zorder=100)
                    ax.plot(depth, accuracy_rect[ix], marker="$R%i$" % (sensor_num), linestyle="None", color=(0, 0, 0), markersize=8, zorder=200)
                    # ax.plot(depth, accuracy_rect[ix], "s", color=plt.cm.plasma(0.5 + (sensor_num - 2) / 6), markersize=8 + depth_num, markeredgecolor="k", markeredgewidth=1.5)
                    # ax.plot(depth, accuracy_rect[ix], marker="$R%i$" % (sensor_num), color=(0, 0, 0), markersize=6 + depth_num)
                else:
                    ax.plot(depth, accuracy_rect[ix], "s", color=(sensor_num / 5.0, sensor_num / 5.0, sensor_num / 5.0), linestyle="None", markersize=10, markeredgecolor=plt.cm.coolwarm((depth_num - 1) / 4), markeredgewidth=3, zorder=100)
                    ax.plot(depth, accuracy_rect[ix], marker="$R%i$" % (sensor_num), color=(0, 0, 0), linestyle="None", markersize=8, zorder=200)
                    # ax.plot(depth, accuracy_rect[ix], "s", color=plt.cm.plasma(0.5 + (sensor_num - 2) / 6), markersize=8 + depth_num)
                    # ax.plot(depth, accuracy_rect[ix], marker="$R%i$" % (sensor_num), color=(0, 0, 0), markersize=6 + depth_num)
                ix += 1
    ax.scatter(100, 100, marker="$R$", s=mst * 5, color=(0, 0, 0), label="rectangle")
    ax.plot(100, 100, marker="s", linestyle="None", color=(0.75, 0.75, 0.75), markersize=10, zorder=100, label="free btm")
    ax.plot(100, 100, marker="D", linestyle="None", color=(0.75, 0.75, 0.75), markersize=10, zorder=100, label="fixed btm")
    ax.plot(100, 100, marker="s", linestyle="None", color=(0.5, 0.5, 0.5), markersize=10, zorder=100, markeredgecolor=plt.cm.coolwarm((1 - 1) / 4), markeredgewidth=3, label="depth = 1.0")
    ax.plot(100, 100, marker="s", linestyle="None", color=(0.5, 0.5, 0.5), markersize=10, zorder=100, markeredgecolor=plt.cm.coolwarm((2 - 1) / 4), markeredgewidth=3, label="depth = 2.5")
    ax.plot(100, 100, marker="s", linestyle="None", color=(0.5, 0.5, 0.5), markersize=10, zorder=100, markeredgecolor=plt.cm.coolwarm((3 - 1) / 4), markeredgewidth=3, label="depth = 5.0")
    ax.plot(100, 100, marker="s", linestyle="None", color=(0.5, 0.5, 0.5), markersize=10, zorder=100, markeredgecolor=plt.cm.coolwarm((4 - 1) / 4), markeredgewidth=3, label="depth = 10.0")
    ax.plot(100, 100, marker="s", linestyle="None", color=(0.5, 0.5, 0.5), markersize=10, zorder=100, markeredgecolor=plt.cm.coolwarm((5 - 1) / 4), markeredgewidth=3, label="depth = 20.0")
    ax.plot([10, 10, 10], accuracy_lattice, "o", markersize=msr, markeredgecolor=(1, 0.25, 0.25), color=(0.75, 0.75, 0.75), zorder=80)
    for kk in range(0, 3):
        ax.plot(10, accuracy_lattice[kk], marker="$L%i$" % (kk + 3), markersize=mst, color=(1, 0.25, 0.25), zorder=100)
    ax.scatter(100, 100, marker="$L$", s=mst * 5, color=(1, 0.25, 0.25), label="lattice")
    ax.plot([10, 10, 10], accuracy_custom, "o", markersize=msr, markeredgecolor=(.5, 0.25, 0.25), color=(0.75, 0.75, 0.75), zorder=80)
    for kk in range(0, 3):
        ax.plot(10, accuracy_custom[kk], marker="$C%i$" % (kk + 1), markersize=mst, color=(.5, 0.25, 0.25), zorder=100)
    ax.scatter(100, 100, marker="$C$", s=mst * 5, color=(0.5, 0.25, 0.25), label="custom")
    return


all_pts(ax)

inset_ax = ax.inset_axes([0.625, 0.35, 0.2, 0.6])
all_pts(inset_ax)
inset_ax.set_xlim((9.4, 10.6))
inset_ax.set_ylim((0.36, 0.415))
inset_ax.grid(True)
inset_ax.set_xticks([])
inset_ax.set_yticks([])
ax.indicate_inset_zoom(inset_ax)


plt.xlabel("depth")
plt.ylabel("accuracy")
plt.ylim((0.0, 1.0))
plt.xlim((0.0, 21.0))

box = ax.get_position()
ax.set_position([box.x0, box.y0, box.width * 0.8, box.height])
ax.set_xticks([1.0, 2.5, 5.0, 10.0, 20.0])

# Put a legend to the right of the current axis
ax.legend(loc='center left', bbox_to_anchor=(1, 0.5))
plt.grid(True)
plt.title("functional performance")

plt.savefig(str(fig_path) + "/accuracy_depth_plot.png")
plt.savefig(str(fig_path) + "/accuracy_depth_plot.eps")


############################################################################
# legend figure
############################################################################

aa = 44