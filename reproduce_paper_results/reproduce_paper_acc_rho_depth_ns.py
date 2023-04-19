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

#############################################################################################

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
accuracy_345678910_true_ensemble = np.loadtxt(str(save_path) + "/true_ensemble_accuracy.txt")

# set up the appropriate sensor number vectors for the simply supported beams
sensor_num_ss = [2, 3, 4, 5, 6, 7, 8, 9, 10]
sensor_num_ss_ensemble = [3, 4, 5, 6, 7, 8, 9, 10]

spearmanr_rect = np.loadtxt(str(save_path) + "/spearmanr_rect.txt")
spearmanr_lattice = np.loadtxt(str(save_path) + "/spearmanr_lattice.txt")
spearmanr_custom = np.loadtxt(str(save_path) + "/spearmanr_custom.txt")

accuracy_rect = np.loadtxt(str(save_path) + "/accuracy_rect.txt")
accuracy_lattice = np.loadtxt(str(save_path) + "/accuracy_lattice.txt")
accuracy_custom = np.loadtxt(str(save_path) + "/accuracy_custom.txt")

depth_rect_fix_true = []
sensors_rect_fix_true = []
spearman_rect_fix_true = []
acc_rect_fix_true = []
depth_rect_fix_false = []
sensors_rect_fix_false = []
spearman_rect_fix_false = []
acc_rect_fix_false = []

ix = 0
for fix_whole_bottom in [True, False]:
    for depth_num in [1.0, 2.5, 5.0, 10.0, 20.0]:
        for sensor_num in [2, 3, 4, 5]:
            if fix_whole_bottom:
                depth_rect_fix_true.append(depth_num)
                sensors_rect_fix_true.append((sensor_num))
                spearman_rect_fix_true.append(spearmanr_rect[ix])
                acc_rect_fix_true.append(accuracy_rect[ix])
            else:
                depth_rect_fix_false.append(depth_num)
                sensors_rect_fix_false.append((sensor_num))
                spearman_rect_fix_false.append(spearmanr_rect[ix])
                acc_rect_fix_false.append(accuracy_rect[ix])
            ix += 1

sensor_num_lattice = [3, 4, 5]
depth_lattice = [10.0, 10.0, 10.0]

sensor_num_custom = [3, 3, 5]
depth_custom = [10.0, 10.0, 10.0]

#############################################################################################
# actual figure
fig, axs = plt.subplots(3, 2, figsize=(7.5, 10))
mst = 8
msr = 12
#############################################################################################
# rho vs. number of sensors ax[0, 0]

ax = axs[0, 0]

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

ax.set_ylabel(r"Spearman's $\rho$ correlation")
ax.set_xlabel("number of sensors")
ax.set_xlim((1.7, 10.25))
ax.set_ylim((0.3, 0.85))
ax.set_xticks([2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0])
ax.grid(True)


#############################################################################################
# rho vs. depth ax[1, 0]
ax = axs[0, 1]

ix = 0
depth_list = [1.0, 2.5, 5.0, 10.0, 20.0]
for fix_whole_bottom in [True, False]:
    for depth_num in range(0, 5):
        for sensor_num in [2, 3, 4, 5]:
            if fix_whole_bottom:
                ax.plot(depth_list[depth_num], spearmanr_rect[ix], "D", color=(sensor_num / 5.0, sensor_num / 5.0, sensor_num / 5.0), markersize=10, markeredgecolor=plt.cm.coolwarm((depth_num) / 4), markeredgewidth=3)
                ax.plot(depth_list[depth_num], spearmanr_rect[ix], marker="$R%i$" % (sensor_num), color=(0, 0, 0), markersize=8)
            else:
                ax.plot(depth_list[depth_num], spearmanr_rect[ix], "s", color=(sensor_num / 5.0, sensor_num / 5.0, sensor_num / 5.0), markersize=10, markeredgecolor=plt.cm.coolwarm((depth_num) / 4), markeredgewidth=3)
                ax.plot(depth_list[depth_num], spearmanr_rect[ix], marker="$R%i$" % (sensor_num), color=(0, 0, 0), markersize=8)
            ix += 1
ax.plot([10.0, 10.0, 10.0], spearmanr_lattice, "o", markersize=msr, markeredgecolor=(1, 0.25, 0.25), color=(0.75, 0.75, 0.75), zorder=80)
for kk in range(0, 3):
    ax.plot(10.0, spearmanr_lattice[kk], marker="$L%i$" % (kk + 3), markersize=mst, color=(1, 0.25, 0.25), zorder=100)
ax.plot([10.0, 10.0, 10.0], spearmanr_custom, "o", markersize=msr, markeredgecolor=(.5, 0.25, 0.25), color=(0.75, 0.75, 0.75), zorder=80)
for kk in range(0, 3):
    ax.plot(10.0, spearmanr_custom[kk], marker="$C%i$" % (kk + 1), markersize=mst, color=(.5, 0.25, 0.25), zorder=100)

ax.set_ylabel(r"Spearman's $\rho$ correlation")
ax.set_xlabel("depth")
ax.set_xlim((0.0, 21.0))
ax.set_ylim((0.3, 0.85))
ax.set_xticks([1.0, 2.5, 5.0, 10.0, 20.0])
ax.grid(True)

#############################################################################################
# acc vs. number of sensors ax[0, 1]

ax = axs[1, 0]

sensor_num_ss = [2, 3, 4, 5, 6, 7, 8, 9, 10]
ax.plot(sensor_num_ss, accuracy_ss, "H", markersize=msr, markeredgecolor=(0.25, 0.25, 1.0), color=(0.75, 0.75, 1.0), zorder=80)
for kk in range(0, 9):
    ax.plot(sensor_num_ss[kk], accuracy_ss[kk], marker="$S%i$" % (kk + 2), markersize=mst, color=(0.25, 0.25, 1.0), zorder=100)
sensor_num_ensemble = [3, 4, 5, 6, 7, 8, 9, 10]
ax.plot(sensor_num_ensemble, accuracy_345678910_true_ensemble, "H", markersize=msr, markeredgecolor=(0.25, 0.25, 0.5), color=(0.75, 0.75, 1.0), zorder=80)
for kk in range(0, 8):
    ax.plot(sensor_num_ensemble[kk], accuracy_345678910_true_ensemble[kk], marker="$E%i$" % (kk + 3), markersize=mst, color=(0.25, 0.25, 0.5), zorder=100)

ax.plot([0, 21], [accuracy_orig, accuracy_orig], "k-", label="input data")
ax.plot([0, 21], [0.05, 0.05], "k--", label="random guess")

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
                ax.plot(sensor_num, accuracy_rect[ix], "D", color=(sensor_num / 5.0, sensor_num / 5.0, sensor_num / 5.0), linestyle="None", markersize=10, markeredgecolor=plt.cm.coolwarm((depth_num - 1) / 4), markeredgewidth=3, zorder=100)
                ax.plot(sensor_num, accuracy_rect[ix], marker="$R%i$" % (sensor_num), linestyle="None", color=(0, 0, 0), markersize=8, zorder=200)
            else:
                ax.plot(sensor_num, accuracy_rect[ix], "s", color=(sensor_num / 5.0, sensor_num / 5.0, sensor_num / 5.0), linestyle="None", markersize=10, markeredgecolor=plt.cm.coolwarm((depth_num - 1) / 4), markeredgewidth=3, zorder=100)
                ax.plot(sensor_num, accuracy_rect[ix], marker="$R%i$" % (sensor_num), color=(0, 0, 0), linestyle="None", markersize=8, zorder=200)
            ix += 1
ax.plot([3, 4, 5], accuracy_lattice, "o", markersize=msr, markeredgecolor=(1, 0.25, 0.25), color=(0.75, 0.75, 0.75), zorder=80)
for kk in range(0, 3):
    ax.plot(kk + 3, accuracy_lattice[kk], marker="$L%i$" % (kk + 3), markersize=mst, color=(1, 0.25, 0.25), zorder=100)
ax.plot([3, 3, 5], accuracy_custom, "o", markersize=msr, markeredgecolor=(.5, 0.25, 0.25), color=(0.75, 0.75, 0.75), zorder=80)
ss = [3, 3, 5]
for kk in range(0, 3):
    ax.plot(ss[kk], accuracy_custom[kk], marker="$C%i$" % (kk + 1), markersize=mst, color=(.5, 0.25, 0.25), zorder=100)

ax.set_xlabel("number of sensors")
ax.set_xlim((1.7, 10.25))
ax.set_ylim((0.0, 0.9))
ax.set_ylabel("classification accuracy")
ax.set_xticks([2, 3, 4, 5, 6, 7, 8, 9, 10])
ax.grid(True)


#############################################################################################
# acc vs. depth ax[1, 1]

ax = axs[1, 1]

ax.plot([0, 21], [accuracy_orig, accuracy_orig], "k-")
ax.plot([0, 21], [0.05, 0.05], "k--")

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
            else:
                ax.plot(depth, accuracy_rect[ix], "s", color=(sensor_num / 5.0, sensor_num / 5.0, sensor_num / 5.0), linestyle="None", markersize=10, markeredgecolor=plt.cm.coolwarm((depth_num - 1) / 4), markeredgewidth=3, zorder=100)
                ax.plot(depth, accuracy_rect[ix], marker="$R%i$" % (sensor_num), color=(0, 0, 0), linestyle="None", markersize=8, zorder=200)
            ix += 1
ax.plot([10, 10, 10], accuracy_lattice, "o", markersize=msr, markeredgecolor=(1, 0.25, 0.25), color=(0.75, 0.75, 0.75), zorder=80)
for kk in range(0, 3):
    ax.plot(10, accuracy_lattice[kk], marker="$L%i$" % (kk + 3), markersize=mst, color=(1, 0.25, 0.25), zorder=100)
ax.scatter(100, 100, marker="$L$", s=mst * 5, color=(1, 0.25, 0.25))
ax.plot([10, 10, 10], accuracy_custom, "o", markersize=msr, markeredgecolor=(.5, 0.25, 0.25), color=(0.75, 0.75, 0.75), zorder=80)
for kk in range(0, 3):
    ax.plot(10, accuracy_custom[kk], marker="$C%i$" % (kk + 1), markersize=mst, color=(.5, 0.25, 0.25), zorder=100)
ax.scatter(100, 100, marker="$C$", s=mst * 5, color=(0.5, 0.25, 0.25))

ax.set_xlabel("depth")
ax.set_ylabel("classification accuracy")
ax.set_ylim((0.0, 0.9))
ax.set_xlim((0.0, 21.0))
ax.set_xticks([1.0, 2.5, 5.0, 10.0, 20.0])
ax.grid(True)

#############################################################################################
# legend settings -- whole figure --
axs[2, 0].get_xaxis().set_visible(False)
axs[2, 0].get_yaxis().set_visible(False)
axs[2, 0].set_axis_off()
axs[2, 1].get_xaxis().set_visible(False)
axs[2, 1].get_yaxis().set_visible(False)
axs[2, 1].set_axis_off()
axs[2, 1].scatter(100, 100, marker=" ", label=" ")
axs[2, 1].scatter(100, 100, marker=" ", label=" ")

# plt.figlegend(bbox_to_anchor=[0.25, 0.25], ncols=10)
plt.figlegend(ncols=4, loc="lower center", bbox_to_anchor=(0.0, 0.175, 1.0, 0.375))

plt.tight_layout()

plt.savefig(str(fig_path) + "/figure_9.png")
plt.savefig(str(fig_path) + "/figure_9.eps")

aa = 44