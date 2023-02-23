import matplotlib.pyplot as plt
import numpy as np
import os
from pathlib import Path


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
tab_path = create_folder(mypath, "table")

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

accuracy_rect = np.loadtxt(str(save_path) + "/accuracy_rect.txt")
spearmanr_rect = np.loadtxt(str(save_path) + "/spearmanr_rect.txt")

accuracy_lattice = np.loadtxt(str(save_path) + "/accuracy_lattice.txt")
spearmanr_lattice = np.loadtxt(str(save_path) + "/spearmanr_lattice.txt")

accuracy_custom = np.loadtxt(str(save_path) + "/accuracy_custom.txt")
spearmanr_custom = np.loadtxt(str(save_path) + "/spearmanr_custom.txt")

#############################################################################################
# create table info
#############################################################################################
name_list = []
ns_list = []
d_list = []
rho_list = []
acc_list = []

# simply supported
for kk in range(0, 9):
    name_list.append("simply supported")
    ns_list.append(kk + 2)
    d_list.append("n/a")
    rho_list.append(spearmanr_ss[kk])
    acc_list.append(accuracy_ss[kk])

# ensemble
for kk in range(0, 8):
    name_list.append("ss ensemble")
    ns_list.append(kk + 3)
    d_list.append("n/a")
    rho_list.append(spearmanr_345678910_true_ensemble[kk])
    acc_list.append(accuracy_345678910_true_ensemble[kk])

# rectange
ix = 0
for fix_whole_bottom in [True, False]:
    for depth_num in [1.0, 2.5, 5.0, 10.0, 20.0]:
        for sensor_num in [2, 3, 4, 5]:
            rho = spearmanr_rect[ix]
            acc = accuracy_rect[ix]
            if fix_whole_bottom:
                name_list.append("rect fixed btm")
            else:
                name_list.append("rect")
            ns_list.append(sensor_num)
            d_list.append(depth_num)
            rho_list.append(rho)
            acc_list.append(acc)
            ix += 1


# lattice
for kk in range(0, 3):
    name_list.append("lattice")
    ns_list.append(kk + 3)
    d_list.append(10)
    rho_list.append(accuracy_lattice[kk])
    acc_list.append(spearmanr_lattice[kk])

# custom
for kk in range(0, 3):
    name_list.append("custom %i" % (kk + 1))
    if kk == 0 or kk == 1:
        ns_list.append(3)
    else:
        ns_list.append(5)
    d_list.append(10)
    rho_list.append(accuracy_custom[kk])
    acc_list.append(spearmanr_custom[kk])


with open(str(tab_path) + "/names.txt", "w") as text_file:
    for kk in range(0, len(name_list)):
        print(name_list[kk], file=text_file)


with open(str(tab_path) + "/depth.txt", "w") as text_file:
    for kk in range(0, len(d_list)):
        print(d_list[kk], file=text_file)


with open(str(tab_path) + "/sensor.txt", "w") as text_file:
    for kk in range(0, len(ns_list)):
        print(ns_list[kk], file=text_file)


with open(str(tab_path) + "/rho.txt", "w") as text_file:
    for kk in range(0, len(rho_list)):
        print("%0.02f" % (rho_list[kk]), file=text_file)


with open(str(tab_path) + "/acc.txt", "w") as text_file:
    for kk in range(0, len(acc_list)):
        print("%0.02f" % (acc_list[kk]), file=text_file)



aa = 44