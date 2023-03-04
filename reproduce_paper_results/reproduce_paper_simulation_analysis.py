import applied_loads as al
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path
import problem_setup_fcns as fcns

compute_loads = True
compute_orig = True
compute_ss = True
compute_ss_ensemble = True
compute_ss_true_ensemble = True
compute_rect = True
compute_lattice = True
compute_custom = True
compute_ss_bins = True

bin_list = [0, 0.14393543185803712, 0.17220304717344215, 0.19785734238406477, 0.23039366502635231, 1.0]

mypath = Path(__file__).resolve().parent
save_path = fcns.create_folder(mypath, "analysis_results")
fea_path = fcns.create_folder(mypath, "FEA_results_summarized")
ss_path = fcns.create_folder(mypath, "SS_results_summarized")
fig_path = fcns.create_folder(mypath, "figures")

num_pts = 1000
if compute_loads:
    all_loads = al.compute_all_loads(num_pts)
    np.save(str(save_path) + "/all_loads.npy", all_loads)
else:
    all_loads = np.load(str(save_path) + "/all_loads.npy")


def area_under_curve(y_load, num_pts):
    x_load = np.linspace(0, 10, num_pts)
    area = np.trapz(y_load, x_load)
    return -1.0 * area


# normalize all_loads
for kk in range(0, len(all_loads)):
    for jj in range(0, all_loads[kk].shape[0]):
        # all_loads[kk][jj, :] = all_loads[kk][jj, :] / np.sqrt(np.sum(all_loads[kk][jj, :] ** 2.0))
        area_of_load = area_under_curve(all_loads[kk][jj, :], num_pts)
        all_loads[kk][jj, :] = all_loads[kk][jj, :] / area_of_load


# compute LOOCV accuracy based on the original timeseries
if compute_orig:
    accuracy_orig, spearmanr_orig, input_data_orig, output_data_orig, predictions = fcns.compute_performance_direct(all_loads)
    np.savetxt(str(save_path) + "/orig_predictions.txt", predictions)
    np.savetxt(str(save_path) + "/orig_performance.txt", np.asarray([accuracy_orig, spearmanr_orig]))
    np.savetxt(str(save_path) + "/input_data_orig.txt", input_data_orig)
    np.savetxt(str(save_path) + "/output_data_orig.txt", output_data_orig)
    # compute all distances
    dist_orig = fcns.dist_vector_all(input_data_orig)
    np.savetxt(str(save_path) + "/input_dist_orig.txt", dist_orig)
else:
    orig = np.loadtxt(str(save_path) + "/orig_performance.txt")
    accuracy_orig = orig[0]
    spearmanr_orig = orig[1]
    input_data_orig = np.loadtxt(str(save_path) + "/input_data_orig.txt")
    output_data_orig = np.loadtxt(str(save_path) + "/output_data_orig.txt")
    dist_orig = np.loadtxt(str(save_path) + "/input_dist_orig.txt")


# compute simply supported hash
def normalize_rows(arr):
    for kk in range(arr.shape[0]):
        # arr[kk, :] = arr[kk, :] / np.sqrt(np.sum(arr[kk, :] ** 2.0))
        arr[kk, :] = arr[kk, :] / np.sum(arr[kk, :])
    return arr


def normalize_rows_list(arr_list):
    for kk in range(0, len(arr_list)):
        arr_list[kk] = normalize_rows(arr_list[kk])
    return arr_list


if compute_ss:
    input_data_ss_2 = np.loadtxt(str(ss_path) + "/ssb_results_2.txt")
    input_data_ss_2 = normalize_rows(input_data_ss_2)
    accuracy_ss_2, spearmanr_ss_2 = fcns.compute_performance(input_data_ss_2, input_data_orig, output_data_orig)
    np.savetxt(str(save_path) + "/ss_performance_2.txt", np.asarray([accuracy_ss_2, spearmanr_ss_2]))
    input_data_ss_3 = np.loadtxt(str(ss_path) + "/ssb_results_3.txt")
    input_data_ss_3 = normalize_rows(input_data_ss_3)
    accuracy_ss_3, spearmanr_ss_3 = fcns.compute_performance(input_data_ss_3, input_data_orig, output_data_orig)
    np.savetxt(str(save_path) + "/ss_performance_3.txt", np.asarray([accuracy_ss_3, spearmanr_ss_3]))
    input_data_ss_4 = np.loadtxt(str(ss_path) + "/ssb_results_4.txt")
    input_data_ss_4 = normalize_rows(input_data_ss_4)
    accuracy_ss_4, spearmanr_ss_4 = fcns.compute_performance(input_data_ss_4, input_data_orig, output_data_orig)
    np.savetxt(str(save_path) + "/ss_performance_4.txt", np.asarray([accuracy_ss_4, spearmanr_ss_4]))
    input_data_ss_5 = np.loadtxt(str(ss_path) + "/ssb_results_5.txt")
    input_data_ss_5 = normalize_rows(input_data_ss_5)
    accuracy_ss_5, spearmanr_ss_5 = fcns.compute_performance(input_data_ss_5, input_data_orig, output_data_orig)
    np.savetxt(str(save_path) + "/ss_performance_5.txt", np.asarray([accuracy_ss_5, spearmanr_ss_5]))
    input_data_ss_6 = np.loadtxt(str(ss_path) + "/ssb_results_6.txt")
    input_data_ss_6 = normalize_rows(input_data_ss_6)
    accuracy_ss_6, spearmanr_ss_6 = fcns.compute_performance(input_data_ss_6, input_data_orig, output_data_orig)
    np.savetxt(str(save_path) + "/ss_performance_6.txt", np.asarray([accuracy_ss_6, spearmanr_ss_6]))
    input_data_ss_7 = np.loadtxt(str(ss_path) + "/ssb_results_7.txt")
    input_data_ss_7 = normalize_rows(input_data_ss_7)
    accuracy_ss_7, spearmanr_ss_7 = fcns.compute_performance(input_data_ss_7, input_data_orig, output_data_orig)
    np.savetxt(str(save_path) + "/ss_performance_7.txt", np.asarray([accuracy_ss_7, spearmanr_ss_7]))
    input_data_ss_8 = np.loadtxt(str(ss_path) + "/ssb_results_8.txt")
    input_data_ss_8 = normalize_rows(input_data_ss_8)
    accuracy_ss_8, spearmanr_ss_8 = fcns.compute_performance(input_data_ss_8, input_data_orig, output_data_orig)
    np.savetxt(str(save_path) + "/ss_performance_8.txt", np.asarray([accuracy_ss_8, spearmanr_ss_8]))
    input_data_ss_9 = np.loadtxt(str(ss_path) + "/ssb_results_9.txt")
    input_data_ss_9 = normalize_rows(input_data_ss_9)
    accuracy_ss_9, spearmanr_ss_9 = fcns.compute_performance(input_data_ss_9, input_data_orig, output_data_orig)
    np.savetxt(str(save_path) + "/ss_performance_9.txt", np.asarray([accuracy_ss_9, spearmanr_ss_9]))
    input_data_ss_10 = np.loadtxt(str(ss_path) + "/ssb_results_10.txt")
    input_data_ss_10 = normalize_rows(input_data_ss_10)
    accuracy_ss_10, spearmanr_ss_10 = fcns.compute_performance(input_data_ss_10, input_data_orig, output_data_orig)
    np.savetxt(str(save_path) + "/ss_performance_10.txt", np.asarray([accuracy_ss_10, spearmanr_ss_10]))
else:
    input_data_ss_2 = np.loadtxt(str(ss_path) + "/ssb_results_2.txt")
    input_data_ss_2 = normalize_rows(input_data_ss_2)
    ss = np.loadtxt(str(save_path) + "/ss_performance_2.txt")
    accuracy_ss_2 = ss[0]
    spearmanr_ss_2 = ss[1]
    input_data_ss_3 = np.loadtxt(str(ss_path) + "/ssb_results_3.txt")
    input_data_ss_3 = normalize_rows(input_data_ss_3)
    ss = np.loadtxt(str(save_path) + "/ss_performance_3.txt")
    accuracy_ss_3 = ss[0]
    spearmanr_ss_3 = ss[1]
    input_data_ss_4 = np.loadtxt(str(ss_path) + "/ssb_results_4.txt")
    input_data_ss_4 = normalize_rows(input_data_ss_4)
    ss = np.loadtxt(str(save_path) + "/ss_performance_4.txt")
    accuracy_ss_4 = ss[0]
    spearmanr_ss_4 = ss[1]
    input_data_ss_5 = np.loadtxt(str(ss_path) + "/ssb_results_5.txt")
    input_data_ss_5 = normalize_rows(input_data_ss_5)
    ss = np.loadtxt(str(save_path) + "/ss_performance_5.txt")
    accuracy_ss_5 = ss[0]
    spearmanr_ss_5 = ss[1]
    input_data_ss_6 = np.loadtxt(str(ss_path) + "/ssb_results_6.txt")
    input_data_ss_6 = normalize_rows(input_data_ss_6)
    ss = np.loadtxt(str(save_path) + "/ss_performance_6.txt")
    accuracy_ss_6 = ss[0]
    spearmanr_ss_6 = ss[1]
    input_data_ss_7 = np.loadtxt(str(ss_path) + "/ssb_results_7.txt")
    input_data_ss_7 = normalize_rows(input_data_ss_7)
    ss = np.loadtxt(str(save_path) + "/ss_performance_7.txt")
    accuracy_ss_7 = ss[0]
    spearmanr_ss_7 = ss[1]
    input_data_ss_8 = np.loadtxt(str(ss_path) + "/ssb_results_8.txt")
    input_data_ss_8 = normalize_rows(input_data_ss_8)
    ss = np.loadtxt(str(save_path) + "/ss_performance_8.txt")
    accuracy_ss_8 = ss[0]
    spearmanr_ss_8 = ss[1]
    input_data_ss_9 = np.loadtxt(str(ss_path) + "/ssb_results_9.txt")
    input_data_ss_9 = normalize_rows(input_data_ss_9)
    ss = np.loadtxt(str(save_path) + "/ss_performance_9.txt")
    accuracy_ss_9 = ss[0]
    spearmanr_ss_9 = ss[1]
    input_data_ss_10 = np.loadtxt(str(ss_path) + "/ssb_results_10.txt")
    input_data_ss_10 = normalize_rows(input_data_ss_10)
    ss = np.loadtxt(str(save_path) + "/ss_performance_10.txt")
    accuracy_ss_10 = ss[0]
    spearmanr_ss_10 = ss[1]


def compute_acc_spr_all(input_data_all):
    accuracy_ss_ensemble = []
    spearmanr_ss_ensemble = []
    for kk in range(0, len(input_data_all)):
        input_data = input_data_all[kk]
        input_data = normalize_rows(input_data)
        acc, spr = fcns.compute_performance(input_data, input_data_orig, output_data_orig)
        accuracy_ss_ensemble.append(acc)
        spearmanr_ss_ensemble.append(spr)
    return accuracy_ss_ensemble, spearmanr_ss_ensemble


if compute_ss_ensemble:
    input_data_ss_3_all = np.load(str(ss_path) + "/ensemble_ssb_results_3.npy")
    accuracy_ss_3_ensemble, spearmanr_ss_3_ensemble = compute_acc_spr_all(input_data_ss_3_all)
    accuracy_spearmanr_ss_3_ensemble = np.hstack([np.asarray(accuracy_ss_3_ensemble).reshape((-1, 1)), np.asarray(spearmanr_ss_3_ensemble).reshape((-1, 1))])
    np.savetxt(str(save_path) + "/ss_performance_3_ensemble.txt", accuracy_spearmanr_ss_3_ensemble)
    input_data_ss_4_all = np.load(str(ss_path) + "/ensemble_ssb_results_4.npy")
    accuracy_ss_4_ensemble, spearmanr_ss_4_ensemble = compute_acc_spr_all(input_data_ss_4_all)
    accuracy_spearmanr_ss_4_ensemble = np.hstack([np.asarray(accuracy_ss_4_ensemble).reshape((-1, 1)), np.asarray(spearmanr_ss_4_ensemble).reshape((-1, 1))])
    np.savetxt(str(save_path) + "/ss_performance_4_ensemble.txt", accuracy_spearmanr_ss_4_ensemble)
    input_data_ss_5_all = np.load(str(ss_path) + "/ensemble_ssb_results_5.npy")
    accuracy_ss_5_ensemble, spearmanr_ss_5_ensemble = compute_acc_spr_all(input_data_ss_5_all)
    accuracy_spearmanr_ss_5_ensemble = np.hstack([np.asarray(accuracy_ss_5_ensemble).reshape((-1, 1)), np.asarray(spearmanr_ss_5_ensemble).reshape((-1, 1))])
    np.savetxt(str(save_path) + "/ss_performance_5_ensemble.txt", accuracy_spearmanr_ss_5_ensemble)
    input_data_ss_6_all = np.load(str(ss_path) + "/ensemble_ssb_results_6.npy")
    accuracy_ss_6_ensemble, spearmanr_ss_6_ensemble = compute_acc_spr_all(input_data_ss_6_all)
    accuracy_spearmanr_ss_6_ensemble = np.hstack([np.asarray(accuracy_ss_6_ensemble).reshape((-1, 1)), np.asarray(spearmanr_ss_6_ensemble).reshape((-1, 1))])
    np.savetxt(str(save_path) + "/ss_performance_6_ensemble.txt", accuracy_spearmanr_ss_6_ensemble)
    input_data_ss_7_all = np.load(str(ss_path) + "/ensemble_ssb_results_7.npy")
    accuracy_ss_7_ensemble, spearmanr_ss_7_ensemble = compute_acc_spr_all(input_data_ss_7_all)
    accuracy_spearmanr_ss_7_ensemble = np.hstack([np.asarray(accuracy_ss_7_ensemble).reshape((-1, 1)), np.asarray(spearmanr_ss_7_ensemble).reshape((-1, 1))])
    np.savetxt(str(save_path) + "/ss_performance_7_ensemble.txt", accuracy_spearmanr_ss_7_ensemble)
    input_data_ss_8_all = np.load(str(ss_path) + "/ensemble_ssb_results_8.npy")
    accuracy_ss_8_ensemble, spearmanr_ss_8_ensemble = compute_acc_spr_all(input_data_ss_8_all)
    accuracy_spearmanr_ss_8_ensemble = np.hstack([np.asarray(accuracy_ss_8_ensemble).reshape((-1, 1)), np.asarray(spearmanr_ss_8_ensemble).reshape((-1, 1))])
    np.savetxt(str(save_path) + "/ss_performance_8_ensemble.txt", accuracy_spearmanr_ss_8_ensemble)
    input_data_ss_9_all = np.load(str(ss_path) + "/ensemble_ssb_results_9.npy")
    accuracy_ss_9_ensemble, spearmanr_ss_9_ensemble = compute_acc_spr_all(input_data_ss_9_all)
    accuracy_spearmanr_ss_9_ensemble = np.hstack([np.asarray(accuracy_ss_9_ensemble).reshape((-1, 1)), np.asarray(spearmanr_ss_9_ensemble).reshape((-1, 1))])
    np.savetxt(str(save_path) + "/ss_performance_9_ensemble.txt", accuracy_spearmanr_ss_9_ensemble)
    input_data_ss_10_all = np.load(str(ss_path) + "/ensemble_ssb_results_10.npy")
    accuracy_ss_10_ensemble, spearmanr_ss_10_ensemble = compute_acc_spr_all(input_data_ss_10_all)
    accuracy_spearmanr_ss_10_ensemble = np.hstack([np.asarray(accuracy_ss_10_ensemble).reshape((-1, 1)), np.asarray(spearmanr_ss_10_ensemble).reshape((-1, 1))])
    np.savetxt(str(save_path) + "/ss_performance_10_ensemble.txt", accuracy_spearmanr_ss_10_ensemble)
else:
    accuracy_spearmanr_ss_3_ensemble = np.loadtxt(str(save_path) + "/ss_performance_3_ensemble.txt")
    accuracy_spearmanr_ss_4_ensemble = np.loadtxt(str(save_path) + "/ss_performance_4_ensemble.txt")
    accuracy_spearmanr_ss_5_ensemble = np.loadtxt(str(save_path) + "/ss_performance_5_ensemble.txt")
    input_data_ss_3_all = np.load(str(ss_path) + "/ensemble_ssb_results_3.npy")
    input_data_ss_3_all = normalize_rows_list(input_data_ss_3_all)
    input_data_ss_4_all = np.load(str(ss_path) + "/ensemble_ssb_results_4.npy")
    input_data_ss_4_all = normalize_rows_list(input_data_ss_4_all)
    input_data_ss_5_all = np.load(str(ss_path) + "/ensemble_ssb_results_5.npy")
    input_data_ss_5_all = normalize_rows_list(input_data_ss_5_all)
    accuracy_spearmanr_ss_6_ensemble = np.loadtxt(str(save_path) + "/ss_performance_6_ensemble.txt")
    accuracy_spearmanr_ss_7_ensemble = np.loadtxt(str(save_path) + "/ss_performance_7_ensemble.txt")
    accuracy_spearmanr_ss_8_ensemble = np.loadtxt(str(save_path) + "/ss_performance_8_ensemble.txt")
    accuracy_spearmanr_ss_9_ensemble = np.loadtxt(str(save_path) + "/ss_performance_9_ensemble.txt")
    accuracy_spearmanr_ss_10_ensemble = np.loadtxt(str(save_path) + "/ss_performance_10_ensemble.txt")
    input_data_ss_6_all = np.load(str(ss_path) + "/ensemble_ssb_results_6.npy")
    input_data_ss_6_all = normalize_rows_list(input_data_ss_6_all)
    input_data_ss_7_all = np.load(str(ss_path) + "/ensemble_ssb_results_7.npy")
    input_data_ss_7_all = normalize_rows_list(input_data_ss_7_all)
    input_data_ss_8_all = np.load(str(ss_path) + "/ensemble_ssb_results_8.npy")
    input_data_ss_8_all = normalize_rows_list(input_data_ss_8_all)
    input_data_ss_9_all = np.load(str(ss_path) + "/ensemble_ssb_results_9.npy")
    input_data_ss_9_all = normalize_rows_list(input_data_ss_9_all)
    input_data_ss_10_all = np.load(str(ss_path) + "/ensemble_ssb_results_10.npy")
    input_data_ss_10_all = normalize_rows_list(input_data_ss_10_all)

if compute_ss_true_ensemble:
    input_data_ss_3_all = np.load(str(ss_path) + "/ensemble_ssb_results_3.npy")
    input_data_ss_3_all = normalize_rows_list(input_data_ss_3_all)
    accuracy_ss_3_true_ensemble = fcns.ensemble_evaluate_LOOCV(input_data_ss_3_all, output_data_orig)
    input_data_ss_4_all = np.load(str(ss_path) + "/ensemble_ssb_results_4.npy")
    input_data_ss_4_all = normalize_rows_list(input_data_ss_4_all)
    accuracy_ss_4_true_ensemble = fcns.ensemble_evaluate_LOOCV(input_data_ss_4_all, output_data_orig)
    input_data_ss_5_all = np.load(str(ss_path) + "/ensemble_ssb_results_5.npy")
    input_data_ss_5_all = normalize_rows_list(input_data_ss_5_all)
    accuracy_ss_5_true_ensemble = fcns.ensemble_evaluate_LOOCV(input_data_ss_5_all, output_data_orig)
    input_data_ss_6_all = np.load(str(ss_path) + "/ensemble_ssb_results_6.npy")
    input_data_ss_6_all = normalize_rows_list(input_data_ss_6_all)
    accuracy_ss_6_true_ensemble = fcns.ensemble_evaluate_LOOCV(input_data_ss_6_all, output_data_orig)
    input_data_ss_7_all = np.load(str(ss_path) + "/ensemble_ssb_results_7.npy")
    input_data_ss_7_all = normalize_rows_list(input_data_ss_7_all)
    accuracy_ss_7_true_ensemble = fcns.ensemble_evaluate_LOOCV(input_data_ss_7_all, output_data_orig)
    input_data_ss_8_all = np.load(str(ss_path) + "/ensemble_ssb_results_8.npy")
    input_data_ss_8_all = normalize_rows_list(input_data_ss_8_all)
    accuracy_ss_8_true_ensemble = fcns.ensemble_evaluate_LOOCV(input_data_ss_8_all, output_data_orig)
    input_data_ss_9_all = np.load(str(ss_path) + "/ensemble_ssb_results_9.npy")
    input_data_ss_9_all = normalize_rows_list(input_data_ss_9_all)
    accuracy_ss_9_true_ensemble = fcns.ensemble_evaluate_LOOCV(input_data_ss_9_all, output_data_orig)
    input_data_ss_10_all = np.load(str(ss_path) + "/ensemble_ssb_results_10.npy")
    input_data_ss_10_all = normalize_rows_list(input_data_ss_10_all)
    accuracy_ss_10_true_ensemble = fcns.ensemble_evaluate_LOOCV(input_data_ss_10_all, output_data_orig)
    accuracy_345678910_true_ensemble = np.asarray([accuracy_ss_3_true_ensemble, accuracy_ss_4_true_ensemble, accuracy_ss_5_true_ensemble, accuracy_ss_6_true_ensemble, accuracy_ss_7_true_ensemble, accuracy_ss_8_true_ensemble, accuracy_ss_9_true_ensemble, accuracy_ss_10_true_ensemble])
    np.savetxt(str(save_path) + "/true_ensemble_accuracy.txt", accuracy_345678910_true_ensemble)
else:
    accuracy_345678910_true_ensemble = np.loadtxt(str(save_path) + "/true_ensemble_accuracy.txt")


# compute rectangular domain
if compute_rect:
    accuracy_rect = []
    spearmanr_rect = []
    input_data_rect = []
    middle_dist_rect = []
    match_ratio_rect = []
    mesh_param = 200
    for fix_whole_bottom in [True, False]:
        for depth_num in [1, 2, 3, 4, 5]:
            for sensor_num in [2, 3, 4, 5]:
                fname = "depth_num%i_sensor_num%i_fix_whole_bottom%i.txt" % (depth_num, sensor_num, fix_whole_bottom)
                fname = fea_path.joinpath(fname).resolve()
                data = np.loadtxt(fname)
                data = normalize_rows(data)
                accuracy, spearmanr = fcns.compute_performance(data, input_data_orig, output_data_orig)
                input_data_rect.append(data)
                accuracy_rect.append(accuracy)
                spearmanr_rect.append(spearmanr)
                middle_dist, match_ratio = fcns.hash_probability_dist_all(data, dist_orig, bin_list)
                middle_dist_rect.append(middle_dist)
                match_ratio_rect.append(match_ratio)
    np.save(str(save_path) + "/middle_dist_rect.npy", middle_dist_rect)
    np.save(str(save_path) + "/match_ratio_rect.npy", match_ratio_rect)
    np.savetxt(str(save_path) + "/accuracy_rect.txt", np.asarray(accuracy_rect))
    np.savetxt(str(save_path) + "/spearmanr_rect.txt", np.asarray(spearmanr_rect))
else:
    accuracy_rect = np.loadtxt(str(save_path) + "/accuracy_rect.txt")
    spearmanr_rect = np.loadtxt(str(save_path) + "/spearmanr_rect.txt")
    middle_dist_rect = np.load(str(save_path) + "/middle_dist_rect.npy")
    match_ratio_rect = np.load(str(save_path) + "/match_ratio_rect.npy")

# compute lattice domain
if compute_lattice:
    accuracy_lattice = []
    spearmanr_lattice = []
    input_data_lattice = []
    middle_dist_lattice = []
    match_ratio_lattice = []
    mesh_param = 200
    for lattice_num in [3, 4, 5]:
        fname = "lattice_%i.txt" % (lattice_num)
        fname = fea_path.joinpath(fname).resolve()
        data = np.loadtxt(fname)
        data = normalize_rows(data)
        accuracy, spearmanr = fcns.compute_performance(data, input_data_orig, output_data_orig)
        input_data_lattice.append(data)
        accuracy_lattice.append(accuracy)
        spearmanr_lattice.append(spearmanr)
        middle_dist, match_ratio = fcns.hash_probability_dist_all(data, dist_orig, bin_list)
        middle_dist_lattice.append(middle_dist)
        match_ratio_lattice.append(match_ratio)
    np.save(str(save_path) + "/middle_dist_lattice.npy", middle_dist_lattice)
    np.save(str(save_path) + "/match_ratio_lattice.npy", match_ratio_lattice)
    np.savetxt(str(save_path) + "/accuracy_lattice.txt", np.asarray(accuracy_lattice))
    np.savetxt(str(save_path) + "/spearmanr_lattice.txt", np.asarray(spearmanr_lattice))
else:
    accuracy_lattice = np.loadtxt(str(save_path) + "/accuracy_lattice.txt")
    spearmanr_lattice = np.loadtxt(str(save_path) + "/spearmanr_lattice.txt")
    middle_dist_lattice = np.load(str(save_path) + "/middle_dist_lattice.npy")
    match_ratio_lattice = np.load(str(save_path) + "/match_ratio_lattice.npy")

# compute custom domains
if compute_custom:
    accuracy_custom = []
    spearmanr_custom = []
    input_data_custom = []
    middle_dist_custom = []
    match_ratio_custom = []
    mesh_param = 200
    for custom_num in [1, 2, 3]:
        fname = "grid_25x25_device%i.txt" % (custom_num)
        fname = fea_path.joinpath(fname).resolve()
        data = np.loadtxt(fname)
        data = normalize_rows(data)
        accuracy, spearmanr = fcns.compute_performance(data, input_data_orig, output_data_orig)
        input_data_custom.append(data)
        accuracy_custom.append(accuracy)
        spearmanr_custom.append(spearmanr)
        middle_dist, match_ratio = fcns.hash_probability_dist_all(data, dist_orig, bin_list)
        middle_dist_custom.append(middle_dist)
        match_ratio_custom.append(match_ratio)
    np.save(str(save_path) + "/middle_dist_custom.npy", middle_dist_custom)
    np.save(str(save_path) + "/match_ratio_custom.npy", match_ratio_custom)
    np.savetxt(str(save_path) + "/accuracy_custom.txt", np.asarray(accuracy_custom))
    np.savetxt(str(save_path) + "/spearmanr_custom.txt", np.asarray(spearmanr_custom))
else:
    accuracy_custom = np.loadtxt(str(save_path) + "/accuracy_custom.txt")
    spearmanr_custom = np.loadtxt(str(save_path) + "/spearmanr_custom.txt")
    middle_dist_custom = np.load(str(save_path) + "/middle_dist_custom.npy")
    match_ratio_custom = np.load(str(save_path) + "/match_ratio_custom.npy")


if compute_ss_bins:
    data = input_data_ss_2
    middle_dist_ss_2, match_ratio_ss_2 = fcns.hash_probability_dist_all(data, dist_orig, bin_list)
    data = input_data_ss_3
    middle_dist_ss_3, match_ratio_ss_3 = fcns.hash_probability_dist_all(data, dist_orig, bin_list)
    data = input_data_ss_4
    middle_dist_ss_4, match_ratio_ss_4 = fcns.hash_probability_dist_all(data, dist_orig, bin_list)
    data = input_data_ss_5
    middle_dist_ss_5, match_ratio_ss_5 = fcns.hash_probability_dist_all(data, dist_orig, bin_list)
    data = input_data_ss_6
    middle_dist_ss_6, match_ratio_ss_6 = fcns.hash_probability_dist_all(data, dist_orig, bin_list)
    data = input_data_ss_7
    middle_dist_ss_7, match_ratio_ss_7 = fcns.hash_probability_dist_all(data, dist_orig, bin_list)
    data = input_data_ss_8
    middle_dist_ss_8, match_ratio_ss_8 = fcns.hash_probability_dist_all(data, dist_orig, bin_list)
    data = input_data_ss_9
    middle_dist_ss_9, match_ratio_ss_9 = fcns.hash_probability_dist_all(data, dist_orig, bin_list)
    data = input_data_ss_10
    middle_dist_ss_10, match_ratio_ss_10 = fcns.hash_probability_dist_all(data, dist_orig, bin_list)
    middle_dist_ss_3_all = []
    match_ratio_ss_3_all = []
    for kk in range(0, len(input_data_ss_3_all)):
        data = input_data_ss_3_all[kk]
        middle, match = fcns.hash_probability_dist_all(data, dist_orig, bin_list)
        middle_dist_ss_3_all.append(middle)
        match_ratio_ss_3_all.append(match)
    middle_dist_ss_3_ensemble = np.mean(np.asarray(middle_dist_ss_3_all), axis=0)
    match_ratio_ss_3_ensemble = np.mean(np.asarray(match_ratio_ss_3_all), axis=0)
    middle_dist_ss_4_all = []
    match_ratio_ss_4_all = []
    for kk in range(0, len(input_data_ss_4_all)):
        data = input_data_ss_4_all[kk]
        middle, match = fcns.hash_probability_dist_all(data, dist_orig, bin_list)
        middle_dist_ss_4_all.append(middle)
        match_ratio_ss_4_all.append(match)
    middle_dist_ss_4_ensemble = np.mean(np.asarray(middle_dist_ss_4_all), axis=0)
    match_ratio_ss_4_ensemble = np.mean(np.asarray(match_ratio_ss_4_all), axis=0)
    middle_dist_ss_5_all = []
    match_ratio_ss_5_all = []
    for kk in range(0, len(input_data_ss_5_all)):
        data = input_data_ss_5_all[kk]
        middle, match = fcns.hash_probability_dist_all(data, dist_orig, bin_list)
        middle_dist_ss_5_all.append(middle)
        match_ratio_ss_5_all.append(match)
    middle_dist_ss_5_ensemble = np.mean(np.asarray(middle_dist_ss_5_all), axis=0)
    match_ratio_ss_5_ensemble = np.mean(np.asarray(match_ratio_ss_5_all), axis=0)
    middle_dist_ss_6_all = []
    match_ratio_ss_6_all = []
    for kk in range(0, len(input_data_ss_6_all)):
        data = input_data_ss_6_all[kk]
        middle, match = fcns.hash_probability_dist_all(data, dist_orig, bin_list)
        middle_dist_ss_6_all.append(middle)
        match_ratio_ss_6_all.append(match)
    middle_dist_ss_6_ensemble = np.mean(np.asarray(middle_dist_ss_6_all), axis=0)
    match_ratio_ss_6_ensemble = np.mean(np.asarray(match_ratio_ss_6_all), axis=0)
    middle_dist_ss_7_all = []
    match_ratio_ss_7_all = []
    for kk in range(0, len(input_data_ss_7_all)):
        data = input_data_ss_7_all[kk]
        middle, match = fcns.hash_probability_dist_all(data, dist_orig, bin_list)
        middle_dist_ss_7_all.append(middle)
        match_ratio_ss_7_all.append(match)
    middle_dist_ss_7_ensemble = np.mean(np.asarray(middle_dist_ss_7_all), axis=0)
    match_ratio_ss_7_ensemble = np.mean(np.asarray(match_ratio_ss_7_all), axis=0)
    middle_dist_ss_8_all = []
    match_ratio_ss_8_all = []
    for kk in range(0, len(input_data_ss_8_all)):
        data = input_data_ss_8_all[kk]
        middle, match = fcns.hash_probability_dist_all(data, dist_orig, bin_list)
        middle_dist_ss_8_all.append(middle)
        match_ratio_ss_8_all.append(match)
    middle_dist_ss_8_ensemble = np.mean(np.asarray(middle_dist_ss_8_all), axis=0)
    match_ratio_ss_8_ensemble = np.mean(np.asarray(match_ratio_ss_8_all), axis=0)
    middle_dist_ss_9_all = []
    match_ratio_ss_9_all = []
    for kk in range(0, len(input_data_ss_9_all)):
        data = input_data_ss_9_all[kk]
        middle, match = fcns.hash_probability_dist_all(data, dist_orig, bin_list)
        middle_dist_ss_9_all.append(middle)
        match_ratio_ss_9_all.append(match)
    middle_dist_ss_9_ensemble = np.mean(np.asarray(middle_dist_ss_9_all), axis=0)
    match_ratio_ss_9_ensemble = np.mean(np.asarray(match_ratio_ss_9_all), axis=0)
    middle_dist_ss_10_all = []
    match_ratio_ss_10_all = []
    for kk in range(0, len(input_data_ss_10_all)):
        data = input_data_ss_10_all[kk]
        middle, match = fcns.hash_probability_dist_all(data, dist_orig, bin_list)
        middle_dist_ss_10_all.append(middle)
        match_ratio_ss_10_all.append(match)
    middle_dist_ss_10_ensemble = np.mean(np.asarray(middle_dist_ss_10_all), axis=0)
    match_ratio_ss_10_ensemble = np.mean(np.asarray(match_ratio_ss_10_all), axis=0)
    md_ss = [middle_dist_ss_2, middle_dist_ss_3, middle_dist_ss_4, middle_dist_ss_5, middle_dist_ss_6, middle_dist_ss_7, middle_dist_ss_8, middle_dist_ss_9, middle_dist_ss_10]
    mr_ss = [match_ratio_ss_2, match_ratio_ss_3, match_ratio_ss_4, match_ratio_ss_5, match_ratio_ss_6, match_ratio_ss_7, match_ratio_ss_8, match_ratio_ss_9, match_ratio_ss_10]
    md_ss_e = [middle_dist_ss_3_ensemble, middle_dist_ss_4_ensemble, middle_dist_ss_5_ensemble, middle_dist_ss_6_ensemble, middle_dist_ss_7_ensemble, middle_dist_ss_8_ensemble, middle_dist_ss_9_ensemble, middle_dist_ss_10_ensemble]
    mr_ss_e = [match_ratio_ss_3_ensemble, match_ratio_ss_4_ensemble, match_ratio_ss_5_ensemble, match_ratio_ss_6_ensemble, match_ratio_ss_7_ensemble, match_ratio_ss_8_ensemble, match_ratio_ss_9_ensemble, match_ratio_ss_10_ensemble]
    middle_distance_ss = np.asarray(md_ss)
    match_ratio_ss = np.asarray(mr_ss)
    middle_distance_ss_ensemble = np.asarray(md_ss_e)
    match_ratio_ss_ensemble = np.asarray(mr_ss_e)
    np.savetxt(str(save_path) + "/middle_distance_ss.txt", middle_distance_ss)
    np.savetxt(str(save_path) + "/match_ratio_ss.txt", match_ratio_ss)
    np.savetxt(str(save_path) + "/middle_distance_ss_ensemble.txt", middle_distance_ss_ensemble)
    np.savetxt(str(save_path) + "/match_ratio_ss_ensemble.txt", match_ratio_ss_ensemble)
else:
    middle_distance_ss = np.loadtxt(str(save_path) + "/middle_distance_ss.txt")
    match_ratio_ss = np.loadtxt(str(save_path) + "/match_ratio_ss.txt")
    middle_distance_ss_ensemble = np.loadtxt(str(save_path) + "/middle_distance_ss_ensemble.txt")
    match_ratio_ss_ensemble = np.loadtxt(str(save_path) + "/match_ratio_ss_ensemble.txt")

# first set of plots -- probability of collision vs. distance
# simply supported
middle_dist_all = []
match_ratio_all = []
plt.figure()
for kk in range(0, middle_distance_ss.shape[0]):
    plt.plot(middle_distance_ss[kk, :], match_ratio_ss[kk, :], "-o", label="single %i" % (kk + 2))
    middle_dist_all.append(middle_distance_ss[kk, :])
    match_ratio_all.append(match_ratio_ss[kk, :])

# simply supported ensemble
for kk in range(0, middle_distance_ss_ensemble.shape[0]):
    plt.plot(middle_distance_ss_ensemble[kk, :], match_ratio_ss_ensemble[kk, :], "-s", label="ensemble %i" % (kk + 3))
    middle_dist_all.append(middle_distance_ss_ensemble[kk, :])
    match_ratio_all.append(match_ratio_ss_ensemble[kk, :])

# FEA simulations -- rectangle
ix = 0
for fix_whole_bottom in [True, False]:
    for depth_num in [1, 2, 3, 4, 5]:
        for sensor_num in [2, 3, 4, 5]:
            if fix_whole_bottom:
                plt.plot(middle_dist_rect[ix], match_ratio_rect[ix], "-s", color="r")
            else:
                plt.plot(middle_dist_rect[ix], match_ratio_rect[ix], "-o", color="r")
            middle_dist_all.append(middle_dist_rect[ix])
            match_ratio_all.append(match_ratio_rect[ix])
            ix += 1

# FEA simulations -- lattice
for kk in range(0, len(middle_dist_lattice)):
    plt.plot(middle_dist_lattice[kk], match_ratio_lattice[kk], "--", color="c")
    middle_dist_all.append(middle_dist_lattice[kk])
    match_ratio_all.append(match_ratio_lattice[kk])

# FEA simulations -- custom
for kk in range(0, len(middle_dist_custom)):
    plt.plot(middle_dist_custom[kk], match_ratio_custom[kk], "--", color="g")
    middle_dist_all.append(middle_dist_custom[kk])
    match_ratio_all.append(match_ratio_custom[kk])

middle_dist_all = np.asarray(middle_dist_all)
match_ratio_all = np.asarray(match_ratio_all)
middle_dist_mean = np.mean(middle_dist_all, axis=0)
match_ratio_mean = np.mean(match_ratio_all, axis=0)

plt.legend()
plt.xlabel("L-infty distance")
plt.ylabel("probability of hash collision")
plt.plot(middle_dist_mean, match_ratio_mean, "k--o")
plt.savefig(str(fig_path) + "/collision_curve_draft.png")
plt.ylim((0, 0.025))
plt.savefig(str(fig_path) + "/collision_curve_draft_zoom.png")


# second set of plots -- accuracy vs. rho
plt.figure()
ix = 0
for fix_whole_bottom in [True, False]:
    for depth_num in [1, 2, 3, 4, 5]:
        for sensor_num in [2, 3, 4, 5]:
            if fix_whole_bottom:
                plt.plot(spearmanr_rect[ix], accuracy_rect[ix], "o", color = plt.cm.cividis((depth_num-1) / 4), markersize=sensor_num * 2, markeredgecolor="k", alpha=0.5)
            else:
                plt.plot(spearmanr_rect[ix], accuracy_rect[ix], "s", color = plt.cm.cividis((depth_num-1) / 4), markersize=sensor_num * 2, markeredgecolor="k", alpha=0.5)
            ix += 1

for kk in range(0, len(middle_dist_lattice)):
    plt.plot(spearmanr_lattice[kk], accuracy_lattice[kk], "*", color=plt.cm.autumn(kk / (len(middle_dist_lattice)-1)))

for kk in range(0, len(middle_dist_custom)):
    plt.plot(spearmanr_custom[kk], accuracy_custom[kk], "^", color=plt.cm.cool(kk / (len(middle_dist_custom)-1)))


accuracy_ss = [accuracy_ss_2, accuracy_ss_3, accuracy_ss_4, accuracy_ss_5, accuracy_ss_6, accuracy_ss_7, accuracy_ss_8, accuracy_ss_9, accuracy_ss_10]
spearmanr_ss = [spearmanr_ss_2, spearmanr_ss_3, spearmanr_ss_4, spearmanr_ss_5, spearmanr_ss_6, spearmanr_ss_7, spearmanr_ss_8, spearmanr_ss_9, spearmanr_ss_10]
spearmanr_345678910_true_ensemble = [np.mean(accuracy_spearmanr_ss_3_ensemble[:, 1]), np.mean(accuracy_spearmanr_ss_4_ensemble[:, 1]), np.mean(accuracy_spearmanr_ss_5_ensemble[:, 1]), np.mean(accuracy_spearmanr_ss_6_ensemble[:, 1]), np.mean(accuracy_spearmanr_ss_7_ensemble[:, 1]), np.mean(accuracy_spearmanr_ss_8_ensemble[:, 1]), np.mean(accuracy_spearmanr_ss_9_ensemble[:, 1]), np.mean(accuracy_spearmanr_ss_10_ensemble[:, 1])]
# accuracy_345_true_ensemble
# accuracy_spearmanr_ss_3_ensemble
# accuracy_spearmanr_ss_4_ensemble
# accuracy_spearmanr_ss_5_ensemble

for kk in range(0, len(accuracy_ss)):
    plt.plot(spearmanr_ss[kk], accuracy_ss[kk], "ko")

for kk in range(0, len(spearmanr_345678910_true_ensemble)):
    plt.plot(spearmanr_345678910_true_ensemble[kk], accuracy_345678910_true_ensemble[kk], "kd")


# plt.plot(accuracy_spearmanr_ss_3_ensemble[:, 1], accuracy_spearmanr_ss_3_ensemble[:, 0], "r.")
# plt.plot(accuracy_spearmanr_ss_4_ensemble[:, 1], accuracy_spearmanr_ss_4_ensemble[:, 0], "g.")
# plt.plot(accuracy_spearmanr_ss_5_ensemble[:, 1], accuracy_spearmanr_ss_5_ensemble[:, 0], "b.")
# accuracy_orig
# spearmanr_orig

plt.plot([0, 1],[accuracy_orig, accuracy_orig], "k--")
plt.plot([0, 1],[0.05, 0.05], "k--")


plt.xlabel("spearman r correlation")
plt.ylabel("accuracy")
plt.savefig(str(fig_path) + "/accuracy_plot_draft.png")
plt.xlim((.3, .45))
plt.ylim((.15, .25))
plt.savefig(str(fig_path) + "/accuracy_plot_draft_zoom.png")
