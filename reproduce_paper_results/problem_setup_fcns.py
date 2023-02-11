import applied_loads as al
from collections import Counter
import numpy as np
import os
from pathlib import Path
from scipy import stats
from sklearn.metrics import accuracy_score
from sklearn.model_selection import LeaveOneOut
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import StandardScaler
from typing import List


def create_folder(folder_path: Path, new_folder_name: str) -> Path:
    """Given a path to a directory and a folder name. Will create a directory in the given directory."""
    new_path = folder_path.joinpath(new_folder_name).resolve()
    if new_path.exists() is False:
        os.mkdir(new_path)
    return new_path


def values_to_hash(val_vec):
    """Given a single data point. Will return hash value string and hash integer vector for the single data point."""
    hash_value_string = ""
    val_sum = np.sum(val_vec)
    for kk in range(0, val_vec.shape[0]):
        val = val_vec[kk] / val_sum
        str = "%i" % (np.round(val * 10))
        hash_value_string += str
    return hash_value_string


def values_to_hash_all(data_array: np.ndarray):
    """Given the data. Will return hash value string and hash integer vectors for all example."""
    hash_string_all = []
    for kk in range(0, data_array.shape[0]):
        hash_value_string = values_to_hash(data_array[kk, :])
        hash_string_all.append(hash_value_string)
    return hash_string_all


# def hash_probability_dist_all(input_data: np.ndarray, dist_orig: np.ndarray, num_bins: int):
#     hash_values = values_to_hash_all(input_data)
#     match_all = []
#     for kk in range(0, len(hash_values)):
#         for jj in range(kk + 1, len(hash_values)):
#             if hash_values[kk] == hash_values[jj]:
#                 match_all.append(1)
#             else:
#                 match_all.append(0)
#     match_all = np.asarray(match_all)
#     arg_sort = np.argsort(dist_orig)
#     dist_orig = dist_orig[arg_sort]
#     match_all = match_all[arg_sort]
#     bin_size = int(np.floor(dist_orig.shape[0] / num_bins))
#     middle_dist = []
#     match_ratio = []
#     for kk in range(0, num_bins):
#         ix_start = kk * bin_size
#         ix_end = (kk + 1) * bin_size
#         if kk == num_bins - 1:
#             ix_end = -1
#         mean_dist = np.mean(dist_orig[ix_start:ix_end])
#         ratio = np.sum(match_all[ix_start:ix_end]) / match_all[ix_start:ix_end].shape[0]
#         middle_dist.append(mean_dist)
#         match_ratio.append(ratio)
#     return middle_dist, match_ratio, hash_values
# def hash_probability_dist_all(input_data: np.ndarray, dist_orig: np.ndarray, bin_list: List = [0.5, 0.6, 0.7, 0.8, 0.9, 1.0]):
#     hash_values = values_to_hash_all(input_data)
#     match_all = []
#     for kk in range(0, len(hash_values)):
#         for jj in range(kk + 1, len(hash_values)):
#             if hash_values[kk] == hash_values[jj]:
#                 match_all.append(1)
#             else:
#                 match_all.append(0)
#     match_all = np.asarray(match_all)
#     arg_sort = np.argsort(dist_orig)
#     dist_orig = dist_orig[arg_sort]
#     match_all = match_all[arg_sort]
#     middle_dist = []
#     match_ratio = []
#     for kk in range(0, len(bin_list) - 1):
#         ix_start = np.argmin(np.abs(dist_orig - bin_list[kk]))
#         ix_end = np.argmin(np.abs(dist_orig - bin_list[kk + 1]))
#         mean_dist = np.mean(dist_orig[ix_start:ix_end])
#         ratio = np.sum(match_all[ix_start:ix_end]) / match_all[ix_start:ix_end].shape[0]
#         middle_dist.append(mean_dist)
#         match_ratio.append(ratio)
#     return middle_dist, match_ratio, hash_values
def hash_probability_dist_all(input_data: np.ndarray, dist_orig: np.ndarray, bin_list: List = [0.5, 0.6, 0.7, 0.8, 0.9, 1.0]):
    match_all = []
    for kk in range(0, input_data.shape[0]):
        for jj in range(kk + 1, input_data.shape[0]):
            # if np.max(np.abs(input_data[kk, :] - input_data[jj, :])) < 0.025:
            if np.max(np.abs(input_data[kk, :] - input_data[jj, :])) < 0.01:
                match_all.append(1)
            else:
                match_all.append(0)
    match_all = np.asarray(match_all)
    arg_sort = np.argsort(dist_orig)
    dist_orig = dist_orig[arg_sort]
    match_all = match_all[arg_sort]
    middle_dist = []
    match_ratio = []
    for kk in range(0, len(bin_list) - 1):
        ix_start = np.argmin(np.abs(dist_orig - bin_list[kk]))
        ix_end = np.argmin(np.abs(dist_orig - bin_list[kk + 1]))
        mean_dist = np.mean(dist_orig[ix_start:ix_end])
        ratio = np.sum(match_all[ix_start:ix_end]) / match_all[ix_start:ix_end].shape[0]
        middle_dist.append(mean_dist)
        match_ratio.append(ratio)
    return middle_dist, match_ratio


def compute_unique_fraction(hash_values_str_list: List):
    """Given the hash values string list. Will compute the collision fraction."""
    num_unique = len(Counter(hash_values_str_list).keys())
    num_sampled = len(hash_values_str_list)
    unique_fraction = num_unique / num_sampled
    return unique_fraction


def fit_ml_model(input_data: np.ndarray, output_data_vector: np.ndarray) -> object:
    """Given input and output data. Will fit a simple ML model."""
    scaler = StandardScaler()
    scaler.fit(input_data)
    data_inputs_scaled = scaler.transform(input_data)
    ml_model = KNeighborsClassifier(n_neighbors=1)
    ml_model.fit(data_inputs_scaled, output_data_vector)
    y_mean = ml_model.predict(data_inputs_scaled)
    train_accuracy = accuracy_score(y_mean, output_data_vector)
    return scaler, ml_model , train_accuracy


def ml_model_predict(scaler: object, ml_model: object, new_input: np.ndarray):
    """Given trained GRP and new input data. Will return predictions."""
    data_inputs_scaled = scaler.transform(new_input)
    y_mean = ml_model.predict(data_inputs_scaled)
    return y_mean


def evaluate_LOOCV(input_data: np.ndarray, output_data_vector: np.ndarray) -> np.double:
    """Given data. Will evaluate NN model via LOOCV."""
    loo = LeaveOneOut()
    predictions = []
    truth = []
    for i, (train_index, test_index) in enumerate(loo.split(input_data)):
        input_train = input_data[train_index, :]
        output_train = output_data_vector[train_index]
        input_test = input_data[test_index, :]
        output_test = output_data_vector[test_index]
        scaler, ml_model, _ = fit_ml_model(input_train, output_train)
        predict_test = ml_model_predict(scaler, ml_model, input_test)
        predictions.append(predict_test)
        truth.append(output_test)
    predictions = np.asarray(predictions)
    truth = np.asarray(truth)
    accuracy = accuracy_score(predictions, truth)
    return accuracy, predictions, truth


def ensemble_evaluate_LOOCV(input_data_all: List, output_data_vector: np.ndarray):
    num_predictions = output_data_vector.shape[0]
    num_devices = len(input_data_all)
    predictions_all = np.zeros((num_predictions, num_devices))
    for kk in range(0, len(input_data_all)):
        input_data = input_data_all[kk]
        _, predictions, _ = evaluate_LOOCV(input_data, output_data_vector)
        predictions_all[:, kk] = predictions[:, 0]
    predictions_vote = []
    for kk in range(0, num_predictions):
        predictions_vote.append(stats.mode(predictions_all[kk, :]).mode[0])
    accuracy = accuracy_score(predictions_vote, output_data_vector)
    return accuracy


def measure_performance(input_data: np.ndarray, output_data_vector: np.ndarray) -> float:
    """Given the data and the target labels. Will compute LOOCV accuracy and the fraction of unique hashes."""
    hash_string_all, hash_value_array = values_to_hash_all(input_data)
    unique_fraction = compute_unique_fraction(hash_string_all)
    accuracy, predictions, _ = evaluate_LOOCV(hash_value_array, output_data_vector)
    return accuracy, unique_fraction, (predictions, hash_string_all, hash_value_array)


def dist_vector_euclidean(vec1: np.ndarray, vec2: np.ndarray):
    dist = np.sum((vec1 - vec2) ** 2.0) ** 0.5
    return dist


def dist_vector_cosine_similarity(vec1: np.ndarray, vec2: np.ndarray):
    dist = np.dot(vec1, vec2) / (np.linalg.norm(vec1) * np.linalg.norm(vec2))
    return dist


def dist_vector_L_infty(vec1: np.ndarray, vec2: np.ndarray):
    dist = np.max(np.abs(vec1 - vec2))
    return dist


def dist_vector_all(input_data: np.ndarray, dist_type: int = 1):
    dist_all = []
    for kk in range(0, input_data.shape[0]):
        for jj in range(kk + 1, input_data.shape[0]):
            if dist_type == 1:
                dist = dist_vector_L_infty(input_data[kk, :], input_data[jj, :])
            elif dist_type == 2:
                dist = dist_vector_cosine_similarity(input_data[kk, :], input_data[jj, :])
            else:
                dist = dist_vector_euclidean(input_data[kk, :], input_data[jj, :])
            dist_all.append(dist)
    return np.asarray(dist_all)


def compute_correlation(input_data_orig: np.ndarray, input_data_hash: np.ndarray):
    dist_orig = dist_vector_all(input_data_orig)
    dist_hash = dist_vector_all(input_data_hash)
    res = stats.spearmanr(dist_orig, dist_hash)
    return res.correlation, dist_orig, dist_hash


def compute_performance_direct(all_loads: List):
    input_data = []
    output_data = []
    for jj in range(0, len(all_loads)):
        for kk in range(0, 20):
            load_beam = all_loads[jj][kk, :]
            input_data.append(load_beam)
            output_data.append(kk + 1)
    input_data = np.asarray(input_data)
    output_data = np.asarray(output_data)
    accuracy, predictions, _ = evaluate_LOOCV(input_data, output_data)
    spearmanr = 1.0
    return accuracy, spearmanr, input_data, output_data, predictions


def compute_performance_simply_supported(all_loads: List, input_data_orig: np.ndarray, num_pts: int = 200) -> float:
    input_data = []
    output_data = []
    Lmin, Lmax, _ = al.get_constants()
    x_beam = np.linspace(Lmin, Lmax, num_pts)
    for jj in range(0, 5):
        for kk in range(0, 20):
            load_beam = all_loads[jj][kk, :]
            Ay, By = mechHS_simply_supported(x_beam, load_beam)
            input_data.append([Ay, By])
            output_data.append(kk + 1)
    input_data = np.asarray(input_data)
    output_data = np.asarray(output_data)
    accuracy, _, _ = measure_performance(input_data, output_data)
    spearmanr, _, _ = compute_correlation(input_data_orig, input_data)
    return accuracy, spearmanr


def compute_performance(input_data: np.ndarray, input_data_orig: np.ndarray, output_data: np.ndarray):
    accuracy, _, _ = evaluate_LOOCV(input_data, output_data)
    spearmanr, _, _ = compute_correlation(input_data_orig, input_data)
    return accuracy, spearmanr



def compute_performance_rectangle_sweep(mypath: Path, input_data_orig: np.ndarray):
    save_path = create_folder(mypath, "FEA_results_rectangle_summarized")
    FEA_RES_accuracy = []
    FEA_RES_spearman = []
    labels = [i for i in range(1, 21)]
    outputs = labels + labels + labels + labels + labels
    output_data = np.asarray(outputs)
    for depth_num in [1, 2, 3, 4, 5]:
        for sensor_num in [1, 2, 3, 4, 5]:
            for fix_whole_bottom in [True, False]:
                if sensor_num == 1 and fix_whole_bottom is False:
                    # this scenario was skipped
                    continue
                fname = "depth_num%i_sensor_num%i_fix_whole_bottom%i.txt" % (depth_num, sensor_num, fix_whole_bottom)
                fname = save_path.joinpath(fname).resolve()
                input_data = np.loadtxt(str(fname))
                # accuracy, _, (_, _, hash_value_array) = measure_performance(input_data, output_data)
                # spearmanr_corr, _, _ = compute_correlation(input_data_orig, hash_value_array)
                accuracy, _, _ = evaluate_LOOCV(input_data, output_data)
                spearmanr_corr, _, _ = compute_correlation(input_data_orig, input_data)
                FEA_RES_accuracy.append(accuracy)
                FEA_RES_spearman.append(spearmanr_corr)
    fname_accuracy = save_path.joinpath("accuracy.txt").resolve()
    np.savetxt(str(fname_accuracy), np.asarray(FEA_RES_accuracy))
    fname_spearman = save_path.joinpath("spearman.txt").resolve()
    np.savetxt(str(fname_spearman), np.asarray(FEA_RES_spearman))
    return


def custom_devices(input_data: np.ndarray, output_data: np.ndarray, input_data_orig: np.ndarray):
    accuracy, _, (_, _, hash_value_array) = measure_performance(input_data, output_data)
    spearmanr, _, _ = compute_correlation(input_data_orig, hash_value_array)
    depth = 10
    return accuracy, spearmanr, depth


def custom_devices_many_steps(step_list: List, folder_name, device_name, output_data, input_data_orig):
    ad_1 = []
    sd_1 = []
    dd_1 = []
    for num_steps in step_list:
        input_data_device_1 = np.loadtxt(folder_name + "/" + device_name + "_step_num%i.txt" % (num_steps))
        accuracy_device_1, spearmanr_device_1, depth_device_1 = custom_devices(input_data_device_1, output_data, input_data_orig)
        ad_1.append(accuracy_device_1)
        sd_1.append(spearmanr_device_1)
        dd_1.append(depth_device_1)
    return ad_1, sd_1, dd_1


def custom_devices_many_steps_v2(step_list: List, folder_name, device_name, output_data, input_data_orig):
    ad_1 = []
    sd_1 = []
    dd_1 = []
    dist_hash_all = []
    for num_steps in step_list:
        input_data_device_1 = np.loadtxt(folder_name + "/" + device_name + "_step_num%i.txt" % (num_steps))
        accuracy, _, _ = evaluate_LOOCV(input_data_device_1, output_data)
        spearmanr, dist_orig, dist_hash = compute_correlation(input_data_orig, input_data_device_1)
        dist_hash_all.append(dist_hash)
        depth = 10
        ad_1.append(accuracy)
        sd_1.append(spearmanr)
        dd_1.append(depth)
    return ad_1, sd_1, dd_1, dist_orig, dist_hash_all

