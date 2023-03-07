import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path
from perlin_noise import PerlinNoise
from sklearn.neighbors import KernelDensity
from typing import List


def get_perlin_noise_constants():
    """Will return the information needed to reproduce the noise used in this study."""
    np.random.seed(11)
    octave_seeds = np.random.randint(2, 11, size=400)
    perlin_seeds = np.random.randint(0, 10000, size=400)
    noise_constants = np.zeros((400, 2))
    noise_constants[:, 0] = octave_seeds
    noise_constants[:, 1] = perlin_seeds
    return noise_constants


def get_kde_noise_constants():
    """Will return the information needed to reproduce the kde functions used in this study."""
    np.random.seed(111)
    kde_seeds = np.random.randint(0, 10000, size=9)
    return kde_seeds


def initialize_perlin_noise(octaves: int, seed: int) -> object:
    """Given octaves and seed. Will initialize perlin noise."""
    noise = PerlinNoise(int(octaves), int(seed))
    return noise


def get_perlin_noise_objects(noise_constants):
    dict = {}
    for kk in range(0, noise_constants.shape[0]):
        dict[kk] = initialize_perlin_noise(noise_constants[kk, 0], noise_constants[kk, 1])
    return dict


def initialize_kde_fcn(num_pts: int, seed: int, bandwidth: float = 0.1) -> object:
    """Given the number of points and bandwidth. Will initialize kde function."""
    np.random.seed(int(seed))
    X = np.random.random(num_pts).reshape(-1, 1)
    kde = KernelDensity(kernel="gaussian", bandwidth=bandwidth).fit(X)
    return kde


def get_kde_objects(kde_seeds, bandwidth: float = 0.1) -> object:
    dict = {}
    for kk in range(12, 21):
        np.random.seed(kde_seeds[kk - 12])
        if kk == 12 or kk == 13 or kk == 14:
            X = np.random.random(2).reshape(-1, 1)
        elif kk == 15 or kk == 16 or kk == 17:
            X = np.random.random(5).reshape(-1, 1)
        elif kk == 18 or kk == 19 or kk == 20:
            X = np.random.random(25).reshape(-1, 1)
        kde = KernelDensity(kernel="gaussian", bandwidth=bandwidth).fit(X)
        dict[kk] = kde
    return dict


def compute_load_piecewise(x0: float, constant_load_list: List, linear_load_list: List, TOL: float = 10E-5):
    """Given a position x0 and constant and linear load inputs."""
    val = 0
    for cl in constant_load_list:
        x_min = cl[0] - TOL
        x_max = cl[1] + TOL
        if x0 > x_min and x0 < x_max:
            val += cl[2]
    for ll in linear_load_list:
        x_min = ll[0] - TOL
        x_max = ll[1] + TOL
        if x0 > x_min and x0 < x_max:
            val += ((ll[3] - ll[2]) / (ll[1] - ll[0]) * (x0 - ll[0]) + ll[2])
    return val


def compute_load_perlin(noise: object, x0: float, Lmin: float, Lmax: float, scale_factor: float) -> float:
    """Given the Perlin noise object and a location. Will return the noise value."""
    sample_pt = (x0 - Lmin) / (Lmax - Lmin)
    val = noise(sample_pt) * scale_factor
    return val


def compute_load_kde(kde: object, x0: float, Lmin: float, Lmax: float):
    """Given the kde object and a location. Will return the value."""
    sample_pt = (x0 - Lmin) / (Lmax - Lmin)
    log_dens = kde.score_samples(np.asarray([sample_pt]).reshape(1, -1))
    dens = np.exp(log_dens)
    val = dens * 1.0 / (Lmax - Lmin) * -1.0
    return val


def compute_load_sinusoidal(x0: float, Lmin: float, Lmax: float, wave_num: float, offset: float = 0.0):
    """Given the """
    factor = wave_num / ((Lmax - Lmin) / (np.pi * 2.0))
    x_sin = -1.0 * np.abs(np.sin(factor * (x0) - offset * np.pi * 2.0)) / (4.0 / factor * wave_num)
    return x_sin


def get_dicts():
    noise_constants = get_perlin_noise_constants()
    noise_dict = get_perlin_noise_objects(noise_constants)
    kde_constants = get_kde_noise_constants()
    kde_dict = get_kde_objects(kde_constants)
    return noise_dict, kde_dict


def load_at_a_point(
    x0: float,
    Lmin: float,
    Lmax: float,
    load_type: int,
    load_num: int,
    noise_scale_factor: float,
    noise_dict: dict,
    kde_dict: dict
) -> float:
    """Given """
    if load_type == 1:
        # constant load
        constant_load_list = [[Lmin, Lmax, -1.0 / (Lmax - Lmin)]]
        linear_load_list = []
        curve_value = compute_load_piecewise(x0, constant_load_list, linear_load_list)
    elif load_type == 2:
        # V shaped load
        constant_load_list = []
        linear_load_list = [[Lmin, (Lmax - Lmin) / 2.0, -2.0 / (Lmax - Lmin), 0], [(Lmax - Lmin)/ 2.0, Lmax, 0, -2.0 / (Lmax - Lmin)]]
        curve_value = compute_load_piecewise(x0, constant_load_list, linear_load_list)
    elif load_type == 3:
        # ^ shaped load
        constant_load_list = []
        linear_load_list = [[Lmin, (Lmax - Lmin) / 2.0, 0, -2.0 / (Lmax - Lmin)], [(Lmax - Lmin)/ 2.0, Lmax, -2.0 / (Lmax - Lmin), 0]]
        curve_value = compute_load_piecewise(x0, constant_load_list, linear_load_list)
    elif load_type == 4:
        # \ shaped load
        constant_load_list = []
        linear_load_list = [[Lmin, Lmax, -2.0 / (Lmax - Lmin), 0]]
        curve_value = compute_load_piecewise(x0, constant_load_list, linear_load_list)
    elif load_type == 5:
        # / shaped load
        constant_load_list = []
        linear_load_list = [[Lmin, Lmax, 0, -2.0 / (Lmax - Lmin)]]
        curve_value = compute_load_piecewise(x0, constant_load_list, linear_load_list)
    elif load_type == 6:
        # half abs sin wave
        wave_num = 0.5
        offset = 0
        curve_value = compute_load_sinusoidal(x0, Lmin, Lmax, wave_num, offset)
    elif load_type == 7:
        # abs sin wave
        wave_num = 1
        offset = 0
        curve_value = compute_load_sinusoidal(x0, Lmin, Lmax, wave_num, offset)
    elif load_type == 8:
        # abs sine wave offset
        wave_num = 1
        offset = 0.25
        curve_value = compute_load_sinusoidal(x0, Lmin, Lmax, wave_num, offset)
    elif load_type == 9:
        # 1.5 abs sin wave
        wave_num = 1.5
        offset = 0
        curve_value = compute_load_sinusoidal(x0, Lmin, Lmax, wave_num, offset)
    elif load_type == 10:
        # 1.5 abs sin wave offset
        wave_num = 1.5
        offset = 0.25
        curve_value = compute_load_sinusoidal(x0, Lmin, Lmax, wave_num, offset)
    elif load_type == 11:
        # 2 abs sin wave
        wave_num = 2.0
        offset = 0
        curve_value = compute_load_sinusoidal(x0, Lmin, Lmax, wave_num, offset)
    elif load_type == 12:
        # kde 2, random 1
        kde = kde_dict[load_type]
        curve_value = compute_load_kde(kde, x0, Lmin, Lmax)[0]
    elif load_type == 13:
        # kde 2, random 2
        kde = kde_dict[load_type]
        curve_value = compute_load_kde(kde, x0, Lmin, Lmax)[0]
    elif load_type == 14:
        # kde 2, random 3
        kde = kde_dict[load_type]
        curve_value = compute_load_kde(kde, x0, Lmin, Lmax)[0]
    elif load_type == 15:
        # kde 5, random 1
        kde = kde_dict[load_type]
        curve_value = compute_load_kde(kde, x0, Lmin, Lmax)[0]
    elif load_type == 16:
        # kde 5, random 2
        kde = kde_dict[load_type]
        curve_value = compute_load_kde(kde, x0, Lmin, Lmax)[0]
    elif load_type == 17:
        # kde 5, random 3
        kde = kde_dict[load_type]
        curve_value = compute_load_kde(kde, x0, Lmin, Lmax)[0]
    elif load_type == 18:
        # kde 25, random 1
        kde = kde_dict[load_type]
        curve_value = compute_load_kde(kde, x0, Lmin, Lmax)[0]
    elif load_type == 19:
        # kde 25, random 2
        kde = kde_dict[load_type]
        curve_value = compute_load_kde(kde, x0, Lmin, Lmax)[0]
    elif load_type == 20:
        # kde 25, random 3
        kde = kde_dict[load_type]
        curve_value = compute_load_kde(kde, x0, Lmin, Lmax)[0]
    noise = noise_dict[(load_num - 1) * 20 + (load_type - 1)]
    noise_val = compute_load_perlin(noise, x0, Lmin, Lmax, noise_scale_factor)
    return curve_value + noise_val


def load_across_beam(
    Lmin: float,
    Lmax: float,
    load_type: int,
    load_num: int,
    noise_scale_factor: float,
    noise_dict: dict,
    kde_dict: dict,
    num_pts: int = 200
) -> np.ndarray:
    x_vals = np.linspace(Lmin, Lmax, num_pts)
    load = []
    for kk in range(0, x_vals.shape[0]):
        val = load_at_a_point(x_vals[kk], Lmin, Lmax, load_type, load_num, noise_scale_factor, noise_dict, kde_dict)
        load.append(val)
    return np.asarray(load)


def get_constants():
    Lmin = 0
    Lmax = 10
    noise_scale_factor = 0.25
    return Lmin, Lmax, noise_scale_factor


def compute_all_loads(num_pts: float = 200):
    noise_dict, kde_dict = get_dicts()
    Lmin, Lmax, noise_scale_factor = get_constants()
    all_loads = []
    num_examples = 20
    for jj in range(0, num_examples):
        all_loads_num = np.zeros((20, num_pts))
        for kk in range(0, 20):
            load_type = kk + 1
            load_num = jj + 1
            all_loads_num[kk, :] = load_across_beam(Lmin, Lmax, load_type, load_num, noise_scale_factor, noise_dict, kde_dict, num_pts)
        all_loads.append(all_loads_num)
    return all_loads


def all_loads_to_array(all_loads: List):
    reformat = []
    for kk in range(0, len(all_loads)):
        aa = all_loads[kk]
        for jj in range(0, aa.shape[0]):
            reformat.append(aa[jj, :])
    return np.asarray(reformat)


def visualize_all_loads(all_loads):
    fig, axs = plt.subplots(4, 5, figsize=(40, 20))
    num_examples = 20
    for kk in range(0, num_examples):
        for jj in range(0, 20):
            row_ix = int(np.floor(jj / 5))
            col_ix = jj % 5
            axs[row_ix, col_ix].plot(all_loads[kk][jj, :])
            axs[row_ix, col_ix].set_ylim((-0.4, 0.05))
    plt.savefig("all_loads.png")
    return