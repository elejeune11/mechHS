import matplotlib.pyplot as plt
import numpy as np
from typing import List
# import problem_setup_fcns as fcns


def visualize_P2_vs_N():
    L = 10
    num_pts = 100000
    N_list = [10, 15, 20, 25, 30, 35, 40, 45, 50, 60, 70, 80, 90, 100]
    m_list = [0.025, 0.05, 0.1]
    m_res_all = []
    for m in m_list:
        P2_list = []
        Lm = m * L
        for kk in range(0, len(N_list)):
            num_hits = 0
            N = N_list[kk]
            barrier_lower = L / N * (np.floor(N / 2) - 1)
            barrier_upper = barrier_lower + L / N * 3
            for kk in range(0, num_pts):
                pt_1 = np.random.random() * L
                pt_2 = np.random.random() * L
                while np.abs(pt_2 - pt_1) < Lm:
                    pt_2 = np.random.random() * L
                pt_3 = np.random.random() * L
                while np.abs(pt_2 - pt_3) < Lm or np.abs(pt_1 - pt_3) < Lm:
                    pt_3 = np.random.random() * L
                if (pt_1 > barrier_lower and pt_1 < barrier_upper) or (pt_2 > barrier_lower and pt_2 < barrier_upper) or (pt_3 > barrier_lower and pt_3 < barrier_upper):
                    num_hits += 1
            P2_list.append((num_pts - num_hits)/num_pts)
        m_res_all.append(P2_list)
    N_list = np.asarray(N_list)
    P_m0 = ((N_list - 3) / N_list) ** 3.0
    plt.figure()
    plt.plot(N_list, P_m0, "k--o", label="m=0")
    for kk in range(0, len(m_res_all)):
        plt.plot(N_list, m_res_all[kk], "-o", color=plt.cm.autumn(kk / (len(m_res_all) - 1)), label="m=%04f" % (m_list[kk]))
    plt.legend()
    plt.xlabel("N")
    plt.ylabel("P_2")
    # TODO: fix where file gets saved!
    # fig_path = str(fcns.create_folder(mypath, "figures"))
    # fig_path = "figures"
    # plt.savefig(fig_path + "/P2.png")
    plt.savefig("P2.png")
    return


def mechHS_simply_supported(load_x: np.ndarray, load: np.ndarray, Fp: float = 0, a: float = None, b: float = None) -> float:
    """Given load_x and load arrays across the length of the beam.
    Will compute the reactions Ay and By as a simply supported beam.
    Direction of point load is + if pointing up, - if pointing down."""
    if a is None:
        a = np.min(load_x)
    if b is None:
        b = np.max(load_x)
    Fr = np.trapz(load, x=load_x)
    area_x = np.trapz(load * load_x, x=load_x)
    if Fr == 0:
        x_bar = (b - a) / 2.0
    else:
        x_bar = area_x / Fr
    By = Fr * (x_bar - a) / (b - a)
    Ay = Fr - Fp - By
    return Ay, By


def mechHS_simply_supported_composite(support_list: List, load_x_list: List, load_list: List):
    num_segments = len(support_list) - 1
    # for segment 0, support can be anywhere along the length
    # for segment num_segments - 1, support can be anywhere along the length
    # for all intermediate segments support lines up with the composite transition
    support_forces = []
    if num_segments == 1:
        s1 = support_list[0]
        s2 = support_list[1]
        Fp = 0
        load_x = load_x_list[0]
        load = load_list[0]
        S1, S2 = mechHS_simply_supported(load_x, load, Fp, s1, s2)
        support_forces.append(S1)
        support_forces.append(S2)
    else:
        for kk in range(0, num_segments):
            s1 = support_list[kk]
            s2 = support_list[kk + 1]
            load_x = load_x_list[kk]
            load = load_list[kk]
            if kk == 0:
                Fp = 0
                S1, S2 = mechHS_simply_supported(load_x, load, Fp, s1, s2)
                support_forces.append(S1)
                Fp = -1.0 * S2
            elif kk == num_segments - 1:
                S1, S2 = mechHS_simply_supported(load_x, load, Fp, s1, s2)
                support_forces.append(S1)
                support_forces.append(S2)
            else:
                S1, S2 = mechHS_simply_supported(load_x, load, Fp, s1, s2)
                support_forces.append(S1)
                Fp = -1.0 * S2
    return support_forces


def run_simply_supported_composite(input_data_orig: np.ndarray, Lmin: float, Lmax: float, support_list: List):
    num_pts = input_data_orig.shape[1]
    x_beam = np.linspace(Lmin, Lmax, num_pts)
    # break everything up according to the support locations
    # first segment: Lmin - support_list[0]
    # last segment: support_list[-1] - Lmax
    ix_x_beam = []
    ix_x_beam.append(0)
    num_segments = len(support_list) - 1
    if num_segments == 1:
        ix_x_beam.append(num_pts)
    else:
        for kk in range(1, num_segments + 1):
            if kk == (num_segments):
                ix_x_beam.append(num_pts)
            else:
                loc = support_list[kk]
                arg = np.argmin(np.abs(x_beam - loc))
                ix_x_beam.append(arg)
    load_x_list = []
    for kk in range(0, num_segments):
        ix0 = ix_x_beam[kk]
        ix1 = ix_x_beam[kk + 1]
        load_x_list.append(x_beam[ix0:ix1])
    support_forces_all = []
    for kk in range(input_data_orig.shape[0]):
        load_list = []
        for jj in range(0, num_segments):
            ix0 = ix_x_beam[jj]
            ix1 = ix_x_beam[jj + 1]
            load_list.append(input_data_orig[kk, ix0:ix1])
        support_forces = mechHS_simply_supported_composite(support_list, load_x_list, load_list)
        support_forces_all.append(support_forces)
    return np.asarray(support_forces_all)


def check_all_dist(pt_new: float, pt_old_list: List, mL: float):
    for pt_old in pt_old_list:
        if np.abs(pt_old - pt_new) < mL:
            return False
    return True


def generate_new_mechHS(Lmin: float, Lmax: float, m: float, num_supports: int):
    L = Lmax - Lmin
    mL = m * L
    # generate all supports
    support_list = []
    pt_0 = np.random.random() * L + Lmin
    support_list.append(pt_0)
    for _ in range(1, num_supports):
        pt_new = np.random.random() * L + Lmin
        while check_all_dist(pt_new, support_list, mL) is False:
            pt_new = np.random.random() * L + Lmin
        support_list.append(pt_new)
    return support_list


def generate_ensemble_mechHS(Lmin: float, Lmax: float, m: float, num_supports: int, num_examples: int, seed: int):
    np.random.seed(seed)
    support_list_all = []
    for _ in range(0, num_examples):
        support_list = generate_new_mechHS(Lmin, Lmax, m, num_supports)
        support_list_sorted = np.sort(support_list)
        support_list_all.append(support_list_sorted)
    return support_list_all


def run_ensemble_mechHS(support_list_all: List, input_data_orig: np.ndarray, Lmin: float, Lmax: float):
    results_all = []
    num_examples = len(support_list_all)
    for kk in range(0, num_examples):
        support_list = support_list_all[kk]
        results_single = run_simply_supported_composite(input_data_orig, Lmin, Lmax, support_list)
        results_all.append(results_single)
    return results_all


# Lmin = 0
# Lmax = 10
# input_data_orig = -1.0 * np.ones((5, 1000))
# # support_list = [Lmin + 1, Lmax / 2.0, 3.0 * Lmax / 4.0, Lmax - 0.5]
# # # support_list = [Lmin, Lmax / 2.0, Lmax]
# # support_forces_all = run_simply_supported_composite(input_data_orig, Lmin, Lmax, support_list)
# m = 0.05
# num_supports = 5
# num_examples = 10
# seed = 92
# support_list_all = generate_ensemble_mechHS(Lmin, Lmax, m, num_supports, num_examples, seed)
# results_all = run_ensemble_mechHS(support_list_all, input_data_orig, Lmin, Lmax)
# aa = 44

