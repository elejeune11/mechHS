import matplotlib.pyplot as plt
import numpy as np
import os
from pathlib import Path
import reproduce_figure_fcns as rff


plt.rcParams['text.usetex'] = True


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
# simply supported (2)
#############################################################################################
x0 = 0
y0 = 0
s = 1
r = 1
x1 = 11
y1 = y0
num_segs = 1
plt.figure()
ax = plt.gca()
rff.composite_beam(ax, x0, y0, x1, s, num_segs)
ax.set_aspect("equal")
ax.axis("off")

plt.savefig(str(fig_path) + "/ss_s2.png")
plt.savefig(str(fig_path) + "/ss_s2.pdf")

#############################################################################################
# simply supported (3)
#############################################################################################
x0 = 0
y0 = 0
s = 1
r = 1
x1 = 11
y1 = y0
num_segs = 2
plt.figure()
ax = plt.gca()
rff.composite_beam(ax, x0, y0, x1, s, num_segs)
ax.set_aspect("equal")
ax.axis("off")

plt.savefig(str(fig_path) + "/ss_s3.png")
plt.savefig(str(fig_path) + "/ss_s3.pdf")

#############################################################################################
# plot of P2 vs. N
#############################################################################################

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
siz = (3.5, 3.5)
plt.figure(figsize=siz)
plt.plot(N_list, P_m0, "k-o", label="m=0")
label_list = ["m=0.025", "m=0.05", "m=0.1"]
for kk in range(0, len(m_res_all)):
    val = (kk + 1) / (4)
    plt.plot(N_list, m_res_all[kk], "-o", color=(val, val, val), label=label_list[kk])
plt.legend()
plt.xlabel(r"$N$")
plt.ylabel(r"$p_{collision}$")
plt.title(r"$p_2$ vs. $N$ for $\mathcal{F}_{ss-c3}$")
plt.grid(True)
plt.tight_layout()
plt.savefig(str(fig_path) + "/P2.png")
plt.savefig(str(fig_path) + "/P2.pdf")
plt.savefig(str(fig_path) + "/P2.eps")


daa = 4