import matplotlib.pyplot as plt
import numpy as np
import os
from pathlib import Path
import simply_supported_beams as ssb



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
#############################################################################################

la_val = np.linspace(0, 1, 11)
lb_val = np.linspace(0, 1, 11)

la_all, lb_all = np.meshgrid(la_val, lb_val)
amp_all = np.zeros(la_all.shape)

Fp = 0
load_x = np.linspace(0, 1, 1001)
load = np.ones(1001)

for kk in range(0, la_all.shape[0]):
    for jj in range(0, la_all.shape[1]):
        la = la_all[kk, jj]
        lb = lb_all[kk, jj]
        if la == lb:
            continue
        Ay, By = ssb.mechHS_simply_supported(load_x, load, Fp, la, lb)
        amp_all[kk, jj] = np.max([np.abs(Ay), np.abs(By)])

siz = (4.0 / 3.0 * 3.5, 2.5)

plt.figure(figsize=siz)
plt.imshow(amp_all, cmap=plt.cm.cividis)
plt.plot(np.linspace(0, 10, 11), np.linspace(0, 10, 11), "wx")
plt.xlabel(r"support position $l_a$")
plt.ylabel(r"support position $l_b$")
ax = plt.gca()
ax.set_xticks([0, 10])
ax.set_xticklabels([r"$0$", r"$L$"])
ax.set_yticks([0, 10])
ax.set_yticklabels([r"$0$", r"$L$"])
ax.invert_yaxis()

cbar = plt.colorbar(location="left", pad=0.15, fraction=0.1)
cbar.set_ticks([0, np.max(amp_all)])
cbar.set_ticklabels([r"$0$", r"$\bigg|\frac{RL}{2m}\bigg|$"])
plt.tight_layout()
plt.savefig(str(fig_path) + "/max_amp_2.pdf")

aa = 44


#############################################################################################
#############################################################################################
# !! start here
la_val = np.linspace(0, 1, 11)
lb_val = np.linspace(0, 1, 11)
lc_val = np.linspace(0, 1, 11)

la_all, lb_all, lc_all = np.meshgrid(la_val, lb_val, lc_val)
amp_all = np.zeros(la_all.shape)

Lmin = 0.0
Lmax = 1.0
input_data_orig = np.ones((1, 1001))

for kk in range(0, la_all.shape[0]):
    for jj in range(0, la_all.shape[1]):
        for pp in range(0, la_all.shape[2]):
            if kk == jj or kk == pp or jj == pp:
                continue
            support_list = [la_all[kk, jj, pp], lb_all[kk, jj, pp], lc_all[kk, jj, pp]]
            support_list = np.sort(support_list)
            support_forces = ssb.run_simply_supported_composite(input_data_orig, Lmin, Lmax, support_list)
            amp_all[kk, jj, pp] = np.max(np.abs(support_forces))


plt.figure(figsize=siz)
plt.imshow(amp_all[0, : , :], cmap=plt.cm.cividis)
ax = plt.gca()
plt.plot(np.linspace(1, 10, 10), np.linspace(1, 10, 10), "wx")
plt.plot(np.linspace(1, 10, 10), np.zeros((10, 1)), "wx")
plt.plot(np.zeros((10, 1)), np.linspace(1, 10, 10), "wx")
plt.plot([0], [0] , "wo", markersize=7)
plt.xlabel(r"support position $l_b$")
plt.ylabel(r"support position $l_c$")
ax = plt.gca()
ax.set_xticks([0, 10])
ax.set_xticklabels([r"$0$", r"$L$"])
ax.set_yticks([0, 10])
ax.set_yticklabels([r"$0$", r"$L$"])
ax.invert_yaxis()

cbar = plt.colorbar(location="left", pad=0.15, fraction=0.1)
cbar.set_ticks([0, np.max(amp_all)])
cbar.set_ticklabels([r"$0$", r"$\bigg|\frac{(L - mL)^2R}{2mL}\bigg|$"])
plt.tight_layout()
plt.savefig(str(fig_path) + "/max_amp_3.pdf")
#############################################################################################
#############################################################################################

aa = 44
