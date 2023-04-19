import matplotlib.pyplot as plt
import numpy as np
import os
from pathlib import Path
from sklearn.metrics import confusion_matrix
from sklearn.metrics import plot_confusion_matrix

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

orig_predictions = np.loadtxt(str(save_path) + "/orig_predictions.txt")

true_predictions = []
for _ in range(0, 20):
    for kk in range(1, 21):
        true_predictions.append(kk)

conf_matrix = confusion_matrix(np.asarray(true_predictions), np.asarray(orig_predictions))

fig, ax = plt.subplots(figsize=(11 / 9 * 6, 6))
cax = ax.matshow(conf_matrix, cmap=plt.cm.Reds, alpha=0.5)
cbar = fig.colorbar(cax, ticks=[0, 5, 10, 15, 20])

for i in range(conf_matrix.shape[0]):
    for j in range(conf_matrix.shape[1]):
        ax.text(x=j, y=i, s=conf_matrix[i, j], va='center', ha='center', size='xx-large')


ticks = []
tick_labels = []
for kk in range(0, 20):
    ticks.append(kk)
    tick_labels.append(str(kk + 1))

ax.set_xticks(ticks)
ax.set_xticklabels(tick_labels)
plt.ylabel("ground truth", fontsize=18)
plt.xlabel("prediction", fontsize=18)
ax.set_yticks(ticks)
ax.set_yticklabels(tick_labels)
plt.tick_params(left=False, right=False, top=False, bottom=False)
plt.title("confusion matrix: original inputs", fontsize=18)
plt.savefig(str(fig_path) + "/confusion_matrix.png")
plt.savefig(str(fig_path) + "/confusion_matrix.eps")

aa = 44