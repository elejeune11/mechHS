import sys
sys.path.append("code")

import fea_simulation as fea
import mesh_generation as mg
import problem_setup_fcns as fcns

import numpy as np
from pathlib import Path

mypath = Path(__file__).resolve().parent

# 1. create geometry for a custom domain
num_grid_row = 25
num_grid_col = 25
fname_grid = "grid_%ix%i" % (num_grid_row, num_grid_col)
mg.create_blank_grid(mypath, num_grid_row, num_grid_col, fname_grid)

# 2. draw a mechHS device
# this you will do offline by filling in the grid

# 3. create FEA mesh for custom domains
fname = "tutorial_example"
mesh_param = 200
row_dim = 10
col_dim = 10
geom_mat, xx_sensor, yy_sensor, height = mg.read_design_grid(mypath, num_grid_row, num_grid_col, fname, row_dim, col_dim)
mg.create_mesh_design_grid(mypath, fname, mesh_param, geom_mat, row_dim, col_dim)

# 4. run FEA simulations -- note this step is default set to False as it is better implemented on a workstation -- we provide example results saved as a text file
if False:
    fix_whole_btm = True
    mypath_res = Path(__file__).resolve().parent.joinpath("FEA_results").resolve()
    mesh_param = 200
    fname_mesh = fname + "_mesh_%i" % (mesh_param)
    save_pvd = False
    step_size = 0.001
    num_steps = 11
    _, xx_sensor_bc_list, yy_sensor_bc_list, height = mg.read_design_grid(mypath, 25, 25, fname, 10, 10)
    for applied_load_type in range(1, 21):
        for applied_load_num in range(1, 21):
            fname_res = fname + "_mesh%i_alt%i_aln%i_force_list.txt" % (mesh_param, applied_load_type, applied_load_num)
            file_res = mypath_res.joinpath(fname_res).resolve()
            if file_res.is_file():
                continue
            fname_save = fname + "_mesh%i_alt%i_aln%i" % (mesh_param, applied_load_type, applied_load_num)
            try:
                fea.run_simulation(mypath, fname_mesh, fname_save, save_pvd, xx_sensor_bc_list, yy_sensor_bc_list, height, fix_whole_btm, applied_load_type, applied_load_num, step_size, num_steps)
            except RuntimeError:
                print("runtime error occured")
                print(RuntimeError)

# 5. import FEA results
data_path = Path(__file__).resolve().parent.joinpath("data").resolve()
data = np.loadtxt(str(data_path) + "/" + fname + ".txt")


def normalize_rows(arr):
    for kk in range(arr.shape[0]):
        arr[kk, :] = arr[kk, :] / np.sum(arr[kk, :])
    return arr


def normalize_rows_list(arr_list):
    for kk in range(0, len(arr_list)):
        arr_list[kk] = normalize_rows(arr_list[kk])
    return arr_list


# 6. assess performance of the MechHS systems
data = normalize_rows(data)
input_data_orig = np.loadtxt(str(data_path) + "/input_data_orig.txt")
output_data_orig = np.loadtxt(str(data_path) + "/output_data_orig.txt")
accuracy, spearmanr = fcns.compute_performance(data, input_data_orig, output_data_orig)
print(fname)
print("accuracy:", accuracy)
print("Spearman's rho:", spearmanr)
