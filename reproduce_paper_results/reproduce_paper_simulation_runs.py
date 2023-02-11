import applied_loads as al
import FEA_code_for_workstation_custom as FEA_custom
import FEA_code_for_workstation_lattice as FEA_lattice
import FEA_code_for_workstation_rectangle as FEA_rect
import matplotlib.pyplot as plt
import mesh_generation as mg
import numpy as np
import os
from pathlib import Path
import problem_setup_fcns as fcns
import simply_supported_beams as ssb

visualize_loads = False
create_mesh_rectangle = False
create_mesh_lattice = False
create_mesh_devices = False
run_FEA_rectangle = False
create_mesh_lattice = False
run_FEA_lattice = False
create_custom_grid = False
create_mesh_device = False
run_FEA_device = False
run_simply_supported = False
run_simply_supported_ensemble = False
run_simply_supported_678910 = False
run_simply_supported_ensemble_678910 = True

mypath = Path(__file__).resolve().parent

# visualize all loads
if visualize_loads:
    num_pts = 200
    all_loads = al.compute_all_loads(num_pts)
    al.visualize_all_loads(all_loads)


# create FEA meshes for rectangular domains
if create_mesh_rectangle:
    mesh_param = 200
    sensor_num = 1
    for depth_num in [1, 2, 3, 4, 5]:
        fname = "rectangle_depthnum%i" % (depth_num)
        x0, x1, y0, y1, _, _, _ = mg.get_rectangle_domain_constants(depth_num, sensor_num)
        mg.create_mesh_rectangle(mypath, fname, mesh_param, x0, y0, x1, y1)

# run FEA simulation for rectangular domains
if run_FEA_rectangle:
    qsub_inputs, qsub_inputs_list, counter = FEA_rect.get_inputs_for_qsub_script()  # we used this function to specify inputs for our qsub script to run these simulations in parallel on our workstation.
    for input in qsub_inputs_list:  # note, this will run 8,000 simulations in series on your machine. We recommend doing this in a different way.
        os.system("python3 FEA_code_for_workstation_rectangle.py " + input)
    qsub_inputs, qsub_inputs_list, counter = FEA_rect.get_inputs_for_qsub_script(True)  # we used this function to specify inputs for our qsub script to run these simulations in parallel on our workstation.
    for input in qsub_inputs_list:  # note, this will run 8,000 simulations in series on your machine. We recommend doing this in a different way.
        os.system("python3 FEA_code_for_workstation_rectangle.py " + input)

# create FEA mesh for lattice domains
if create_mesh_lattice:
    for size in [3, 4, 5]:
        mesh_param = 200
        fname = "lattice_size%i" % (size)
        mg.create_mesh_lattice(mypath, fname, mesh_param, size)

# run FEA simulations for lattice domains
if run_FEA_lattice:
    qsub_inputs, qsub_inputs_list, counter = FEA_lattice.get_inputs_for_qsub_script()  # we used this function to specify inputs for our qsub script to run these simulations in parallel on our workstation.
    for input in qsub_inputs_list:  # note, this will run 1,200 simulations in series on your machine. We recommend doing this in a different way.
        os.system("python3 FEA_code_for_workstation_lattice.py " + input)

# create FEA mesh for custom domains
if create_custom_grid:
    num_grid_row = 25
    num_grid_col = 25
    fname_grid = "grid_%ix%i" % (num_grid_row, num_grid_col)
    mg.create_blank_grid(mypath, num_grid_row, num_grid_col, fname_grid)

# this is where you fill in the blank grids following the instructions in the README.md file

if create_mesh_device:
    for device in [1, 2, 3]:
        fname = "grid_25x25_device%i" % (device)
        mesh_param = 200
        row_dim = 10
        col_dim = 10
        num_grid_row = 25
        num_grid_col = 25
        geom_mat, xx_sensor, yy_sensor, height = mg.read_design_grid(mypath, num_grid_row, num_grid_col, fname, 10, 10)
        mg.create_mesh_design_grid(mypath, fname, mesh_param, geom_mat, row_dim, col_dim)

# run FEA simulations for custom domains
if run_FEA_device:
    qsub_inputs, qsub_inputs_list, counter = FEA_custom.get_inputs_for_qsub_script()  # we used this function to specify inputs for our qsub script to run these simulations in parallel on our workstation.
    for input in qsub_inputs_list:  # note, this will run 1,200 simulations in series on your machine. We recommend doing this in a different way.
        os.system("python3 FEA_code_for_workstation_custom.py " + input)

# run simply supported
if run_simply_supported:
    Lmin, Lmax, _ = al.get_constants()
    # 2 supports -- at the ends
    support_list_2 = [Lmin, Lmax]
    # 3 supports -- at the ends + evenly spaced
    support_list_3 = [Lmin, (Lmax - Lmin) / 2.0 + Lmin, Lmax]
    # 4 supports -- at the ends + evenly spaced
    support_list_4 = [Lmin, (Lmax - Lmin) / 3.0 + Lmin, 2.0 * (Lmax - Lmin) / 3.0 + Lmin, Lmax]
    # 5 supports -- at the ends + evenly spaced
    support_list_5 = [Lmin, (Lmax - Lmin) / 4.0 + Lmin, 2.0 * (Lmax - Lmin) / 4.0 + Lmin, 3.0 * (Lmax - Lmin) / 4.0 + Lmin, Lmax]
    # compute all loads
    num_pts = 10000
    all_loads = al.compute_all_loads(num_pts)
    input_data = []
    for jj in range(0, len(all_loads)):
        for kk in range(0, 20):
            load_beam = all_loads[jj][kk, :]
            input_data.append(load_beam)
    input_data = np.asarray(input_data)
    # find the support reactions for all loads
    folder = fcns.create_folder(mypath, "SS_results_summarized")
    ssb_results_2 = ssb.run_simply_supported_composite(input_data, Lmin, Lmax, support_list_2)
    ssb_results_3 = ssb.run_simply_supported_composite(input_data, Lmin, Lmax, support_list_3)
    ssb_results_4 = ssb.run_simply_supported_composite(input_data, Lmin, Lmax, support_list_4)
    ssb_results_5 = ssb.run_simply_supported_composite(input_data, Lmin, Lmax, support_list_5)
    np.savetxt(str(folder) + "/ssb_results_2.txt", np.asarray(ssb_results_2))
    np.savetxt(str(folder) + "/ssb_results_3.txt", np.asarray(ssb_results_3))
    np.savetxt(str(folder) + "/ssb_results_4.txt", np.asarray(ssb_results_4))
    np.savetxt(str(folder) + "/ssb_results_5.txt", np.asarray(ssb_results_5))
    np.savetxt(str(folder) + "/ssb_input_data.txt", input_data)


if run_simply_supported_ensemble:
    Lmin, Lmax, _ = al.get_constants()
    num_examples = 100
    m = 0.05
    folder = fcns.create_folder(mypath, "SS_results_summarized")
    input_data = np.loadtxt(str(folder) + "/ssb_input_data.txt")
    seed = 33111
    num_supports = 3
    support_list_3 = ssb.generate_ensemble_mechHS(Lmin, Lmax, m, num_supports, num_examples, seed)
    seed = 33112
    num_supports = 4
    support_list_4 = ssb.generate_ensemble_mechHS(Lmin, Lmax, m, num_supports, num_examples, seed)
    seed = 33113
    num_supports = 5
    support_list_5 = ssb.generate_ensemble_mechHS(Lmin, Lmax, m, num_supports, num_examples, seed)
    ensemble_ssb_results_3 = ssb.run_ensemble_mechHS(support_list_3, input_data, Lmin, Lmax)
    ensemble_ssb_results_4 = ssb.run_ensemble_mechHS(support_list_4, input_data, Lmin, Lmax)
    ensemble_ssb_results_5 = ssb.run_ensemble_mechHS(support_list_5, input_data, Lmin, Lmax)
    np.save(str(folder) + "/ensemble_ssb_results_3.npy", ensemble_ssb_results_3)
    np.save(str(folder) + "/ensemble_ssb_results_4.npy", ensemble_ssb_results_4)
    np.save(str(folder) + "/ensemble_ssb_results_5.npy", ensemble_ssb_results_5)


if run_simply_supported_678910:
    Lmin, Lmax, _ = al.get_constants()
    # 6 supports -- at the ends
    support_list_6 = [Lmin, (Lmax - Lmin) / 5.0 + Lmin, 2.0 * (Lmax - Lmin) / 5.0 + Lmin, 3.0 * (Lmax - Lmin) / 5.0 + Lmin, 4.0 * (Lmax - Lmin) / 5.0 + Lmin, Lmax]
    # 7 supports -- at the ends
    support_list_7 = []
    for kk in range(0, 7):
        support_list_7.append(Lmin + (Lmax - Lmin) / 6.0 * kk)
    # 8 supports -- at the ends
    support_list_8 = []
    for kk in range(0, 8):
        support_list_8.append(Lmin + (Lmax - Lmin) / 7.0 * kk)
    # 9 supports -- at the ends
    support_list_9 = []
    for kk in range(0, 9):
        support_list_9.append(Lmin + (Lmax - Lmin) / 8.0 * kk)
    # 10 supports -- at the ends
    support_list_10 = []
    for kk in range(0, 10):
        support_list_10.append(Lmin + (Lmax - Lmin) / 9.0 * kk)
    # compute all loads
    num_pts = 10000
    input_data = np.loadtxt(str(folder) + "/ssb_input_data.txt")
    # find the support reactions for all loads
    folder = fcns.create_folder(mypath, "SS_results_summarized")
    ssb_results_6 = ssb.run_simply_supported_composite(input_data, Lmin, Lmax, support_list_6)
    ssb_results_7 = ssb.run_simply_supported_composite(input_data, Lmin, Lmax, support_list_7)
    ssb_results_8 = ssb.run_simply_supported_composite(input_data, Lmin, Lmax, support_list_8)
    ssb_results_9 = ssb.run_simply_supported_composite(input_data, Lmin, Lmax, support_list_9)
    ssb_results_10 = ssb.run_simply_supported_composite(input_data, Lmin, Lmax, support_list_10)
    np.savetxt(str(folder) + "/ssb_results_6.txt", np.asarray(ssb_results_6))
    np.savetxt(str(folder) + "/ssb_results_7.txt", np.asarray(ssb_results_7))
    np.savetxt(str(folder) + "/ssb_results_8.txt", np.asarray(ssb_results_8))
    np.savetxt(str(folder) + "/ssb_results_9.txt", np.asarray(ssb_results_9))
    np.savetxt(str(folder) + "/ssb_results_10.txt", np.asarray(ssb_results_10))


if run_simply_supported_ensemble_678910:
    Lmin, Lmax, _ = al.get_constants()
    num_examples = 100
    m = 0.05
    folder = fcns.create_folder(mypath, "SS_results_summarized")
    input_data = np.loadtxt(str(folder) + "/ssb_input_data.txt")
    seed = 33114
    num_supports = 6
    support_list_6 = ssb.generate_ensemble_mechHS(Lmin, Lmax, m, num_supports, num_examples, seed)
    seed = 33115
    num_supports = 7
    support_list_7 = ssb.generate_ensemble_mechHS(Lmin, Lmax, m, num_supports, num_examples, seed)
    seed = 33116
    num_supports = 8
    support_list_8 = ssb.generate_ensemble_mechHS(Lmin, Lmax, m, num_supports, num_examples, seed)
    seed = 33117
    num_supports = 9
    support_list_9 = ssb.generate_ensemble_mechHS(Lmin, Lmax, m, num_supports, num_examples, seed)
    seed = 33118
    num_supports = 10
    support_list_10 = ssb.generate_ensemble_mechHS(Lmin, Lmax, m, num_supports, num_examples, seed)
    ensemble_ssb_results_6 = ssb.run_ensemble_mechHS(support_list_6, input_data, Lmin, Lmax)
    ensemble_ssb_results_7 = ssb.run_ensemble_mechHS(support_list_7, input_data, Lmin, Lmax)
    ensemble_ssb_results_8 = ssb.run_ensemble_mechHS(support_list_8, input_data, Lmin, Lmax)
    ensemble_ssb_results_9 = ssb.run_ensemble_mechHS(support_list_9, input_data, Lmin, Lmax)
    ensemble_ssb_results_10 = ssb.run_ensemble_mechHS(support_list_10, input_data, Lmin, Lmax)
    np.save(str(folder) + "/ensemble_ssb_results_6.npy", ensemble_ssb_results_6)
    np.save(str(folder) + "/ensemble_ssb_results_7.npy", ensemble_ssb_results_7)
    np.save(str(folder) + "/ensemble_ssb_results_8.npy", ensemble_ssb_results_8)
    np.save(str(folder) + "/ensemble_ssb_results_9.npy", ensemble_ssb_results_9)
    np.save(str(folder) + "/ensemble_ssb_results_10.npy", ensemble_ssb_results_10)


