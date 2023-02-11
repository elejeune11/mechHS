import applied_loads as al
import fea_simulation as fea
import matplotlib.pyplot as plt
import mesh_generation as mg
import numpy as np
from pathlib import Path
import problem_setup_fcns as fcns


visualize_loads = False
create_mesh_rectangle = False
create_mesh_lattice = False
create_mesh_devices = True
run_example_FEA = False
run_rectangle_FEA_hash = False

mypath = Path(__file__).resolve().parent

# visualize all loads
num_pts = 200  # increase to higher number for convergence w/ SS + direct
all_loads = al.compute_all_loads(num_pts)
if visualize_loads:
    al.visualize_all_loads(all_loads)

# create FEA meshes for rectangular domains
if create_mesh_rectangle:
    mesh_param = 250
    sensor_num = 1
    for depth_num in [1, 2, 3, 4, 5]:
        fname = "rectangle_depthnum%i" % (depth_num)
        x0, x1, y0, y1, _, _, _ = mg.get_rectangle_domain_constants(depth_num, sensor_num)
        mg.create_mesh_rectangle(mypath, fname, mesh_param, x0, y0, x1, y1)

if create_mesh_lattice:
    for size in [2, 3, 4, 5]:
        mesh_param = 200
        fname = "lattice_size%i" % (size)
        mg.create_mesh_lattice(mypath, fname, mesh_param, size)

if create_mesh_devices:
    num_grid_row = 25
    num_grid_col = 25
    # fname_grid = "grid_%ix%i" % (num_grid_row, num_grid_col)
    # mg.create_blank_grid(mypath, num_grid_row, num_grid_col, fname_grid)
    # fname = "grid_25x25_device1"
    # mesh_param = 200
    # row_dim = 10
    # col_dim = 10
    # geom_mat, xx_sensor, yy_sensor, height = mg.read_design_grid(mypath, num_grid_row, num_grid_col, fname, 10, 10)
    # mg.create_mesh_design_grid(mypath, fname, mesh_param, geom_mat, row_dim, col_dim)
    # fname = "grid_25x25_device2"
    # mesh_param = 200
    # row_dim = 10
    # col_dim = 10
    # geom_mat, xx_sensor, yy_sensor, height = mg.read_design_grid(mypath, num_grid_row, num_grid_col, fname, 10, 10)
    # mg.create_mesh_design_grid(mypath, fname, mesh_param, geom_mat, row_dim, col_dim)
    # fname = "grid_25x25_device3"
    # mesh_param = 200
    # row_dim = 10
    # col_dim = 10
    # geom_mat, xx_sensor, yy_sensor, height = mg.read_design_grid(mypath, num_grid_row, num_grid_col, fname, 10, 10)
    # mg.create_mesh_design_grid(mypath, fname, mesh_param, geom_mat, row_dim, col_dim)
    # fname = "grid_25x25_device4"
    # mesh_param = 200
    # row_dim = 10
    # col_dim = 10
    # geom_mat, xx_sensor, yy_sensor, height = mg.read_design_grid(mypath, num_grid_row, num_grid_col, fname, 10, 10)
    # mg.create_mesh_design_grid(mypath, fname, mesh_param, geom_mat, row_dim, col_dim)
    fname = "grid_25x25_device5"
    mesh_param = 200
    row_dim = 10
    col_dim = 10
    geom_mat, xx_sensor, yy_sensor, height = mg.read_design_grid(mypath, num_grid_row, num_grid_col, fname, 10, 10)
    mg.create_mesh_design_grid(mypath, fname, mesh_param, geom_mat, row_dim, col_dim)

# run FEA simulations -- here we show just one example -- see also "FEA_code_for_workstation"
"""
FEA input parameter definitions for reference:

path --> folder path needed to access data
fname_mesh --> file name of the mesh to use
fname_save --> file name to use for saving
save_pvd --> set true if you want to save a Paraview file (note these files may be large)
xx_sensor_bc --> locations where xx force will be recorded (not used)
yy_sensor_bc --> locations where yy force will be recorded (not used)
height --> height of the domain (can vary, bottom is always set at y = 0)
fix_whole_btm --> set True if the whole bottom boundary should be fixed. If False it will only be fixed at the sensors
applied_load_type --> will call the applied load function -- range 1-20
applied_load_num --> will call the applied load function -- range 1-5
step_size --> FEA step size (multiplier on applied load)
num_steps --> number of FEA steps

NOTE: -- if you run the code as is, it will run for one device for 1 simulations. 
We recommend configuring it to run best on your workstation for all 5000+ simulations.
In our work, we run 100 simulations per device
"""
if run_example_FEA:
    mesh_param = 200
    path = mypath
    save_pvd = True
    depth_num = 1
    sensor_num = 5
    _, _, _, _, height, xx_sensor_bc_list, yy_sensor_bc_list = mg.get_rectangle_domain_constants(depth_num, sensor_num)
    fix_whole_btm = False
    applied_load_type = 1
    applied_load_num = 1
    step_size = 0.025
    num_steps = 20
    fname_mesh = "rectangle_depthnum%i_mesh_%i" % (depth_num, mesh_param)
    fname_save = "rectangle_depthnum%i_mesh%i_sensornum%i_fixwholebtm%i_alt%i_aln%i" % (depth_num, mesh_param, sensor_num, fix_whole_btm, applied_load_type, applied_load_num)
    force_list = fea.run_simulation(path, fname_mesh, fname_save, save_pvd, xx_sensor_bc_list, yy_sensor_bc_list, height, fix_whole_btm, applied_load_type, applied_load_num, step_size, num_steps)


# compute hash performance of timeseries directly
# accuracy_raw, spearmanr_raw, input_data_orig_raw = fcns.compute_performance_direct(all_loads)

# # compute hash performance of simply supported beam
# accuracy_ss, spearmanr_ss = fcns.compute_performance_simply_supported(all_loads, input_data_orig_raw, num_pts)

# # compute hash performance based on FEA results -- rectangle
# if run_rectangle_FEA_hash is True:
#     fcns.compute_performance_rectangle_sweep(mypath, input_data_orig_raw)

# save_path = fcns.create_folder(mypath, "FEA_results_rectangle_summarized")
# fname_accuracy = save_path.joinpath("accuracy.txt").resolve()
# accuracy_rect = np.loadtxt(str(fname_accuracy))
# fname_spearman = save_path.joinpath("spearman.txt").resolve()
# spearman_rect = np.loadtxt(str(fname_spearman))

# # compute hash performance based on FEA results -- bespoke devices
# # step_list = [5, 10, 20, 30, 40, 50, 60, 70, 80, 90, 100, 110, 120, 130, 140, 150, 160, 170, 180, 190, 199]
# step_list = [5, 50]

# labels = [i for i in range(1, 21)]
# outputs = labels + labels + labels + labels + labels
# output_data = np.asarray(outputs)
# folder_name = "FEA_results_lattice_summarized"
# device_name = "lattice_5"
# al_5, sl_5, dl_5, _, _ = fcns.custom_devices_many_steps_v2(step_list, folder_name, device_name, output_data, input_data_orig_raw)
# device_name = "lattice_4"
# al_4, sl_4, dl_4, _, _ = fcns.custom_devices_many_steps_v2(step_list, folder_name, device_name, output_data, input_data_orig_raw)


# # step_list = [5, 50] #[5, 10, 15, 20, 25, 30, 35, 40, 45, 50, 55, 60, 65, 70, 75, 80, 85, 90, 95, 100, 105, 110, 115, 120, 125, 130, 135, 140, 145, 150, 155, 160, 165, 170, 175, 180, 185, 190, 195, 200, 205, 210, 215, 220, 225, 230, 235, 240, 245, 250, 255, 260, 265, 270, 275, 280, 285, 290, 295, 300]
# folder_name = "FEA_results_device_summarized"
# device_name = "device_1"
# ad_1, sd_1, dd_1, _, _ = fcns.custom_devices_many_steps_v2(step_list, folder_name, device_name, output_data, input_data_orig_raw)

# folder_name = "FEA_results_device_summarized"
# device_name = "device_2"
# ad_2, sd_2, dd_2, _, _ = fcns.custom_devices_many_steps_v2(step_list, folder_name, device_name, output_data, input_data_orig_raw)

# folder_name = "FEA_results_device_summarized"
# device_name = "device_3"
# ad_3, sd_3, dd_3, _, _ = fcns.custom_devices_many_steps_v2(step_list, folder_name, device_name, output_data, input_data_orig_raw)


# # visualize hash performance for all results
# plt.figure()
# ix = 0
# for depth_num in [1, 2, 3, 4, 5]:
#     for sensor_num in [1, 2, 3, 4, 5]:
#         for fix_whole_bottom in [True, False]:
#             save_info = []
#             if sensor_num == 1 and fix_whole_bottom is False:
#                 continue
#             sp = spearman_rect[ix]
#             ac = accuracy_rect[ix]
#             if fix_whole_bottom:
#                 plt.plot(sp, ac, "o", color=plt.cm.coolwarm(sensor_num / 5), markersize=depth_num * 3)
#             else:
#                 plt.plot(sp, ac, "s", color=plt.cm.coolwarm(sensor_num / 5), markersize=depth_num * 3)
#             ix += 1

# plt.plot(spearmanr_ss, accuracy_ss, "dk", markersize=15, markeredgecolor="y")
# plt.plot(sl_4, al_4, "k-")
# plt.plot(sl_5, al_5, "k-")
# plt.plot(sd_1, ad_1, "k-")
# plt.plot(sd_2, ad_2, "k-")
# plt.plot(sd_3, ad_3, "k-")

# for kk in range(0, len(sl_4)):
#     val = kk / len(sl_4)
#     plt.plot(sl_4[kk], al_4[kk], "*", color=plt.cm.autumn(val), markersize=15, markeredgecolor="y")

# for kk in range(0, len(sl_5)):
#     val = kk / len(sl_5)
#     plt.plot(sl_5[kk], al_5[kk], "*", color=plt.cm.spring(val), markersize=15, markeredgecolor="y")

# for kk in range(0, len(sd_1)):
#     val = kk / len(sd_1)
#     plt.plot(sd_1[kk], ad_1[kk], "^", color=plt.cm.inferno(val), markersize=15)

# for kk in range(0, len(sd_2)):
#     val = kk / len(sd_2)
#     plt.plot(sd_2[kk], ad_2[kk], "^", color=plt.cm.viridis(val), markersize=15)


# for kk in range(0, len(sd_3)):
#     val = kk / len(sd_3)
#     plt.plot(sd_3[kk], ad_3[kk], "^", color=plt.cm.YlGn(val), markersize=15)


# plt.plot([0, 1], [accuracy_raw, accuracy_raw], "k--", markersize=10)
# plt.plot([0, 1], [1.0 / 20.0, 1.0 / 20.0], "k--", markersize=10)
# plt.xlabel("spearman")
# plt.ylabel("accuracy")
# plt.savefig("res3.png")

# plt.figure()
# ix = 0
# for depth_num in [1, 2, 3, 4, 5]:
#     for sensor_num in [1, 2, 3, 4, 5]:
#         for fix_whole_bottom in [True, False]:
#             _, _, _, _, height, _, _ = mg.get_rectangle_domain_constants(depth_num, sensor_num)
#             save_info = []
#             if sensor_num == 1 and fix_whole_bottom is False:
#                 continue
#             ac = accuracy_rect[ix]
#             if fix_whole_bottom:
#                 plt.plot(height, ac, "o", color=plt.cm.coolwarm(sensor_num / 5), markersize=depth_num * 3)
#             else:
#                 plt.plot(height, ac, "s", color=plt.cm.coolwarm(sensor_num / 5), markersize=depth_num * 3)
#             ix += 1

# for kk in range(0, len(sl_5)):
#     val = kk / len(sl_5)
#     plt.plot(dl_5[kk], al_5[kk], "*", color=plt.cm.spring(val), markersize=15, markeredgecolor="y")

# for kk in range(0, len(sl_4)):
#     val = kk / len(sl_4)
#     plt.plot(dl_4[kk], al_4[kk], "*", color=plt.cm.autumn(val), markersize=15, markeredgecolor="y")

# for kk in range(0, len(sd_1)):
#     val = kk / len(sd_1)
#     plt.plot(dd_1[kk], ad_1[kk], "^", color=plt.cm.inferno(val), markersize=15, markeredgecolor="y")


# for kk in range(0, len(sd_2)):
#     val = kk / len(sd_2)
#     plt.plot(dd_2[kk], ad_2[kk], "^", color=plt.cm.viridis(val), markersize=15)


# for kk in range(0, len(sd_3)):
#     val = kk / len(sd_3)
#     plt.plot(dd_3[kk], ad_3[kk], "^", color=plt.cm.YlGn(val), markersize=15)


# plt.plot([0, 20], [accuracy_raw, accuracy_raw], "k--", markersize=10)
# plt.plot([0, 20], [1.0 / 20.0, 1.0 / 20.0], "k--", markersize=10)

# plt.xlabel("depth")
# plt.ylabel("accuracy")
# plt.savefig("res2.png")