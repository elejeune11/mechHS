import fea_simulation as fea
import mesh_generation as mg
import numpy as np
from pathlib import Path
import problem_setup_fcns as fcns
import sys


def get_inputs_for_qsub_script():
    qsub_inputs = ""
    qsub_inputs_list = []
    counter = 0
    for applied_load_type in range(1, 21):
        for device_num in [1, 2, 3]:
            str_qsub = "'True %i %i' " % (applied_load_type, device_num)
            qsub_inputs += str_qsub
            counter += 1
            qsub_inputs_list.append("True %i %i" % (applied_load_type, device_num))
    return qsub_inputs, qsub_inputs_list, counter


if __name__ == "__main__":
    # read command line arguments
    args = sys.argv
    # FEA or script to process results
    run_FEA = str(args[1]) == "True"
    if run_FEA:
        # command line inputs
        applied_load_type = int(args[2])
        device_num = int(args[3])
        # additional parameters
        fix_whole_btm = True
        # convert inputs to what we need to run the code and set inputs
        mypath = Path(__file__).resolve().parent
        mypath_res = Path(__file__).resolve().parent.joinpath("FEA_results").resolve()
        mesh_param = 200
        fname_device = "grid_25x25_device%i" % (device_num)
        fname_mesh = fname_device + "_mesh_%i" % (mesh_param)
        save_pvd = False
        step_size = 0.001
        num_steps = 11
        _, xx_sensor_bc_list, yy_sensor_bc_list, height = mg.read_design_grid(mypath, 25, 25, fname_device, 10, 10)
        for kk in range(1, 21):
            applied_load_num = kk
            fname_res = fname_device + "_mesh%i_alt%i_aln%i_force_list.txt" % (mesh_param, applied_load_type, applied_load_num)
            file_res = mypath_res.joinpath(fname_res).resolve()
            if file_res.is_file():
                continue
            fname_save = fname_device + "_mesh%i_alt%i_aln%i" % (mesh_param, applied_load_type, applied_load_num)
            try:
                fea.run_simulation(mypath, fname_mesh, fname_save, save_pvd, xx_sensor_bc_list, yy_sensor_bc_list, height, fix_whole_btm, applied_load_type, applied_load_num, step_size, num_steps)
            except RuntimeError:
                print("runtime error occured")
                print(RuntimeError)
    else:
        mypath = Path(__file__).resolve().parent
        mypath_fea = Path(__file__).resolve().parent.joinpath("FEA_results_custom_workstation").resolve()
        save_path = fcns.create_folder(mypath, "FEA_results_summarized")
        mesh_param = 200
        for device_num in [1, 2, 3]:
            fname_device = "grid_25x25_device%i" % (device_num)
            save_info = []
            for applied_load_num in range(1, 21):
                for applied_load_type in range(1, 21):
                    fix_whole_bottom = False
                    fname = fname_device + "_mesh%i_alt%i_aln%i_force_list.txt" % (mesh_param, applied_load_type, applied_load_num)
                    file = mypath_fea.joinpath(fname).resolve()
                    data = np.loadtxt(str(file))
                    if device_num == 1 or device_num == 2:
                        num_sensors = 3
                    elif device_num == 3:
                        num_sensors = 5
                    save_info.append(data[10, num_sensors:])  # end of simulation results, y force, x is ignored here.
            save_info = np.asarray(save_info)
            fname = fname_device + ".txt"
            fname = save_path.joinpath(fname).resolve()
            np.savetxt(fname, save_info)
