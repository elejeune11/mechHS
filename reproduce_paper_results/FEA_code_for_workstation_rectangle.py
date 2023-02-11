import fea_simulation as fea
import mesh_generation as mg
import numpy as np
from pathlib import Path
import problem_setup_fcns as fcns
import sys


def get_inputs_for_qsub_script(fixed_btm=False):
    qsub_inputs = ""
    qsub_inputs_list = []
    counter = 0
    for applied_load_type in range(1, 21):
        for depth_num in [1, 2, 3, 4, 5]:
            for sensor_num in [2, 3, 4, 5]:
                if fixed_btm is False:
                    str_qsub = "'True %i %i %i " % (applied_load_type, depth_num, sensor_num) + "False' "
                else:
                    str_qsub = "'True %i %i %i " % (applied_load_type, depth_num, sensor_num) + "True' "
                qsub_inputs += str_qsub
                counter += 1
                if fixed_btm is False:
                    qsub_inputs_list.append("True %i %i %i " % (applied_load_type, depth_num, sensor_num) + "False")
                else:
                    qsub_inputs_list.append("True %i %i %i " % (applied_load_type, depth_num, sensor_num) + "True")
    return qsub_inputs, qsub_inputs_list, counter


if __name__ == "__main__":
    # read command line arguments
    args = sys.argv
    # FEA or script to process results
    run_FEA = str(args[1]) == "True"
    if run_FEA:
        # load number
        applied_load_type = int(args[2])
        # depth number
        depth_num = int(args[3])
        # sensor number
        sensor_num = int(args[4])
        # fix bottom
        fix_whole_btm = str(args[5]) == "True"
        # convert inputs to what we need to run the code and set inputs
        mypath = Path(__file__).resolve().parent
        mypath_res = Path(__file__).resolve().parent.joinpath("FEA_results").resolve()
        mesh_param = 200
        fname_mesh = "rectangle_depthnum%i_mesh_%i" % (depth_num, mesh_param)
        save_pvd = False
        step_size = 0.025
        num_steps = 11
        _, _, _, _, height, xx_sensor_bc_list, yy_sensor_bc_list = mg.get_rectangle_domain_constants(depth_num, sensor_num)
        # the way our workstation code is setup we will run all load types for one device, one bottom fixity, and one number
        for kk in range(1, 21):
            applied_load_num = kk
            fname_res = "rectangle_depthnum%i_mesh%i_sensornum%i_fixwholebtm%i_alt%i_aln%i_force_list.txt" % (depth_num, mesh_param, sensor_num, fix_whole_btm, applied_load_type, applied_load_num)
            file_res = mypath_res.joinpath(fname_res).resolve()
            if file_res.is_file():
                continue
            fname_save = "rectangle_depthnum%i_mesh%i_sensornum%i_fixwholebtm%i_alt%i_aln%i" % (depth_num, mesh_param, sensor_num, fix_whole_btm, applied_load_type, applied_load_num)
            try:
                fea.run_simulation(mypath, fname_mesh, fname_save, save_pvd, xx_sensor_bc_list, yy_sensor_bc_list, height, fix_whole_btm, applied_load_type, applied_load_num, step_size, num_steps)
            except RuntimeError:
                print("runtime error occured")
    else:
        # post-processing -- creating summary files for each device
        mypath = Path(__file__).resolve().parent
        mypath_fea = Path(__file__).resolve().parent.joinpath("FEA_results_rectangle_workstation").resolve()
        save_path = fcns.create_folder(mypath, "FEA_results_summarized")
        mesh_param = 200
        for fix_whole_bottom in [True, False]:
            for depth_num in [1, 2, 3, 4, 5]:
                for sensor_num in [2, 3, 4, 5]:
                    save_info = []
                    for applied_load_num in range(1, 21):
                        for applied_load_type in range(1, 21):
                            fname = "rectangle_depthnum%i_mesh%i_sensornum%i_fixwholebtm%i_alt%i_aln%i_force_list.txt" % (depth_num, mesh_param, sensor_num, fix_whole_bottom, applied_load_type, applied_load_num)
                            file = mypath_fea.joinpath(fname).resolve()
                            data = np.loadtxt(str(file))
                            save_info.append(data[10, sensor_num:])  # end of simulation results, y force, x is ignored here.
                    save_info = np.asarray(save_info)
                    fname = "depth_num%i_sensor_num%i_fix_whole_bottom%i.txt" % (depth_num, sensor_num, fix_whole_bottom)
                    fname = save_path.joinpath(fname).resolve()
                    np.savetxt(fname, save_info)
