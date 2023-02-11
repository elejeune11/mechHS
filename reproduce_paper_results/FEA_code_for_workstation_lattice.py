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
        for lattice_size in [3, 4, 5]:
            str_qsub = "'True %i %i' " % (applied_load_type, lattice_size)
            qsub_inputs += str_qsub
            counter += 1
            qsub_inputs_list.append("True %i %i" % (applied_load_type, lattice_size))
    return qsub_inputs, qsub_inputs_list, counter


if __name__ == "__main__":
    # read command line arguments
    args = sys.argv
    # FEA or script to process results
    run_FEA = str(args[1]) == "True"
    if run_FEA:
        # applied load type
        applied_load_type = int(args[2])
        # size
        size = int(args[3])
        # additional parameters
        fix_whole_btm = True
        # sensor number
        sensor_num = size
        # convert inputs to what we need to run the code and set inputs
        mypath = Path(__file__).resolve().parent
        mypath_res = Path(__file__).resolve().parent.joinpath("FEA_results").resolve()
        mesh_param = 200
        fname_mesh = "lattice_size%i_mesh_%i" % (size, mesh_param)
        save_pvd = False
        step_size = 0.001
        num_steps = 11
        _, _, _, _, height, xx_sensor_bc_list, yy_sensor_bc_list = mg.get_rectangle_domain_constants(4, sensor_num)
        # the way our workstation code is setup we will run all load types for one device, one bottom fixity, and one number
        for kk in range(1, 21):
            applied_load_num = kk
            fname_res = "lattice_size%i_mesh%i_alt%i_aln%i_force_list.txt" % (size, mesh_param, applied_load_type, applied_load_num)
            file_res = mypath_res.joinpath(fname_res).resolve()
            if file_res.is_file():
                continue
            fname_save = "lattice_size%i_mesh%i_alt%i_aln%i" % (size, mesh_param, applied_load_type, applied_load_num)
            try:
                fea.run_simulation(mypath, fname_mesh, fname_save, save_pvd, xx_sensor_bc_list, yy_sensor_bc_list, height, fix_whole_btm, applied_load_type, applied_load_num, step_size, num_steps)
            except RuntimeError:
                print("runtime error occured")
                print(RuntimeError)
    else:
        mypath = Path(__file__).resolve().parent
        mypath_fea = Path(__file__).resolve().parent.joinpath("FEA_results_lattice_workstation").resolve()
        save_path = fcns.create_folder(mypath, "FEA_results_summarized")
        mesh_param = 200
        for size in [3, 4, 5]:
            save_info = []
            for applied_load_num in range(1, 21):
                for applied_load_type in range(1, 21):
                    fix_whole_bottom = False
                    fname = "lattice_size%i_mesh%i_alt%i_aln%i_force_list.txt" % (size, mesh_param, applied_load_type, applied_load_num)
                    file = mypath_fea.joinpath(fname).resolve()
                    data = np.loadtxt(str(file))
                    save_info.append(data[10, size:])  # end of simulation results, y force, x is ignored here.
            save_info = np.asarray(save_info)
            fname = "lattice_%i.txt" % (size)
            fname = save_path.joinpath(fname).resolve()
            np.savetxt(fname, save_info)
