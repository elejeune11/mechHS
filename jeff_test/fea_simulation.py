import applied_loads as al
from dolfin import *
from mshr import *
import numpy as np
from pathlib import Path
import problem_setup_fcns as fcns
from typing import List, Union


def compute_rxn_F(xx_sensor_bc_list: List, yy_sensor_bc_list: List, W: object, f_ext: object, f_int: object, force_list: List, dim_scale: float = 1.0):
    """Given information from the FEA problem at a given step. Will compute and return reaction forces."""
    x_dofs = W.sub(0).dofmap().dofs()
    y_dofs = W.sub(1).dofmap().dofs()
    f_ext_known = assemble(f_ext)
    f_ext_unknown = assemble(f_int) - f_ext_known
    dof_coords = W.tabulate_dof_coordinates().reshape((-1, 2))
    xx_idx_all = []
    yy_idx_all = []
    TOL = 10E-5
    for kk in range(0, len(xx_sensor_bc_list)):
        x_dof_list = []
        y_dof_list = []
        xmin = xx_sensor_bc_list[kk][0] - TOL
        xmax = xx_sensor_bc_list[kk][1] + TOL
        ymin = yy_sensor_bc_list[kk][0] - TOL
        ymax = yy_sensor_bc_list[kk][1] + TOL
        for jj in x_dofs:
            x = dof_coords[jj, 0]
            y = dof_coords[jj, 1]
            if x > xmin and x < xmax and y > ymin and y < ymax:
                x_dof_list.append(jj)
        for jj in y_dofs:
            x = dof_coords[jj, 0]
            y = dof_coords[jj, 1]
            if x > xmin and x < xmax and y > ymin and y < ymax:
                y_dof_list.append(jj)
        xx_idx_all.append(x_dof_list)
        yy_idx_all.append(y_dof_list)
    xx_sum_all = []
    yy_sum_all = []
    for kk in range(0, len(xx_idx_all)):
        val = np.sum(f_ext_unknown[xx_idx_all[kk]])
        xx_sum_all.append(val)
    for kk in range(0, len(yy_idx_all)):
        val = np.sum(f_ext_unknown[yy_idx_all[kk]])
        yy_sum_all.append(val)
    sum_list = xx_sum_all + yy_sum_all
    force_list.append(sum_list)
    return force_list


def run_simulation(
    path: Path,
    fname_mesh: str,
    fname_save: str,
    save_pvd: bool,
    xx_sensor_bc_list: List,
    yy_sensor_bc_list: List,
    height: float,
    fix_whole_btm: bool,
    applied_load_type: int,
    applied_load_num: int,
    step_size: float,
    num_steps: int
) -> np.ndarray:
    """Given FEA simulation inputs. Will run FEA simulations, save deformation and reaction forces.
    # input parameter definitions
    # path --> folder path needed to access data
    # fname_mesh --> file name of the mesh to use
    # fname_save --> file name to use for saving
    # save_pvd --> set true if you want to save a Paraview file (note these files may be large)
    # xx_sensor_bc --> locations where xx force will be recorded (not used)
    # yy_sensor_bc --> locations where yy force will be recorded (not used)
    # height --> height of the domain (can vary, bottom is always set at y = 0)
    # fix_whole_btm --> set True if the whole bottom boundary should be fixed. If False it will only be fixed at the sensors
    # applied_load_type --> will call the applied load function -- range 1-20
    # applied_load_num --> will call the applied load function -- range 1-20
    # step_size --> FEA step size (multiplier on applied load)
    # num_steps --> number of FEA steps
    """
    #####################################################################################
    # set up external loading information and FEA constants / parameters
    #####################################################################################
    is_linear = False
    TOL = 1.0e-5
    save_folder = fcns.create_folder(path, "FEA_results")
    # get additional constants we need for the applied load
    L_min, L_max, noise_scale_factor = al.get_constants()
    # get noise and kde objects to use as needed in the applied loading
    noise_dict, kde_dict = al.get_dicts()
    #####################################################################################
    # import mesh
    #####################################################################################
    mesh_folder = fcns.create_folder(path, "FEA_mesh")
    mesh_fname = str(mesh_folder) + "/" + fname_mesh + ".xml"
    mesh = Mesh(mesh_fname)
    #####################################################################################
    # compliler settings / optimization options
    #####################################################################################
    parameters["form_compiler"]["cpp_optimize"] = True
    parameters["form_compiler"]["representation"] = "uflacs"
    if is_linear is True:
        parameters["form_compiler"]["quadrature_degree"] = 1
    else:
        parameters["form_compiler"]["quadrature_degree"] = 2
    #####################################################################################
    # define function space
    #####################################################################################
    if is_linear:  # linear elements
        P2 = VectorElement("Lagrange", mesh.ufl_cell(), 1)
        TH = P2
        W = FunctionSpace(mesh, TH)
    else:  # quadratic elements
        P2 = VectorElement("Lagrange", mesh.ufl_cell(), 2)
        TH = P2
        W = FunctionSpace(mesh, TH)
    V = FunctionSpace(mesh, 'CG', 1)
    #####################################################################################
    # define material parameters
    #####################################################################################
    E, nu = 10.0, 0.3
    mu, lmbda = Constant(E / (2 * (1 + nu))), Constant(E * nu / ((1 + nu) * (1 - 2 * nu)))
    matdomain = MeshFunction("size_t", mesh, mesh.topology().dim())
    dx = Measure("dx", domain=mesh, subdomain_data=matdomain)
    #####################################################################################
    # define boundary domains
    #####################################################################################
    btm_bc_domain_list = []
    for kk in range(0, len(xx_sensor_bc_list)):
        xmin = xx_sensor_bc_list[kk][0] - TOL
        xmax = xx_sensor_bc_list[kk][1] + TOL
        ymin = yy_sensor_bc_list[kk][0] - TOL
        ymax = yy_sensor_bc_list[kk][1] + TOL
        bc = CompiledSubDomain("x[1] >= ymin && x[1] <= ymax && x[0] >= xmin && x[0] <= xmax", ymin=ymin, ymax=ymax, xmin=xmin, xmax=xmax)
        btm_bc_domain_list.append(bc)
    bcTop_surf = CompiledSubDomain("near(x[1], ytop, TOL)", ytop=height, TOL=TOL)
    #####################################################################################
    # define Dirichlet boundary conditions
    #####################################################################################
    bcs = []
    if fix_whole_btm:
        yval = 0
        bc_dom = CompiledSubDomain("near(x[1], yval, TOL)", yval=yval, TOL=TOL)
        bc_x = DirichletBC(W.sub(0), Constant((0.0)), bc_dom, method="pointwise")
        bc_y = DirichletBC(W.sub(1), Constant((0.0)), bc_dom, method="pointwise")
        bcs.append(bc_x)
        bcs.append(bc_y)
    else:
        for kk in range(0, len(btm_bc_domain_list)):
            bc_dom = btm_bc_domain_list[kk]
            bc_x = DirichletBC(W.sub(0), Constant((0.0)), bc_dom, method="pointwise")
            bc_y = DirichletBC(W.sub(1), Constant((0.0)), bc_dom, method="pointwise")
            bcs.append(bc_x)
            bcs.append(bc_y)
    #####################################################################################
    # set up for traction boundary conditions
    #####################################################################################
    boundary_markers = MeshFunction("size_t", mesh, mesh.topology().dim() - 1)
    boundary_markers.set_all(0)
    bcTop_surf.mark(boundary_markers, 1)  # Prescribed traction
    ds = Measure("ds", domain=mesh, subdomain_data=boundary_markers)
    #####################################################################################
    # define finite element problem
    #####################################################################################
    u = Function(W)
    du = TrialFunction(W)
    v = TestFunction(W)
    #####################################################################################
    # define body force boundary conditions
    #####################################################################################
    B = Constant((0.0, 0.0))
    ######################################################################################
    # loop through solution steps
    ######################################################################################
    if save_pvd:
        fname_paraview = File(str(save_folder) + "/" + fname_save + ".pvd")
    force_list = []
    for jj in range(0, num_steps):
        class AppliedLoad(UserExpression):
            def eval(self, value, x):
                value[0] = 0
                val = al.load_at_a_point(x[0], L_min, L_max, applied_load_type, applied_load_num, noise_scale_factor, noise_dict, kde_dict)
                value[1] = val * jj * step_size

            def value_shape(self):
                return (2, )
        ##################################################################################
        ##################################################################################
        T = AppliedLoad(element=u.ufl_element())
        # Kinematics
        d = len(u)
        I = variable(Identity(d))  # Identity tensor
        F = variable(I + grad(u))  # Deformation gradient
        C = variable(F.T * F)  # Right Cauchy-Green tensor
        # Invariants of deformation tensors
        Ii   = tr(C)
        Iii  = 1/2 * (tr(C) - tr(dot(C, C)))
        Iiii = det(C)
        J    = det(F)
        # Strain energy function 
        psi = (mu / 2) * (Ii - 3) - mu * ln(J) + (lmbda / 2) * (ln(J)) ** 2
        ######################################################################################
        # set up eqn to solve and solve it
        ######################################################################################
        f_int = derivative(psi * dx, u, v)
        f_ext = derivative(dot(B, u) * dx + dot(T, u) * ds(1), u, v)
        F = f_int - f_ext
        # Tangent 
        dF = derivative(F, u, du)
        solve(F == 0, u, bcs, J=dF)
        ######################################################################################
        # visualize output and post processing
        ######################################################################################
        if save_pvd:
            fname_paraview << (u, jj)
        force_list = compute_rxn_F(xx_sensor_bc_list, yy_sensor_bc_list, W, f_ext, f_int, force_list)
        force_list_np = np.asarray(force_list)
        np.savetxt(str(save_folder) + "/" + fname_save + "_force_list.txt", force_list_np)
    return force_list_np
