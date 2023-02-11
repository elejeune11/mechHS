from dolfin import *
import matplotlib.pyplot as plt
from mshr import *
import numpy as np
from pathlib import Path
import problem_setup_fcns as fcns


def get_rectangle_domain_constants(depth_num: int, sensor_num: int):
    Lmax = 10
    L = Lmax
    x0 = 0
    x1 = Lmax
    y0 = 0
    sw = 0.5
    if depth_num == 1:
        y1 = 1
    elif depth_num == 2:
        y1 = 2.5
    elif depth_num == 3:
        y1 = 5.0
    elif depth_num == 4:
        y1 = 10.0
    elif depth_num == 5:
        y1 = 20.0
    xx_sensor = []
    yy_sensor = []
    if sensor_num == 1:
        xx_sensor.append([L / 2.0 - sw / 2.0, L / 2.0 + sw / 2.0])
        yy_sensor.append([y0, y0])
    elif sensor_num == 2:
        xx_sensor.append([0, sw])
        yy_sensor.append([y0, y0])
        xx_sensor.append([L - sw, L])
        yy_sensor.append([y0, y0])
    elif sensor_num == 3:
        xx_sensor.append([0, sw])
        yy_sensor.append([y0, y0])
        xx_sensor.append([L / 2.0 - sw / 2.0, L / 2.0 + sw / 2.0])
        yy_sensor.append([y0, y0])
        xx_sensor.append([L - sw, L])
        yy_sensor.append([y0, y0])
    elif sensor_num == 4:
        xx_sensor.append([0, sw])
        yy_sensor.append([y0, y0])
        xx_sensor.append([L / 3.0 - sw / 2.0, L / 3.0 + sw / 2.0])
        yy_sensor.append([y0, y0])
        xx_sensor.append([2.0 * L / 3.0 - sw / 2.0, 2.0 * L / 3.0 + sw / 2.0])
        yy_sensor.append([y0, y0])
        xx_sensor.append([L - sw, L])
        yy_sensor.append([y0, y0])
    elif sensor_num == 5:
        xx_sensor.append([0, sw])
        yy_sensor.append([y0, y0])
        xx_sensor.append([L / 4.0 - sw / 2.0, L / 4.0 + sw / 2.0])
        yy_sensor.append([y0, y0])
        xx_sensor.append([2.0 * L / 4.0 - sw / 2.0, 2.0 * L / 4.0 + sw / 2.0])
        yy_sensor.append([y0, y0])
        xx_sensor.append([3.0 * L / 4.0 - sw / 2.0, 3.0 * L / 4.0 + sw / 2.0])
        yy_sensor.append([y0, y0])
        xx_sensor.append([L - sw, L])
        yy_sensor.append([y0, y0])
    height = y1 - y0
    return x0, x1, y0, y1, height, xx_sensor, yy_sensor


def create_mesh_rectangle(path: Path, fname: str, mesh_param: int, x0: float, y0: float, x1: float, y1: float) -> object:
    """Create mesh that is a simple rectangle."""
    new_path = fcns.create_folder(path, "FEA_mesh")
    r = Rectangle(Point(x0, y0), Point(x1, y1))
    mesh = generate_mesh(r, mesh_param)
    mesh_fname = str(new_path) + "/" + fname + "_mesh_%i.xml" % (mesh_param)
    mesh_file = File(mesh_fname)
    mesh_file << mesh
    plt.figure()
    plot(mesh)
    plt.savefig(str(new_path) + "/" + fname + "_mesh_%i.png" % (mesh_param))
    return mesh


def create_mesh_lattice(path: Path, fname: str, mesh_param: int, size: int):
    new_path = fcns.create_folder(path, "FEA_mesh")
    depth_num = 4
    _, _, _, _, _, coords, _ = get_rectangle_domain_constants(depth_num, size)
    rect_list = []
    # vertical rectangles
    for kk in range(0, len(coords)):
        x0 = coords[kk][0]
        y0 = 0
        x1 = coords[kk][1]
        y1 = 10
        r = Rectangle(Point(x0, y0), Point(x1, y1))
        rect_list.append(r)
    # horizontal rectangles
    for kk in range(0, len(coords)):
        x0 = 0
        y0 = coords[kk][0]
        x1 = 10
        y1 = coords[kk][1]
        r = Rectangle(Point(x0, y0), Point(x1, y1))
        rect_list.append(r)
    # geometry
    geom = rect_list[0]
    for kk in range(1, len(rect_list)):
        geom += rect_list[kk]
    # mesh
    mesh = generate_mesh(geom, mesh_param)
    mesh_fname = str(new_path) + "/" + fname + "_mesh_%i.xml" % (mesh_param)
    mesh_file = File(mesh_fname)
    mesh_file << mesh
    plt.figure()
    plot(mesh)
    plt.savefig(str(new_path) + "/" + fname + "_mesh_%i.png" % (mesh_param))
    return mesh


def create_blank_grid(path: Path, num_grid_row: int, num_grid_col: int, fname: str, space: int = 12) -> np.ndarray:
    """Given a grid size row x col and file name, "space" will control the size of the grid.
    Will create the blank template for illustrating the mechanical hash devices.
    Note that the bottom two rows of the template are for specifying boundary conditions."""
    new_path = fcns.create_folder(path, "FEA_mesh")
    mat_row = num_grid_row * space + 1 * space + 1
    mat_col = num_grid_col * space + 1
    mat = np.zeros((mat_row, mat_col))
    for kk in range(0, num_grid_row + 1):
        idx = kk * space
        mat[idx, :] = 1
    mat[-1, :] = 1
    for kk in range(0, num_grid_col):
        idx = kk * space
        mat[:, idx] = 1
    mat[:, -1] = 1
    mat[-1 * space - 1, :] = 0.25
    mat[-1, :] = 0.25
    for kk in range(0, num_grid_col):
        idx = kk * space
        mat[-1 * space - 1:, idx] = 0.25
    mat[-1 * space - 1:, -1] = 0.25
    plt.imsave(str(new_path) + "/" + fname + ".png", mat, cmap=plt.cm.binary)
    return mat


def read_design_grid(
    path: Path,
    num_grid_row: int,
    num_grid_col: int,
    fname: str,
    row_dim: float,
    col_dim: float
) -> np.ndarray:
    """Given the path and a file name for the designed grid.
    Will create text files that input geometry to the meshing code, and BCs to the FEA code."""
    new_path = fcns.create_folder(path, "FEA_mesh")
    img = plt.imread(str(new_path) + "/" + fname + ".png")
    img_row = img.shape[0]
    img_col = img.shape[1]
    num_row = num_grid_row + 1
    num_col = num_grid_col
    grid_space_row = int(img_row / num_row)
    grid_space_col = int(img_col / num_col)
    geom_mat = np.zeros((num_grid_row, num_grid_col))
    for kk in range(0, num_grid_row):
        for jj in range(0, num_grid_col):
            ix_row = int(kk * grid_space_row + grid_space_row / 2.0)
            ix_col = int(jj * grid_space_col + grid_space_col / 2.0)
            val = img[ix_row, ix_col, :]
            if np.allclose(val, np.asarray([1, 0, 0, 1])):
                geom_mat[kk, jj] = 1
    xx_sensor = []
    yy_sensor = []
    for kk in range(0, num_grid_col):
        ix_row = int((num_grid_row + 0.5) * grid_space_row)
        ix_col = int(kk * grid_space_col + grid_space_col / 2.0)
        val = img[ix_row, ix_col, :]
        if np.allclose(val, np.asarray([0, 0, 0, 1])):
            xx_sensor.append([col_dim * kk / num_grid_col, col_dim * (kk + 1) / num_grid_col])
            yy_sensor.append([0, 0])
    height = row_dim
    return geom_mat, xx_sensor, yy_sensor, height


def create_mesh_design_grid(path: Path, fname: str, mesh_param: int, geom_mat: np.ndarray, row_dim: float, col_dim: float) -> object:
    """Given input geometry information.
    If subdomains is True, the mesh will be structured to allow each "pixel" to have different material properties.
    Will create a mesh file (xml) that is compatible with FEniCS."""
    new_path = fcns.create_folder(path, "FEA_mesh")
    num_grid_row = geom_mat.shape[0]
    num_grid_col = geom_mat.shape[1]
    rect_list = []
    for kk in range(0, num_grid_row):
        for jj in range(0, num_grid_col):
            if geom_mat[kk, jj] > 0:
                kk_0 = (num_grid_row - kk) / num_grid_row * row_dim
                kk_1 = (num_grid_row - (kk + 1)) / num_grid_row * row_dim
                jj_0 = jj / num_grid_col * col_dim
                jj_1 = (jj + 1) / num_grid_col * col_dim
                r = Rectangle(Point(jj_0, kk_0), Point(jj_1, kk_1))
                rect_list.append(r)
    geom = rect_list[0]
    for kk in range(1, len(rect_list)):
        geom += rect_list[kk]
    mesh = generate_mesh(geom, mesh_param)
    mesh_fname = str(new_path) + "/" + fname + "_mesh_%i.xml" % (mesh_param)
    mesh_file = File(mesh_fname)
    mesh_file << mesh
    plt.figure()
    plot(mesh)
    plt.savefig(str(new_path) + "/" + fname + "_mesh_%i.png" % (mesh_param))
    return mesh