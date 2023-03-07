import fea_simulation as fea
import numpy as np
from pathlib import Path
import problem_setup_fcns as fcns
import applied_loads as al
import sys
from dolfin import *
import matplotlib.pyplot as plt

units = 5
r = 1.3 # Radius of Hole
s = 3.0 # Sides of unit square
x_number = 5 # Number of units in x direction
y_number = 5 # Number of units in y direction
space = s/2

width = 18.0

# mesh = Mesh('FEA_mesh/mesh3.xml')

# plot(mesh)
# plt.show()

path = Path(__file__).resolve().parent # folder for results
fname_mesh = 'mesh3'
fname_save = 'results'
save_pvd = False
xx_sensor_bc_list = [[0.0,0.5],[(width-0.5)/2,width/2],[width-0.5,width]] # range of x
yy_sensor_bc_list = [[0.0,0.5],[0.0,0.5],[0,0.5]] # range of y
height = width
fix_whole_btm = True
applied_load_num = 1
step_size = 0.01
num_steps = 10

# Compute loads
num_pts = 1000
all_loads = al.compute_all_loads(num_pts)

all_loads = np.array(all_loads)

all_loads_first = []

for ii in range(0,10):
    all_loads_first.append(all_loads[0][ii,:])

all_loads_first = np.array(all_loads_first)

#-------Get sensor values from FEM-------------#
sensor_values = []
for ii in range(1,11):
    applied_load_type = ii
    rxn_force = fea.run_simulation(path,fname_mesh,fname_save,save_pvd,
                                xx_sensor_bc_list,yy_sensor_bc_list,height,
                                fix_whole_btm,applied_load_type,applied_load_num,
                                step_size,num_steps)
    

    sensor_values.append(rxn_force[-1,:])


sensor_values = np.array(sensor_values)


print(sensor_values.shape)
print(all_loads_first.shape)
spearman, dist_og, dist_hash = fcns.compute_correlation(all_loads_first, sensor_values)

print(spearman)