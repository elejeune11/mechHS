import numpy as np
import matplotlib.pyplot as plt 
import pygmsh
import meshio
import sys
import meshio

r = 1.3 # Radius of Hole
s = 3.0 # Sides of unit square
x_number = 5 # Number of units in x direction
y_number = 5 # Number of units in y direction

square = []
hole = []

space = s/2 # space in between circles

with pygmsh.occ.Geometry() as geom:
    geom.characteristic_length_max = r/10.0
    geom.characteristic_length_min = r/20.0
    
    square = geom.add_rectangle([0.0,0.0,0.0],s*x_number+2*space,s*y_number +2*space)

    for ii in range(0,x_number):
        for jj in range(0,y_number):
            hole.append(geom.add_disk([(ii*s)+(s/2)+space,(jj*s)+(s/2)+space],r))
            
    #-----------------------------Side half-circles-----------------------------------#
    
    for jj in range(0,y_number):
        hole.append(geom.add_disk([0,(jj*s)+(s/2)+space],r))
        hole.append(geom.add_disk([(x_number*s)+(s/2)+space,(jj*s)+(s/2)+space],r))

            
    geom.boolean_difference(square,hole)
            
        
       
            
    
    mesh = geom.generate_mesh()

mesh.points = mesh.points[:, :2] #prune z = 0 for 2D mesh

for cell in mesh.cells:
    if cell.type == "triangle":
        triangle_cells = cell.data
    
meshio.write("FEA_mesh/mesh3.xml", meshio.Mesh(points=mesh.points,cells={"triangle": triangle_cells}))

print(np.max(mesh.points[:]))
