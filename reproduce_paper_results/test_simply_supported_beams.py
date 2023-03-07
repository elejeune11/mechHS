import simply_supported_beams as ss
import applied_loads as al
import numpy as np


#----------------------Define functions to use in test-------------------------------------------------#

def simply_support(load_x,load,m,L):
    '''Function to calculate the reaction forces when l_a = 0 using the equations in Appendix A1 of the paper'''
    w = np.trapz(load,load_x)
    w_x = np.trapz(load*load_x,load_x)

    Ay = w-(1/(m*L))*w_x
    By = (1/(m*L))* w_x

    return Ay, By

def composite_beam(load_x,load,m,L):
    mL = m*L
    sec_AB = load_x <= mL
    sec_BC = load_x > mL

    midpoint = np.array([mL]) #np.array([load_x[sec_AB][-1]+((load_x[sec_BC][0]-load_x[sec_AB][-1])/2)])
    load_midpoint = np.interp(midpoint,load_x,load)

    load_x_AB = np.concatenate([load_x[sec_AB], midpoint], axis = 0)
    load_x_BC = np.concatenate([midpoint, load_x[sec_BC]], axis = 0)

    load_AB = np.concatenate([load[sec_AB], load_midpoint], axis = 0)
    load_BC = np.concatenate([load_midpoint, load[sec_BC]], axis = 0)

    load_xx = load*load_x
    load_xx_midpoint = np.interp(midpoint,load_x, load_xx)
    load_xx_AB = np.concatenate([load_xx[sec_AB], load_xx_midpoint], axis = 0)
    load_xx_BC = np.concatenate([load_xx_midpoint, load_xx[sec_BC]], axis = 0)
    


    w_total = np.trapz(load,load_x)
    w_ab = np.trapz(load_AB,load_x_AB)
    w_bc = np.trapz(load_BC,load_x_BC)


    w_x_total = np.trapz(load_xx,load_x)
    w_x_ab = np.trapz(load_xx_AB,load_x_AB)
    w_x_bc = np.trapz(load_xx_BC,load_x_BC)

    Ay = (mL*w_ab-w_x_ab)/mL
    Cy = (-mL*w_bc+w_x_bc)/mL
    By = w_total-Ay-Cy

    return Ay, By, Cy
    


def compare_solutions(func_soln, ana_soln, tol):
    '''Compare solutions to make sure they are within tolerance'''
    diff = np.abs((func_soln-ana_soln)/func_soln)
    return np.all(diff < tol)

def test_simply_supported_beam():
    '''test function for simply supported beams'''
    # Beam Geometry
    L = 10
    m = 0.5
    load_x = np.linspace(0,L,load.shape[1])

    for ii in range(0,len(load)):

        simple_code = []
        simple_analytical = []

        simple_code += ss.mechHS_simply_supported(load_x,load[ii,:],a = 0, b=m*L)

        simple_analytical += simply_support(load_x,load[ii,:], m,L)
        
    simple_code = np.array(simple_code)
    simple_analytical = np.array(simple_analytical)

    return compare_solutions(simple_code, simple_analytical, 1e-5)

def test_composite_beam():
    '''test function for composite beams'''

    # Beam Geometry
    L = 10.0
    m = 0.25

    mL = m*L
    load_x = np.linspace(0,L,load.shape[1])

    composite_code = ss.run_simply_supported_composite(load,0.0,L,[0.0,mL,2*mL])
    composite_analytical = []
    for ii in range(0,len(load)):
        composite_analytical.append(composite_beam(load_x,load[ii,:],m,L))

    composite_analytical = np.array(composite_analytical)

    return compare_solutions(composite_code, composite_analytical, 1e-1)

#--------------Test functions---------------------------#
if __name__ == "__main__":
    # get applied load (only use 1 of each type):
    num_pts = 10000

    all_loads = al.compute_all_loads(num_pts)
    all_loads = np.array(all_loads)
    all_loads_first = []

    for ii in range(0,10):
        all_loads_first.append(all_loads[0][ii,:])
    
    load = np.array(all_loads_first)

    #call test simply supported beam
    test_ssb = test_simply_supported_beam()
    print("Test simply supported beam:", test_ssb)


    # call test composite beam
    test_comp = test_composite_beam()
    print("Test composite beam:",test_comp)
