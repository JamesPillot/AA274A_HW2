import numpy as np
import math
import scikits.bvp_solver
import matplotlib.pyplot as plt
from utils import *

dt = 0.005

def ode_fun(tau, z):
    """
    This function computes the dz given tau and z. It is used in the bvp solver.
    Inputs:
        tau: the independent variable. This must be the first argument.
        z: the state vector. The first three states are [x, y, th, ...]
    Output:
        dz: the state derivative vector. Returns a numpy array.
    """
    ########## Code starts here ##########
    
    # pull scalars out of z for convenience
    x, y, th, p1, p2, p3, r = z
    
    Vel = (-1*p1*np.cos(th) - p2*np.sin(th))/2
    w = (-1*p3)/2
    # build components of dz
    dz_1 = Vel*np.cos(th)*r
    dz_2 = Vel*np.sin(th)*r
    dz_3 = w*r
    dz_4 = 0
    dz_5 = 0
    dz_6 = (p1*Vel*np.sin(th) - p2*Vel*np.cos(th))*r
    dz_7 = 0

    dz = np.array([dz_1, dz_2, dz_3, dz_4, dz_5, dz_6, dz_7])
    ########## Code ends here ##########
    return dz


def bc_fun(za, zb):
    """
    This function computes boundary conditions. It is used in the bvp solver.
    Inputs:
        za: the state vector at the initial time
        zb: the state vector at the final time
    Output:
        bca: tuple of boundary conditions at initial time
        bcb: tuple of boundary conditions at final time
    """
    # final goal pose
    xf = 5
    yf = 5
    thf = -np.pi/2.0
    xf = [xf, yf, thf]
    # initial pose
    x0 = [0, 0, -np.pi/2.0]

    ########## Code starts here ##########
    xb, yb, thb, p1b, p2b, p3b, rb = zb

    # left boundary conditions
    bca = np.array([x0[0] - za[0], x0[1] - za[1], x0[2] - za[2]])

    # condition for tf
    L = .325 #for lambda
    Vb = (-1*p1b*np.cos(thb) - p2b*np.sin(thb))/2
    omb = (-1*p3b)/2

    H_f = L + Vb**2 + omb**2 + p1b*Vb*np.cos(thb) + p2b*Vb*np.sin(thb) + p3b*omb

    # right boundary conditions
    bcb = np.array([xf[0] - zb[0], xf[1] - zb[1], xf[2] - zb[2], H_f])   
    ########## Code ends here ##########
    return (bca, bcb)

def solve_bvp(problem_inputs, initial_guess):
    """
    This function solves the bvp_problem.
    Inputs:
        problem_inputs: a dictionary of the arguments needs to define the problem
                        num_ODE, num_parameters, num_left_boundary_conditions,
                        boundary_points, function, boundary_conditions
        initial_guess: initial guess of the solution
    Output:
        z: a numpy array of the solution. It is of size [time, state_dim]

    Read this documentation -- https://pythonhosted.org/scikits.bvp_solver/tutorial.html
    """
    problem = scikits.bvp_solver.ProblemDefinition(**problem_inputs)
    soln = scikits.bvp_solver.solve(problem, solution_guess=initial_guess)

    # Test if time is reversed in bvp_solver solution
    flip, tf = check_flip(soln(0))
    t = np.arange(0,tf,dt)
    z = soln(t/tf)
    if flip:
        z[3:7,:] = -z[3:7,:]
    z = z.T # solution arranged so that it is [time, state_dim]
    return z

def compute_controls(z):
    """
    This function computes the controls V, om, given the state z. It is used in main().
    Input:
        z: z is the state vector for multiple time instances. It has size [time, state_dim]
    Outputs:
        V: velocity control input
        om: angular rate control input
    """
    ########## Code starts here ##########
    # x, y, th, p1, p2, p3, r = z
    x = z[:,0]
    y = z[:,1]
    th = z[:,2]
    p1 = z[:,3]
    p2 = z[:,4]
    p3 = z[:,5]
    r = z[:,6]

    V = (-1*p1*np.cos(th) - p2*np.sin(th))/2
    om = (-1*p3)/2
    ########## Code ends here ##########

    return V, om

def main():
    """
    This function solves the specified bvp problem and returns the corresponding optimal contol sequence
    Outputs:
        V: optimal V control sequence 
        om: optimal om ccontrol sequence
    You are required to define the problem inputs, initial guess, and compute the controls

    Hint: The total time is between 15-25
    """
    ########## Code starts here ##########
    num_ODE = 7
    num_parameters = 0
    num_left_boundary_conditions = 3
    boundary_points = (0,1)
    function = ode_fun
    boundary_conditions = bc_fun
    initial_guess =  np.array([2.5, 2.5, -np.pi/2.0, -2.0, -2.0, 0.5, 20])
    ########## Code ends here ##########

    problem_inputs = {
                      'num_ODE' : num_ODE,
                      'num_parameters' : num_parameters,
                      'num_left_boundary_conditions' : num_left_boundary_conditions,
                      'boundary_points' : boundary_points,
                      'function' : function,
                      'boundary_conditions' : boundary_conditions
                     }

    z = solve_bvp(problem_inputs, initial_guess)
    V, om = compute_controls(z)
    return z, V, om

if __name__ == '__main__':
    z, V, om = main()
    tf = z[0,-1]
    t = np.arange(0,tf,dt)
    x = z[:,0]
    y = z[:,1]
    th = z[:,2]
    data = {'z': z, 'V': V, 'om': om}
    save_dict(data, 'data/optimal_control.pkl')
    maybe_makedirs('plots')

    # plotting
    # plt.rc('font', weight='bold', size=16)
    plt.figure(figsize=(12, 4))
    plt.subplot(1, 2, 1)
    plt.plot(x, y,'k-',linewidth=2)
    plt.quiver(x[1:-1:200], y[1:-1:200],np.cos(th[1:-1:200]),np.sin(th[1:-1:200]))
    plt.grid(True)
    plt.plot(0,0,'go',markerfacecolor='green',markersize=15)
    plt.plot(5,5,'ro',markerfacecolor='red', markersize=15)
    plt.xlabel('X [m]')
    plt.ylabel('Y [m]')
    plt.axis([-1, 6, -1, 6])
    plt.title('Optimal Control Trajectory')

    plt.subplot(1, 2, 2)
    plt.plot(t, V,linewidth=2)
    plt.plot(t, om,linewidth=2)
    plt.grid(True)
    plt.xlabel('Time [s]')
    plt.legend(['V [m/s]', '$\omega$ [rad/s]'], loc='best')
    plt.title('Optimal control sequence')
    plt.tight_layout()
    plt.savefig('plots/optimal_control.png')
    plt.show()
