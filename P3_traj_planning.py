import numpy as np
from P1_astar import DetOccupancyGrid2D, AStar
from P2_rrt import *
import scipy.interpolate
import matplotlib.pyplot as plt
from HW1.P1_differential_flatness import *
from HW1.P2_pose_stabilization import *
from HW1.P3_trajectory_tracking import *

class SwitchingController(object):
    """
    Uses one controller to initially track a trajectory, then switches to a 
    second controller to regulate to the final goal.
    """
    def __init__(self, traj_controller, pose_controller, t_before_switch):
        self.traj_controller = traj_controller
        self.pose_controller = pose_controller
        self.t_before_switch = t_before_switch

    def compute_control(self, x, y, th, t):
        """
        Inputs:
            (x,y,th): Current state 
            t: Current time

        Outputs:
            V, om: Control actions
        """
        ########## Code starts here ##########
        tf = self.traj_controller.traj_times[-1]

        pose_control_V, pose_control_om = self.pose_controller.compute_control(x,y,th,t)
        traj_control_V, traj_control_om = self.traj_controller.compute_control(x,y,th,t)
        if t > tf - self.t_before_switch:
            return pose_control_V, pose_control_om
        else:
            return traj_control_V, traj_control_om
              
        ########## Code ends here ##########

def compute_smoothed_traj(path, V_des, alpha, dt):
    """
    Fit cubic spline to a path and generate a resulting trajectory for our
    wheeled robot.

    Inputs:
        path (np.array [N,2]): Initial path
        V_des (float): Desired nominal velocity, used as a heuristic to assign nominal
            times to points in the initial path
        alpha (float): Smoothing parameter (see documentation for
            scipy.interpolate.splrep)
        dt (float): Timestep used in final smooth trajectory
    Outputs:
        traj_smoothed (np.array [N,7]): Smoothed trajectory
        t_smoothed (np.array [N]): Associated trajectory times
    Hint: Use splrep and splev from scipy.interpolate
    """
    ########## Code starts here ##########
    # Turn list of tuples into array
    path_array = np.array(path)
    x_traj = path_array[:,0]
    y_traj = path_array[:,1]
    # Caculate t smooth with path and V
    N = np.shape(path)[0]
    t_old = np.zeros(N,)
    t_old[0] = 0 #start at t = 0
    for i in range(1,N):
        j  = i-1
        t_old[i] = (np.linalg.norm(path_array[i,:] - path_array[j,:]) / V_des) + t_old[i-1]

    t_smoothed = np.arange(t_old[0], t_old[(N-1)], dt)

    # Grab appropriate x and y co-efficients for splev
    x_traj_tck = scipy.interpolate.splrep(t_old, x_traj, s = alpha)
    y_traj_tck = scipy.interpolate.splrep(t_old, y_traj, s = alpha)

    # For x and y trajectories
    x_traj_smoothed = scipy.interpolate.splev(t_smoothed, x_traj_tck)
    y_traj_smoothed = scipy.interpolate.splev(t_smoothed, y_traj_tck) 

    # For xdot and ydot trajectories
    xdot_traj_smoothed = scipy.interpolate.splev(t_smoothed, x_traj_tck, 1)
    ydot_traj_smoothed = scipy.interpolate.splev(t_smoothed, y_traj_tck, 1) 

    # For xddot and yddot trajectories
    xddot_traj_smoothed = scipy.interpolate.splev(t_smoothed, x_traj_tck, 2)
    yddot_traj_smoothed = scipy.interpolate.splev(t_smoothed, y_traj_tck, 2) 

    # For theta trajectory
    theta_traj_smoothed = np.arctan2(ydot_traj_smoothed, xdot_traj_smoothed)

    traj_smoothed = np.column_stack([x_traj_smoothed, y_traj_smoothed, theta_traj_smoothed, xdot_traj_smoothed, ydot_traj_smoothed, xddot_traj_smoothed, yddot_traj_smoothed])
    ########## Code ends here ##########

    return traj_smoothed, t_smoothed

def modify_traj_with_limits(traj, t, V_max, om_max, dt):
    """
    Modifies an existing trajectory to satisfy control limits and
    interpolates for desired timestep.

    Inputs:
        traj (np.array [N,7]): original trajecotry
        t (np.array [N]): original trajectory times
        V_max, om_max (float): control limits
        dt (float): desired timestep
    Outputs:
        t_new (np.array [N_new]) new timepoints spaced dt apart
        V_scaled (np.array [N_new])
        om_scaled (np.array [N_new])
        traj_scaled (np.array [N_new, 7]) new rescaled traj at these timepoints
    Hint: This should almost entirely consist of calling functions from Problem Set 1
    """
    ########## Code starts here ##########
    # Re-parameterize
    V,om = compute_controls(traj=traj)
    s = compute_arc_length(V, t)
    V_tilde = rescale_V(V, om, V_max, om_max)
    tau = compute_tau(V_tilde, s)
    om_tilde = rescale_om(V, om, V_tilde)
    
    # Get state final for interpolate_traj
    N = np.shape(traj)[0]
    xf = traj[N-1,0]
    yf = traj[N-1,1]
    thf = traj[N-1,2]
    s_f = State(x=xf, y=yf, V=V_max, th=thf)

    # Interpolate Trajectory
    t_new, V_scaled, om_scaled, traj_scaled = interpolate_traj(traj, tau, V_tilde, om_tilde, dt, s_f)

    ########## Code ends here ##########

    return t_new, V_scaled, om_scaled, traj_scaled
