"""
=============== The Dynamic Trajectory Fit to meteor observations ===============
Created on Mon Jun 29 15:13:05 2017
@author: Trent Jansen-Sturgeon

Least squares estimate.
Inputs: Requires azi/alt files with timing for triangulation.
Output: The triangulated data file in the form of time, LLH, ECEF.

"""

# Import modules
import multiprocessing
import argparse
import os
import logging
import time

# Import science modules
import numpy as np
from scipy.interpolate import interp1d
from numpy.linalg import inv, norm
from scipy.linalg import block_diag
from scipy.integrate import odeint
from scipy.optimize import least_squares, brentq
from astropy.table import Table, vstack
from astropy.time import Time
from astropy import units as u

import matplotlib.pyplot as plt
# plt.switch_backend('agg')

# Import custom modules
from trajectory_utilities import LLH2ECEF, ECEF2LLH, \
    EarthRadius, ECEF2ECI, ECEF2ECI_pos, enu_matrix, \
    ECI2ECEF, track_errors, calculate_timing_offsets, \
    altaz2radec_jac, track_errors_radec_jac, \
    angular_difference, gravity_vector, NRLMSISE_00
from trajectory_utilities import TriangulationError, \
    PoorTriangulationResiduals, TriangulationInvalidInput, TriangulationOutOfRange
from StraightLineLeastSquares import SLLS, \
    FirstApprox_using_tables, radiant2dict
from CSV2KML import Path, Points, Rays, merge_trajectory_KMLs

__author__ = "Trent Jansen-Sturgeon"
__copyright__ = "Copyright 2017, Desert Fireball Network"
__license__ = ""
__version__ = "1.4" # Handles timing offsets
__scriptName__ = "DynamicTrajectoryFit.py"


mu_e = 3.986005000e14 #4418e14 # Earth's standard gravitational parameter (m3/s2)
w_e = 7.2921158553e-5  # Earth's rotation rate (rad/s)


def EarthDynamics(X, t_rel, t0, atm_info=None):
    '''
    The state rate dynamics are used in Runge-Kutta integration method to 
    calculate the next set of equinoctial element values.
    ''' 

    # Determine the actual time
    t_jd = t0 + t_rel / (24*60*60)

    # State parameter vector decomposed, where beta is the ballistic coefficient
    Pos_ECI = X[:3]; Vel_ECI = X[3:6]; beta = X[6]; sigma = X[7]

    if atm_info: #rho_a(h) is a lookup table
        radius = norm(Pos_ECI)

        ''' Primary Gravitational Acceleration '''
        a_grav = np.array([interp1d(atm_info[0], atm_info[2], fill_value='extrapolate')(radius),
                           interp1d(atm_info[0], atm_info[3], fill_value='extrapolate')(radius),
                           interp1d(atm_info[0], atm_info[4], fill_value='extrapolate')(radius)])

        ''' Atmospheric Drag Perturbation - NRLMSISE-00 model '''
        rho_a = interp1d(atm_info[0], atm_info[1], fill_value='extrapolate')(radius)
    else:
        a_grav = gravity_vector(Pos_ECI).flatten()
        rho_a = NRLMSISE_00(np.vstack((Pos_ECI)), t_jd)[2]

    # Velocity relative to the atmosphere
    v_atm = np.cross(np.array([0, 0, w_e]), Pos_ECI)
    v_rel = Vel_ECI - v_atm; v = norm(v_rel)

    # Total drag perturbation
    a_drag = - rho_a  * v * v_rel / (2 * beta)
    
    ''' Total Perturbing Acceleration '''
    a_tot = a_grav + a_drag

    # Mass-loss equation
    dbeta_dt = - sigma * rho_a * v**3 / 6

    ''' State Rate Equation '''
    X_dot = np.hstack((Vel_ECI, a_tot, dbeta_dt, 0.0))

    return X_dot

def Propagate(X0, T_rel, t0, atm_info, X0_cov=None):
    X = odeint(EarthDynamics, X0, T_rel, args=(t0,atm_info,))

    if not isinstance(X0_cov, np.ndarray):
        return [X]
    else:
        # Calculate the Jacobian
        Jacobian = np.zeros((len(T_rel), len(X0), len(X0)))
        ARGS = [(i, j, X, T_rel, t0, atm_info) for j in range(len(T_rel)-1)
                                    for i in range(len(X0))]
        jac_column_pool = multiprocessing.Pool()
        result_arr = jac_column_pool.map(dx_dx0_col, ARGS)
        jac_column_pool.close()
        jac_column_pool.join()

        result_arr.reverse()
        for j in range(len(T_rel)-1):
            for i in range(len(X0)):
                Jacobian[j+1,:,i] = result_arr.pop()

        # Jacobian = np.zeros((len(T_rel), len(X0), len(X0)))
        # for j in range(len(T_rel)-1):
        #     for i in range(len(X0)):
        #         Jacobian[j+1,:,i] = dx_dx0_col((i, j, X, T_rel, t0, atm_info))

        # Determine the uncertainties from X0_cov estimate
        X_cov = np.zeros((len(T_rel), len(X0), len(X0))); X_cov[0] = X0_cov
        for j in range(len(T_rel)-1):
            X_cov[j+1] = Jacobian[j+1].dot(X_cov[j]).dot(Jacobian[j+1].T)

        return [X, X_cov]

import copy
def Propagate_frags(X0, T_rel, t0, atm_info, X0_cov=None):

    # Extract the fragment information
    [p_frag, t_frag] = np.hsplit(X0[8:], 2)

    # Save the original values
    X_old, X_cov_old = X0[:8], X0_cov
    t_old, T_rel = T_rel[0], T_rel[1:]

    # Propagate for every fragment
    X = [X_old.reshape((1,8))] 
    if isinstance(X0_cov, np.ndarray):
        X_cov = [X_cov_old.reshape((1,8,8))]
    for t, p in zip(np.append(t_frag, T_rel[-1] - 0.1), np.append(p_frag, 0)):

        # Time array of fragment
        T_frag = T_rel[(T_rel <= t_old) & (T_rel >= t)]
        T_frag = np.hstack((t_old, T_frag, t)); t_old = t.copy()

        # Propagate the fragment
        # if isinstance(X0_cov, np.ndarray): print(X_old, T_frag, X0_cov)
        argsort = np.argsort(T_frag)[::-1]; T_sorted = T_frag[argsort]
        result = Propagate(X_old, T_sorted, t0, atm_info, X_cov_old)

        # Save the result
        X_old = result[0][np.argsort(argsort)][-1]
        X.append(result[0][np.argsort(argsort)][1:-1])
        if isinstance(X0_cov, np.ndarray):
            X_cov_old = result[1][np.argsort(argsort)][-1]
            X_cov.append(result[1][np.argsort(argsort)][1:-1])

        # Fragment that fireball!!
        X_old[6] /= (1 - p)**(1./3)

    # Return the state over time
    if not isinstance(X0_cov, np.ndarray):
        return np.concatenate(X)
    else:
        return np.concatenate(X), np.concatenate(X_cov)

def dx_dx0_col(ARGS):
    ''' Evaluates one column of the Jacobian '''
    i, j, X, T_rel, t0, atm_info = ARGS

    dx = np.array([1e0,1e0,1e0,5e-1,5e-1,5e-1,1e-1,1e-11])
    e = np.zeros(len(dx)); e[i] = 1.0

    x_pos = odeint(EarthDynamics, X[j] + dx*e, T_rel[j:j+2], args=(t0,atm_info,))[1]
    x_neg = odeint(EarthDynamics, X[j] - dx*e, T_rel[j:j+2], args=(t0,atm_info,))[1]
    # return (x_pos - X[j+1]) / dx[i]
    return (x_pos - x_neg) / (2*dx[i])


def adjust_args(state, args):
    [ObsData, UV_obs, cov_obs, t0, atm_info, offset_obs, n_frag] = args
    noc = len(offset_obs) # Number of offset cameras

    if np.any(ObsData['timing']==0) or noc != 0: # Check for any without absolute timing

        # Put together the arguments for those without timing
        index_notime = np.where(ObsData['timing'] == 0)[0]
        ObsData['t_rel'][index_notime] = state[8 + 2*n_frag + noc:]
        
        # Fix the camera with offset problems
        offsets = state[8 + 2*n_frag : 8 + 2*n_frag + noc]

        index_offsets = [np.where((ObsData['obs'] == oo) * (ObsData['timing']))[0] for oo in offset_obs]
        for idx, offset in zip(index_offsets, offsets):
            T_jd = Time(ObsData['datetime'][idx], format='isot', scale='utc').jd
            ObsData['t_rel'][idx] = (T_jd - t0) * 24*60*60 + offset

        # Determine all the indexes that need adjustments
        index_adj = np.concatenate(([index_notime] + index_offsets))
        
        # Recompute the rotation matrix for those without timing
        obs_llh = np.vstack((np.deg2rad(ObsData['obs_lat'][index_adj]), 
                             np.deg2rad(ObsData['obs_lon'][index_adj]), 
                             ObsData['obs_hei'][index_adj]))
        t_jd = t0 + ObsData['t_rel'][index_adj] / (24*60*60)
        
        [C_ENU2ECI, obs_eci] = enu_matrix(obs_llh, t_jd) #[3,3n], [3,n]
        [ObsData['obs_x'][index_adj], ObsData['obs_y'][index_adj], 
                ObsData['obs_z'][index_adj]] = obs_eci

        alt = np.deg2rad(ObsData['altitude'][index_adj])
        azi = np.deg2rad(ObsData['azimuth'][index_adj])
        UV_enu = np.vstack((np.sin(azi) * np.cos(alt),
            np.cos(azi) * np.cos(alt), np.sin(alt))) #[3,n]

        # Current position in local ENU coords
        n = len(index_adj); UV_enu_prime = UV_enu.T.reshape((n,3,1))
        UV_obs[:,index_adj] = np.matmul(C_ENU2ECI,UV_enu_prime).reshape((n,3)).T #[3,n]

        # Adjust the cov_obs for those without timing
        altaz_cov = np.zeros((n,2,2))
        altaz_cov[:,0,0] = np.deg2rad(ObsData['err_plus_altitude'][index_adj])**2
        altaz_cov[:,1,1] = np.deg2rad(ObsData['err_plus_azimuth'][index_adj])**2

        altaz = np.vstack((alt,azi)).T #[n,2]
        jac = altaz2radec_jac(altaz, C_ENU2ECI) #[n,2,2]
        cov_obs[index_adj] = np.matmul(np.transpose(jac,(0,2,1)),
            np.matmul(altaz_cov, jac)) #[n,2,2]

    return [ObsData, UV_obs, cov_obs, t0, atm_info, offset_obs, n_frag]

def state2posvel(state, args):
    '''
    The azimuth and elevation assuming the meteor trajectory of x.
    Input: The state variable, state=[x0;y0;z0;vx0;vy0;vz0;beta;sigma;t_unknowns].
    Output: The estimated azimuth and elevation for all viewed points from every 
    camera along the meteor's trajectory, y=[2*#obs].

    ObsData: is a table of all the observation information, including 
        [alt, azi, alt_err, azi_err, t_rel, obs_no, obs_llh[3], obs_eci[3]]
    '''
    n_frag = args[6]
    if len(state) > 8 + 2*n_frag: # if not all have timing
        args = adjust_args(state, args)
    ObsData = args[0]; [t0, atm_info] = args[3:5]
    T_rel = np.unique(ObsData['t_rel'])[::-1]

    # Propagate the initial state, X0, to all times, T.
    X = Propagate_frags(state[:8+2*n_frag], T_rel, t0, atm_info)

    # Observed proposed positions [3,#obs] 
    t_no = [int(np.where(T_rel == t_rel)[0][0]) for t_rel in ObsData['t_rel']]
    Cameras_ECI = np.vstack((ObsData['obs_x'], ObsData['obs_y'], ObsData['obs_z'])) #[3,n]
    Pos = X[t_no,:3]; Vel = X[t_no,3:6] #[n,3],[n,3]

    return Pos.T, Vel.T, Cameras_ECI

def dy_dx(state, args):

    Jacobian = np.zeros((2*len(args[0]), len(state)))
    ARGS = [(i, state, args) for i in range(len(state))]
    jac_column_pool = multiprocessing.Pool()
    result_arr = jac_column_pool.map(dy_dx_col, ARGS)
    jac_column_pool.close()
    jac_column_pool.join()

    result_arr.reverse()
    for i in range(len(result_arr)):
        Jacobian[:,i] = result_arr.pop()

    # Jacobian = np.zeros((2*len(args[0]), len(state)))
    # for i in range(len(state)):
    #     Jacobian[:,i] = dy_dx_col((i, state, args))

    return Jacobian

def dy_dx_col(ARGS):
    ''' Evaluates one column of the Jacobian '''
    i, state, args = ARGS

    dx = np.array([1e0,1e0,1e0,2e-1,2e-1,2e-1,1e-1,5e-11])
    dstate = np.concatenate((dx, 1e-4*np.ones(len(state)-8)))
    e = np.zeros(len(state)); e[i] = 1.0

    res_pos = res_weighted(state + dstate*e, args)
    res_neg = res_weighted(state - dstate*e, args)
    return (res_pos - res_neg) / (2*dstate[i])

def residuals(state, args, return_err=False):
    [UV_obs, cov_obs] = args[1:3]
    
    [Pos, Vel, Cameras_ECI] = state2posvel(state, args)
    [ATE, CTE] = track_errors(Pos, Vel, Cameras_ECI, UV_obs)
    res = np.vstack((ATE, CTE)).flatten('f')

    if return_err: # Determine the at/ct error from observation covariance    
        jac = track_errors_radec_jac(Pos, Vel, Cameras_ECI, UV_obs) #[n,2,2]
        res_cov = np.matmul(np.transpose(jac,(0,2,1)), np.matmul(cov_obs, jac)) #[n,2,2]
        res_err = np.sqrt(np.vstack((res_cov[:,0,0], res_cov[:,1,1])).flatten('f')) #[2n]

        return res, res_err
    else:
        return res

def res_weighted(state, args):

    # [res, res_err] = residuals(state, args, True)
    res = residuals(state, args)

    # # Include a cross-track weighting
    # res *= np.array([1,5]*(len(res)//2))

    return res #/ res_err

def FirstApprox(ObsTables, kwargs):
    ''' 
    Using SLLS code to best optimise the trajectory then generate X0_est, 
    where X0_est = [x0, y0, z0, vx0, vy0, vz0, beta0, sigma0]
    '''
    
    logger = logging.getLogger('trajectory')

    # Calculate a first radient approximation
    logger.debug('Calculating initial approximation using MOP')
    [state_slls, args_slls] = FirstApprox_using_tables(ObsTables, eci_bool=False)

    # Compute the straight line fit as the initial trajectory guess
    logger.debug('Beginning Straight Line Least Squares')
    [Pos_ECEF_slls, CTE_slls] = SLLS(state_slls, args_slls, eci_bool=False)[:2]

    # Add some table information
    for obs in range(len(ObsTables)):
        ObsTables[obs]['obs'] = obs
        ObsTables[obs]['offset'] = 0.0 # For now...

    # Stack the data packets 
    ObsData = vstack(ObsTables, metadata_conflicts='silent')
    ObsData_timing = ObsData[ObsData['timing']==1]

    if len(ObsData_timing) == 0:
        logger.error('No timing information')
        raise TriangulationInvalidInput('Cannot perform LS method without any timing information.')

    # Cater for any timing offsets
    [obs_with_timing, obs_counts] = np.unique(ObsData_timing['obs'], return_counts=True)
    if len(obs_with_timing) > 1: # Only if there are two or more cameras with timing

        # Determine the timing offsets from the 'master' camera
        offset_dict = calculate_timing_offsets(Pos_ECEF_slls[:,ObsData['timing']==1], 
            args_slls[3][ObsData['timing']==1], ObsData_timing['obs'])

        # Pick out those cameras with timing offsets greater than 0.05 sec
        offset_obs = [int(obs) for obs, offset in offset_dict.items() if np.abs(offset) > 0.05]
        offset_est = [offset for obs, offset in offset_dict.items() if np.abs(offset) > 0.05]

        # Re-decide the master camera for an accurate t0 value
        master_candidates = [obs for obs in obs_with_timing if obs not in offset_obs]
        mc_length = [np.sum(ObsData_timing['obs'] == mc) for mc in master_candidates]
        master_obs = master_candidates[np.argmax(mc_length)]

        # Update ObsData with our best offset estimation
        for obs, offset in zip(offset_obs, offset_est):
            cam_name = ObsTables[obs].meta['telescope']
            logger.info('{0} has a timing offset of roughly {1:.4f} seconds.'.format(cam_name, offset))
            ObsData['offset'][ObsData['obs'] == obs] = offset
    else:
        offset_obs, offset_est = [], []
        master_obs = obs_with_timing[np.argmax(obs_counts)]

    # Find minimum time of those with timing and no offset
    T_jd = Time(ObsData['datetime'], format='isot',scale='utc').jd
    t0 = np.min(T_jd[(ObsData['timing']) * (ObsData['obs'] == master_obs)])

    # Determine t_rel including the offset
    T_jd_adjusted = T_jd + ObsData['offset'] / (24*60*60)
    ObsData['t_rel'] = np.round((T_jd_adjusted - t0)*24*60*60, 3)

    # Find the suspected fragmentation relative time
    tt_frag = kwargs['frag_times'] 
    if len(tt_frag): tt_frag = Time(tt_frag, format='isot', scale='utc').jd
    t_frag = (tt_frag - t0)*24*60*60; n_frag = len(t_frag)
    
    # Determine the C_eci2enu matrix and obs_eci coordinates
    obs_llh = np.vstack((np.deg2rad(ObsData['obs_lat']), 
        np.deg2rad(ObsData['obs_lon']), ObsData['obs_hei']))
    [C_ENU2ECI, obs_eci] = enu_matrix(obs_llh, T_jd)
    [ObsData['obs_x'], ObsData['obs_y'], ObsData['obs_z']] = obs_eci

    # Generate a lookup table for rho_a(r) and gravity vector
    atm_info = atm_lookup_table(Pos_ECEF_slls, t0)

    # Calculate the LOS unit vector in ECI coordinates
    alt = np.deg2rad(ObsData['altitude'])
    azi = np.deg2rad(ObsData['azimuth'])
    UV_enu = np.vstack((np.sin(azi) * np.cos(alt),
        np.cos(azi) * np.cos(alt), np.sin(alt))) #[3,n]

    # Current position in local ENU coords
    n = len(ObsData); UV_enu_prime = UV_enu.T.reshape((n,3,1))
    UV_obs = np.matmul(C_ENU2ECI,UV_enu_prime).reshape((n,3)).T #[3,n]

    # Extract the observation covariance matrix
    altaz_cov = np.zeros((n,2,2))
    altaz_cov[:,0,0] = np.deg2rad(ObsData['err_plus_altitude'])**2
    altaz_cov[:,1,1] = np.deg2rad(ObsData['err_plus_azimuth'])**2

    # Convert the altaz covariance to radec
    altaz = np.vstack((alt,azi)).T #[n,2]
    jac = altaz2radec_jac(altaz, C_ENU2ECI) #[n,2,2]
    cov_obs = np.matmul(np.transpose(jac,(0,2,1)),
        np.matmul(altaz_cov, jac)) #[n,2,2]

    # Determine the initial position, velocity, and unknown timing
    [state_est, slls_len] = state_estimate(Pos_ECEF_slls, 
        ObsData, t0, atm_info, offset_est, t_frag)

    # Adjust the arguments for the new timing estimates
    args_est = [ObsData, UV_obs, cov_obs, t0, atm_info, offset_obs, n_frag]
    args = adjust_args(state_est, args_est)

    # Refine beta guess - based on trajectory length
    # state_est[:8 + 2*n_frag] = RefineBeta(state_est[:8 + 2*n_frag], slls_len, args)
    # state_est[:8 + 2*n_frag] = RefineVelocity(state_est[:8 + 2*n_frag], slls_len, args)

    pts_without_timing = len(state_est) - 8 - 2*n_frag - len(offset_est)
    logger.info('Estimating {:d} points without timing.'.format(pts_without_timing))
    
    slls_args = [Pos_ECEF_slls, CTE_slls]
    return state_est, args, slls_args


def atm_lookup_table(Pos_ECEF_slls, t0):

    n = 400
    index_min = np.argmin(norm(Pos_ECEF_slls, axis=0))
    index_max = np.argmax(norm(Pos_ECEF_slls, axis=0))
    Pos_ECEF_min = Pos_ECEF_slls[:,index_min:index_min+1]
    Pos_ECEF_max = Pos_ECEF_slls[:,index_max:index_max+1]
    pos_interp = Pos_ECEF_min + np.linspace(-0.2,1.2,n) * (Pos_ECEF_max - Pos_ECEF_min)

    radius = norm(pos_interp, axis=0)
    rho_a = [NRLMSISE_00(np.vstack(p), t0, 'ecef')[2] for p in pos_interp.T]
    [grav_x, grav_y, grav_z] = gravity_vector(pos_interp)

    atm_info = [radius, rho_a, grav_x, grav_y, grav_z]
    return atm_info

def first_and_last_positions(Pos_ECEF_slls, ObsData, T_rel):

    # Last position with timing
    Pos_ECEF_end = Pos_ECEF_slls[:,ObsData['t_rel']==T_rel[0]]
    Pos_ECEF_end = np.vstack((np.mean(Pos_ECEF_end, axis=1)))

    # First position with timing
    Pos_ECEF_beg = Pos_ECEF_slls[:,ObsData['t_rel']==T_rel[-1]]
    Pos_ECEF_beg = np.vstack((np.mean(Pos_ECEF_beg, axis=1)))

    return Pos_ECEF_beg, Pos_ECEF_end

def state_estimate(Pos_ECEF_slls, ObsData, t0, atm_info, offset_est, t_frag):
    
    logger = logging.getLogger('trajectory')
    
    # Find the rough velocity direction
    T_rel = np.unique(ObsData['t_rel'][ObsData['timing']==1])[::-1]
    [Pos_ECEF_beg, Pos_ECEF_end] = first_and_last_positions(Pos_ECEF_slls, ObsData, T_rel)
    Vel_UV = (Pos_ECEF_end - Pos_ECEF_beg) / norm(Pos_ECEF_end - Pos_ECEF_beg)

    # Determine rough timing for those without - length vs time
    lengths_all = Vel_UV.T.dot(Pos_ECEF_slls - Pos_ECEF_beg).flatten()
    T_rel_timing = ObsData['t_rel'][ObsData['timing']==1]
    t_mat = np.vstack((np.ones(len(T_rel_timing)), T_rel_timing))#, 0.5*T_rel_timing**2))
    pva = lengths_all[ObsData['timing']==1].dot(t_mat.T).dot(inv(t_mat.dot(t_mat.T)))
    [l0, v0] = pva.flatten()
    T_rel_notiming = (lengths_all[ObsData['timing']==0] - l0) / v0
    # T_rel_notiming = (np.sqrt(v0**2 - 2*a0*(l0 - lengths_all[ObsData['timing']==0])) -v0) / a0

    # plt.figure()
    # plt.plot(T_rel_timing, lengths_all[ObsData['timing']==1], label='timing')
    # plt.plot(T_rel_notiming, lengths_all[ObsData['timing']==0], label='notiming')
    # plt.legend(loc=0); plt.show()

    # Set the no-timing estimates in the ObsData table and recompute the positions
    ObsData['t_rel'][ObsData['timing']==0] = T_rel_notiming
    T_rel = np.unique(ObsData['t_rel'])[::-1]
    [Pos_ECEF_beg, Pos_ECEF_end] = first_and_last_positions(Pos_ECEF_slls, ObsData, T_rel)
    Pos_ECI_beg = ECEF2ECI_pos(Pos_ECEF_beg, t0 + T_rel[-1]/(24*60*60))

    # Use least squares to find the final velocity magnitude
    last_cam = ObsData['obs'] == ObsData['obs'][ObsData['t_rel']==T_rel[0]][0]
    # last_cam_with_timing = (last_cam) & (ObsData['timing']==1)
    # t_crop = ObsData['t_rel'][last_cam_with_timing][-8:]
    t_crop = ObsData['t_rel'][last_cam][-8:]
    pos_last_cam = Pos_ECEF_slls[:,last_cam][:,-8:]

    lengths = Vel_UV.T.dot(pos_last_cam - Pos_ECEF_end)
    t_matrix = np.vstack((np.ones(len(t_crop)), t_crop))#, 0.5*t_crop**2))

    # 1D least squares to guess velocity
    pva = lengths.dot(t_matrix.T).dot(inv(t_matrix.dot(t_matrix.T)))
    [l0, v0] = pva.flatten() #, a0
    # Vel_ECEF0 = (v0 + a0 * T_rel[0]) * Vel_UV
    Vel_ECEF_end = v0 * Vel_UV

    [Pos_ECI_end, Vel_ECI_end] = ECEF2ECI(Pos_ECEF_end, 
        Vel_ECEF_end, t0 + T_rel[0]/(24*60*60))
    logger.info('Final velocity estimate: {:.3f} km/s'.format(norm(Vel_ECI_end)/1000))

    # Initial parameter guesses - 100g
    beta = ((3500.0)**(2./3) * 0.1**(1./3)) / (1.6*1.3)
    sigma = 0.014e-6

    # Setup the intial state guess
    X_est = np.array([Pos_ECI_end[0][0], Pos_ECI_end[1][0], Pos_ECI_end[2][0], 
        Vel_ECI_end[0][0], Vel_ECI_end[1][0], Vel_ECI_end[2][0], beta, sigma])

    # Create the state vector - include fragments and timing guesses
    state_est = np.concatenate((X_est, 0.3*np.ones(len(t_frag)),\
        np.sort(t_frag)[::-1], np.array(offset_est), T_rel_notiming))

    return state_est, norm(Pos_ECI_end - Pos_ECI_beg)


def RefineVelocity(X0_est, slls_len, args):
    
    logger = logging.getLogger('trajectory')

    logger.info('Refining the final velocity estimate')
    logger.info('Velocity before: {:.3f} km/s'.format(norm(X0_est[3:6])/1000))

    ObsData = args[0]; [t0, atm_info] = args[3:5]
    T_rel = np.unique(ObsData['t_rel'])[::-1]

    def len_diff(vel, X0_est, T_rel, t0, atm_info, slls_len):
        X0_est[3:6] = vel * X0_est[3:6] / norm(X0_est[3:6])
        X_est = Propagate_frags(X0_est, T_rel, t0, atm_info)
        traj_len = norm(X_est[-1,:3] - X_est[0,:3])
        return traj_len - slls_len

    # plt.figure(figsize=(16,9))
    # lb = np.linspace(1e3,80e3,100)
    # plt.plot(lb, [len_diff(lbi, X0_est, T_rel, t0, atm_info, slls_len)
    #     for lbi in lb], '.-')
    # plt.show()

    try:
        vel = brentq(len_diff, 1e3, 80e3, xtol=1,
            args=(X0_est, T_rel, t0, atm_info, slls_len))
    except ValueError:
        X0_est[6] = ((3500.0)**(2./3) * 100**(1./3)) / (1.6*1.3) #100kg
        vel = brentq(len_diff, 1e3, 80e3, xtol=1,
            args=(X0_est, T_rel, t0, atm_info, slls_len))
    X0_est[3:6] = vel * X0_est[3:6] / norm(X0_est[3:6])

    # print('Velocity after:  {:.3f} km/s'.format(norm(X0_est[3:6])/1000)); input()
    logger.info('Velocity after:  {:.3f} km/s'.format(norm(X0_est[3:6])/1000))
    return X0_est

def RefineBeta(X0_est, slls_len, args):
    
    logger = logging.getLogger('trajectory')
    
    logger.info('Refining the ballistic estimate')
    # print('Beta before: {:.1f} kg/m^2'.format(norm(X0_est[6])))

    ObsData = args[0]; [t0, atm_info] = args[3:5]
    T_rel = np.unique(ObsData['t_rel'])[::-1]#; beta0 = X0_est[6]

    def len_diff(log_beta, X0_est, T_rel, t0, atm_info, slls_len):
        X0_est[6] = 10**log_beta; X_est = Propagate_frags(X0_est, T_rel, t0, atm_info)
        traj_len = norm(X_est[-1,:3] - X_est[0,:3])
        return traj_len - slls_len

    plt.figure(figsize=(16,9))
    lb = np.logspace(0,12,100)
    plt.plot(lb, [len_diff(lbi, X0_est, T_rel, t0, atm_info, slls_len)
        for lbi in lb], '.-')
    plt.xscale('log'); plt.yscale('log')
    plt.show()

    try:
        ref_log_beta = brentq(len_diff, 1, 4, xtol=0.01,
                args=(X0_est, T_rel, t0, atm_info, slls_len))
        X0_est[6] = 10**ref_log_beta
    except ValueError:
        X0_est[6] = 10 # Near zero mass
        logger.warning('ValueError: Skipped ballistic refinement.')

    print('Estimated ballistic parameter: {:.1f} kg/m^2'.format(X0_est[6])); input()
    logger.info('Estimated ballistic parameter: {:.1f} kg/m^2'.format(X0_est[6]))
    return X0_est

def LeastSquares(state_est, args):
    ''' Where the magic happens '''
    
    logger = logging.getLogger('trajectory')

    # Bounds for the least squares minimisation
    pe = 40e3; ve = 5e3; n_frag = args[6] # 40km pos err, 5km/s vel err
    state_lb = state_est.copy(); state_lb[:3] -= pe; state_lb[3:6] -= ve; 
    state_lb[6] = 1e-10; state_lb[7] = 3e-9; state_lb[8:8+n_frag] = 0
    state_lb[8+n_frag:8+2*n_frag] = np.min(args[0]['t_rel']); state_lb[8+2*n_frag:] -= 10.0
    state_ub = state_est.copy(); state_ub[:3] += pe; state_ub[3:6] += ve; 
    state_ub[6] = 1e4; state_ub[7] = 3e-6; state_ub[8:8+n_frag] = 1
    state_ub[8+n_frag:8+2*n_frag] = np.max(args[0]['t_rel']); state_ub[8+2*n_frag:] += 10.0

    # Step size for the least squares minimisation - same step as jacobian
    x_step = np.array([1e0,1e0,1e0,2e-1,2e-1,2e-1,1e-1,5e-11])
    state_step = np.concatenate((x_step, 1e-4*np.ones(len(state_est)-8)))

    # Algorithms
    t = time.time()

    ''' Least-Squares algorithm - local optimisation '''
    # Best so far, but need a good initial guess to find optimal solution
    logger.info('Removing straight line assumption')
    # if n_frag: # Fragmentation not consistent. Try without first.
    #     state_est[8:8+n_frag] = 0.0005
    #     state_ub_no_frag = state_ub.copy(); state_ub_no_frag[8:8+n_frag] = 0.001
    #     results = least_squares(res_weighted, state_est, diff_step=state_step, x_scale='jac', 
    #         jac=dy_dx, bounds=(state_lb, state_ub_no_frag), args=(args,), method='trf', verbose=2)
    #     state_est = results.x; state_est[8:8+n_frag] = 0.3

    results = least_squares(res_weighted, state_est, diff_step=state_step, x_scale='jac', 
        jac=dy_dx, bounds=(state_lb, state_ub), args=(args,), method='trf', verbose=2)
    state = results.x

    ''' Differential evolution algorithm - global optimisation '''
    # # Jumps around a lot within the bounds, takes a while to find the solution.
    # bnds = tuple(map(tuple, np.vstack((state_lb, state_ub)).T))
    # results = differential_evolution(residuals, bounds=bnds, args=(args,), disp=True)
    
    eval_time = time.time() - t
    logger.debug('Calculation time (s) = %.2f' % eval_time)

    # Extract X0 from the state
    X0 = state[:8 + 2*n_frag]
    offsets = state[8 + 2*n_frag:8 + 2*n_frag + len(args[5])]

    # # https://au.mathworks.com/help/stats/nlinfit.html?requestedDomain=www.mathworks.com
    # jac_f = dy_dx(state, args)[:,:8]; res_f = residuals(state, args)
    # N_coeff = len(state); N_obs = len(res_f)
    # mse = np.sum(res_f**2) / (N_obs - N_coeff)
    # X0_cov_res = mse * np.linalg.inv( jac_f.T.dot(jac_f) )

    ################

    # Compute the covariance due to the spread in residuals
    jac_f = dy_dx(state, args)[:,:8]; res_f = residuals(state, args)
    dx_dres = (np.linalg.inv(jac_f.T.dot(jac_f)).dot(jac_f.T)).T #[2n,8]
    X0_cov_res = dx_dres.T.dot( np.diag(res_f**2) ).dot(dx_dres)

    # Compute the covariance due to the errors in measurement -> same errors that MC gives!
    [UV_obs, cov_obs] = args[1:3]
    [Pos, Vel, Cameras_ECI] = state2posvel(state, args)
    
    dres_dz = track_errors_radec_jac(Pos, Vel, Cameras_ECI, UV_obs) #[n,2,2]
    dres_dz = block_diag(*[j for j in dres_dz]) #[2n,2n]
    cov_z = block_diag(*[c for c in cov_obs]) #[2n,2n]
    cov_res = dres_dz.T.dot(cov_z).dot(dres_dz) #[2n,2n]
    X0_cov_z = dx_dres.T.dot(cov_res).dot(dx_dres)

    # Total covariance on the state
    X0_cov = X0_cov_res + X0_cov_z

    ################
    cov_data = np.vstack((np.sqrt(np.diag(X0_cov_res)),
        np.sqrt(np.diag(X0_cov_z)),np.sqrt(np.diag(X0_cov)))).T
    labels = ['px','py','pz','vx','vy','vz','beta']

    logger.debug('ERRORS:    residual | measurement |       total')
    logger.debug('------:-------------|-------------|------------')
    for row, data in zip(labels,cov_data[:7]):
        logger.debug(' {0:5s}: {1:11.3f} | {2:11.3f} | {3:11.3f}'.format(
            row, data[0], data[1], data[2]))
    logger.debug(' sigma: {0:11.2e} | {1:11.2e} | {2:11.2e}'.format(
        cov_data[7,0], cov_data[7,1], cov_data[7,2]))
    ################

    # Update args with timing
    if len(state) > 8 + 2*n_frag: # if all haven't got timing
        args = adjust_args(state, args)

    return X0, X0_cov, args, offsets, cov_res

def residual_components(X, T_rel, args, cov_res):
    [ObsData, UV_obs, cov_obs, t0, atm_info, offset_obs, n_frag] = args

    # Estimated LOS based on model fit 
    t_no = [int(np.where(T_rel == t_rel)[0][0]) for t_rel in ObsData['t_rel']]
    Cameras_ECI = np.vstack((ObsData['obs_x'], ObsData['obs_y'], ObsData['obs_z']))
    Pos = X[t_no,:3]; Vel = X[t_no,3:6] #[n,3],[n,3]

    # Determine the cross/along track errors
    [ATE, CTE] = track_errors(Pos.T, Vel.T, Cameras_ECI, UV_obs)

    # Determine the cross/along track measurement uncertainty
    res_std = np.sqrt(np.diag(cov_res))
    [ATE_std, CTE_std] = [res_std[::2], res_std[1::2]]

    LOS_errors = np.vstack((ObsData['obs'], ObsData['t_rel'], 
        norm(Pos.T - Cameras_ECI, axis=0), ATE, ATE_std, CTE, CTE_std))
    
    return LOS_errors

def print_output(X, X_cov, segment='both'):
    
    logger = logging.getLogger('trajectory')
    
    if X.ndim == 1:
        X = np.vstack((X))
    if X_cov.ndim == 2:
        X_cov = np.dstack((X_cov))

    Pos_ECI = X[:3]
    
    if segment == 'both' or segment == 'beg':
        logger.info('Orbit conditions:')
        lat_initial = np.arcsin(Pos_ECI[2,0] / norm(Pos_ECI[:,0]))
        hei_initial = norm(Pos_ECI[:,0]) - EarthRadius(lat_initial)

        logger.info('Altitude = {0:.3f} +- {1:.3f} km'.format(hei_initial/1000,
            np.sqrt(np.sum([X_cov[0,i,i] for i in range(3)]))/1000))
        logger.info('Vel_eci = {0:.3f} +- {1:.3f} km/s'.format(norm(X[3:6,0])/1000, 
            np.sqrt(np.sum([X_cov[0,i,i] for i in range(3,6)]))/1000))
        logger.info('beta = {0:.2e} +- {1:.2e} kg/m^2'.format(X[6,0], np.sqrt(X_cov[0,6,6])))
        logger.info('sigma = {0:.2e} +- {1:.2e} s^2/m^2'.format(X[7,0], np.sqrt(X_cov[0,7,7])))

    if segment == 'both' or segment == 'end':
        logger.info('Darkflight conditions:')
        lat_final = np.arcsin(Pos_ECI[2,-1] / norm(Pos_ECI[:,-1]))
        hei_final = norm(Pos_ECI[:,-1]) - EarthRadius(lat_final)

        logger.info('Altitude = {0:.3f} +- {1:.3f} km'.format(hei_final/1000, 
            np.sqrt(np.sum([X_cov[-1,i,i] for i in range(3)]))/1000))
        logger.info('Vel_eci = {0:.3f} +- {1:.3f} km/s'.format(norm(X[3:6,-1])/1000, 
            np.sqrt(np.sum([X_cov[-1,i,i] for i in range(3,6)]))/1000))
        logger.info('beta = {0:.2e} +- {1:.2e} kg/m^2'.format(X[6,-1], np.sqrt(X_cov[-1,6,6])))
        logger.info('sigma = {0:.2e} +- {1:.2e} s^2/m^2'.format(X[7,-1], np.sqrt(X_cov[-1,7,7])))


def check_camera_height_overlaps(min_heights, max_heights):
    heights = np.vstack((min_heights, max_heights)).T #[n,2]

    # Construct the camera height range overlap bins
    cams_to_remove = []
    for i, [min_h, max_h] in enumerate(heights):

        h_other = np.vstack((heights[:i], heights[i+1:]))
        if not np.any((max_h > h_other[:,0]) * (min_h < h_other[:,1])):# or min_h < 0 or max_h > 200e3:
            cams_to_remove.append(i) # should probably handle the other criteria separately -^

    if len(cams_to_remove) > 1: # If there are more than one, take the further away one
        mean_separation = np.mean(heights) - np.mean(heights[cams_to_remove], axis=1)
        cam_to_remove = [cams_to_remove[np.argmax(np.abs(mean_separation))]]
    else:
        cam_to_remove = cams_to_remove

    return cam_to_remove


def sanity_checks(state_est, args, slls_args, ObsTables):
    # Raising errors if sanity checks fail
    
    logger = logging.getLogger('trajectory')
    fails_messages = []
    
    [ObsData, UV_obs, cov_obs, t0, atm_info, offset_obs, n_frag] = args
    [Pos_ECEF_slls, CTE_slls] = slls_args

    # Check the event has consistant timing
    height_slls = ECEF2LLH(Pos_ECEF_slls[:,ObsData['timing']])[2] #[n]
    obs_slls = ObsData['obs'][ObsData['timing']]
    t_rel_slls = ObsData['t_rel'][ObsData['timing']]
    unique_obs_timing = np.unique(obs_slls)
    unique_obs = np.unique(ObsData['obs'])

    min_heights = np.array([np.min(height_slls[obs_slls==o]) for o in unique_obs_timing])
    max_heights = np.array([np.max(height_slls[obs_slls==o]) for o in unique_obs_timing])

    # plt.figure()
    # for o in unique_obs_timing:
    #     plt.plot(t_rel_slls[obs_slls==o], height_slls[obs_slls==o]/1000, 
    #         label=ObsTables[int(o)].meta['telescope'])
    # plt.xlabel('Relative time [s]'); plt.ylabel('Height [km]')
    # plt.legend(); plt.show()

    ###################################################
    # Remove a camera if it is increasing height with time

    # Check the SLLS height is decreasing with time
    rough_vel = np.zeros(len(unique_obs_timing))
    for i, o in enumerate(unique_obs_timing): # Fit a straight line to the h(t) plot
        t_obs = t_rel_slls[obs_slls==o]; height_obs = height_slls[obs_slls==o]
        t_mat = np.vstack((t_obs, np.ones(len(t_obs)))).T #[n,2]
        [v_h, h0] = np.linalg.inv(t_mat.T.dot(t_mat)).dot(t_mat.T).dot(height_obs) #[2]
        rough_vel[i] = v_h 

    # Check the SLLS height is decreasing with time
    if np.any(rough_vel > 0):
        msg = 'At least one camera predicts the fireball going up.'
        if len(unique_obs) > 2:
            logger.warning(msg)
            return np.argmax(rough_vel)
        else:
            logger.error(msg); fails_messages += [msg]

    ###################################################
    # Remove a camera if it isn't decreasing in height like the rest

    # Check the SLLS height is decreasing at roughly the same rate
    median_vel = np.median(rough_vel)
    if np.any(np.abs(median_vel - rough_vel) > 8e3): #8km/s
        msg = "Trajectory height vs time is inconsistent."
        if len(np.unique(ObsData['obs'])) > 2: # Check we can try ditching a camera

            cam_to_remove = check_camera_height_overlaps(min_heights, max_heights)
            logger.warning(msg)
            
            if len(cam_to_remove) != 0:
                return int(unique_obs_timing[cam_to_remove])
            else:
                return np.argmax(np.abs(median_vel - rough_vel))
        else:
            logger.error(msg); fails_messages += [msg]
    
    ###################################################
    # Remove a camera if it's residuals are > 0.5 deg

    obs_llh = np.vstack((np.deg2rad(ObsData['obs_lat']), 
        np.deg2rad(ObsData['obs_lon']), ObsData['obs_hei']))
    CTE_slls_ang = CTE_slls / norm(Pos_ECEF_slls - LLH2ECEF(obs_llh), axis=0)
    ctr_list = [CTE_slls_ang[ObsData['obs'] == obs] for obs in np.unique(ObsData['obs'])]
    residuals_std = np.array([np.sqrt(np.sum(ctr**2) / len(ctr)) for ctr in ctr_list])

    if np.any(np.rad2deg(residuals_std) > 0.5): # half a degree... 
        msg = "The trajectory has abnormally high SLLS residuals."
        if len(unique_obs) > 2:
            logger.warning(msg)
            return np.argmax(residuals_std)
        else:
            logger.error(msg); fails_messages += [msg]

    ###################################################
    # Other fail conditions...

    # Check the SLLS height is above the ground
    if np.any(height_slls < 0): 
        msg = "The trajectory is predicted through the ground."
        logger.error(msg); fails_messages += [msg]

    # Check the SLLS height is always < 200km
    if np.any(height_slls > 200e3):
        msg = "The maximum height of {0:.0f}km is > 200km...".format(np.max(height_slls)/1e3)
        logger.error(msg); fails_messages += [msg]

    # Check the predicted state velocity < 200km/s
    if norm(state_est[3:6]) > 200e3:
        msg = "The predicted velocity of {:.0f}km/s is > 200km/s...".format(norm(state_est[3:6])/1000)
        logger.error(msg); fails_messages += [msg]

    # Plot the timing inconsistancy and raise the errors
    if len(fails_messages) > 0:

        plt.figure()
        for o in unique_obs_timing:
            plt.plot(t_rel_slls[obs_slls==o], height_slls[obs_slls==o]/1000, 
                label=ObsTables[int(o)].meta['telescope'])
        plt.xlabel('Relative time [s]'); plt.legend(loc=0)
        plt.ylabel('Straight Line Least Squares Height [km]')
        
        event_codename = ObsData.meta['event_codename']
        PointDir = os.path.dirname(ObsData.meta['self_disk_path'])

        ofile_pic = os.path.join(PointDir, event_codename+'_SLLS_timing_consistancy_check.png')
        plt.savefig(ofile_pic, format='png')

        logger.debug('Timing consistancy check has been saved to: ' + ofile_pic)

        raise TriangulationOutOfRange('Trajectory sanity fail.')

    return None

def main_using_table(ObsTables, kwargs):
    
    logger = logging.getLogger('trajectory')

    # First approximation
    [state_est, args, slls_args] = FirstApprox(ObsTables, kwargs)    

    # Catch the dodgy events with some sanity checks
    kept_obs = np.ones(len(ObsTables)).astype(bool)
    ''''''
    obs_to_remove = sanity_checks(state_est, args, slls_args, ObsTables)
    unique_obs = np.unique(args[0]['obs'])
    
    while obs_to_remove is not None and len(np.array(unique_obs)) > 2:

        cam_name = ObsTables[obs_to_remove].meta['telescope']
        logger.info('Attempting to remove {0} from the triangulation...'.format(cam_name))

        # Remove the problem camera and try the first approximation again
        del(ObsTables[obs_to_remove]); # v--- Bit nasty, but don't wanna recode everything
        kept_obs[(np.cumsum(kept_obs)-1) == obs_to_remove] = False 
        [state_est, args, slls_args] = FirstApprox(ObsTables, kwargs)
        obs_to_remove = sanity_checks(state_est, args, slls_args, ObsTables)
    ''''''

    # Least Squares business
    [Xn, Xn_cov, args_new, offsets, cov_res] = LeastSquares(state_est, args)
    [ObsData, UV_obs, cov_obs, t0, atm_info, offset_obs, n_frag] = args_new

    # Save the fragment information in the log file
    for p_f, t_f in zip(*np.hsplit(Xn[8:8 + 2*args[6]], 2)):
        t_isot = Time(t_f / (24*60*60) + t0, format='jd', scale='utc').isot
        logger.info('At {0} there is a determined mass loss of {1:.4f} percent.'.format(t_isot, p_f * 100))

    # Save the offsets in the log file
    for obs, offset in zip(offset_obs, offsets):
        cam_name = ObsTables[obs].meta['telescope']
        logger.info('{0} has a determined offset of {1:.4f} seconds.'.format(cam_name, offset))

    # ''' Check the errors through MC analysis '''
    # mean1run = Xn; cov1run = Xn_cov; mc = 200

    # means = np.zeros((len(Xn), mc))
    # covs = np.zeros((len(Xn), len(Xn), mc))
    # for i in range(mc):

    #     ObsTables_rand = copy.deepcopy(ObsTables)
    #     for o in range(len(ObsTables_rand)):
    #         ObsTables_rand[o]['altitude'] = np.random.normal(ObsTables_rand[o]['altitude'], 
    #                                                 2*ObsTables_rand[o]['err_plus_altitude'])
    #         ObsTables_rand[o]['azimuth'] = np.random.normal(ObsTables_rand[o]['azimuth'], 
    #                                                 2*ObsTables_rand[o]['err_plus_azimuth'])
    #     # First approximation
    #     [state_rand, args_rand] = FirstApprox(ObsTables_rand, cam_names)[:2]
    #     [means[:,i], covs[:,:,i]] = LeastSquares(state_rand, args_rand)[:2]

    # Xn = np.mean(means, axis=1)
    # Xn_cov = np.cov(means)

    # main_cov = np.sqrt(np.diag(cov1run))
    # mc_cov_mean = np.sqrt(np.diag(np.cov(means)))
    # mc_mean_cov = np.sqrt(np.diag(np.mean(covs, axis=2)))

    # print('1 run pos_x_cov:', cov1run[0,0])
    # print('MC cov(means) pos_x_cov:', np.cov(means[0]))
    # print('MC min(cov) pos_x_cov:', np.min(covs[0,0]))
    # print('MC mean(cov) pos_x_cov:', np.mean(covs[0,0]))
    # print('MC max(cov) pos_x_cov:', np.max(covs[0,0]))

    # print('1 run vel_x_cov:', cov1run[3,3])
    # print('MC cov(means) vel_x_cov:', np.cov(means[3]))
    # print('MC min(cov) vel_x_cov:', np.min(covs[3,3]))
    # print('MC mean(cov) vel_x_cov:', np.mean(covs[3,3]))
    # print('MC max(cov) vel_x_cov:', np.max(covs[3,3]))

    # print('\n1 run pos_std:', norm(main_cov[:3]))
    # print('MC cov(means) pos_std:', norm(mc_cov_mean[:3]))
    # print('MC mean(cov) pos_std:', norm(mc_mean_cov[:3]))

    # print('1 run vel_std:', norm(main_cov[3:6]))
    # print('MC cov(means) vel_std:', norm(mc_cov_mean[3:6]))
    # print('MC mean(cov) vel_std:', norm(mc_mean_cov[3:6]))

    # print('1 run beta_std:', main_cov[6])
    # print('MC cov(means) beta_std:', mc_cov_mean[6])
    # print('MC mean(cov) beta_std:', mc_mean_cov[6])

    # print('1 run sigma_std:', main_cov[7])
    # print('MC cov(means) sigma_std:', mc_cov_mean[7])
    # print('MC mean(cov) sigma_std:', mc_mean_cov[7])

    # ''''''


    # Determined positions from X0 estimate
    logger.info('Propagating errors')
    t = time.time()
    [T_rel, t_counts] = np.unique(ObsData['t_rel'], return_counts=True)
    T_rel_rev = T_rel[::-1] #Propagate from the bottom-up
    [X, X_cov] = Propagate_frags(Xn, T_rel_rev, t0, atm_info, Xn_cov)
    X = X[::-1]; X_cov = X_cov[::-1] #Flip back into order
    logger.debug('Done in {:.2f} seconds.'.format(time.time() - t))

    # Along and cross track error between LOS and point
    LOS_errors = residual_components(X, T_rel, args_new, cov_res)

    time_args = [T_rel, t_counts, t0, ObsData['timing'].astype(bool)]
    plt_args = [LOS_errors, slls_args]
    return X.T, X_cov, time_args, plt_args, kept_obs # Leave the X transposed until you have time to change the lot #FIXME


def main(file_list, kwargs, tri_method='LS'):

    logger = logging.getLogger('trajectory')

    ObsTables, CameraTables, cam_names = [], [], []
    for cam, file_name in enumerate(file_list):

        # Read the file
        data = Table.read(file_name, format='ascii.ecsv', guess=False, delimiter=',')
        data.meta['self_disk_path'] = file_name
        cam_names.append( data.meta['telescope'] )

        time_err = data['time_err_plus'] + data['time_err_minus']
        data['timing'] = (time_err < 0.1)
        logger.info('{} has all timing = {}'.format(cam_names[cam], np.all(data['timing'])))

        # Give the camera a number and record its coordinates
        data['obs_lat'] = data.meta['obs_latitude']
        data['obs_lon'] = data.meta['obs_longitude']
        data['obs_hei'] = data.meta['obs_elevation']

        # Sort the data in reverse time
        data.sort(['datetime'])

        CameraTables.append(data)
        ObsTables.append(data['datetime','timing','time_err_minus','time_err_plus',
            'azimuth','altitude','err_plus_azimuth','err_plus_altitude', 
            'obs_lat','obs_lon','obs_hei'])

    # Do the business...
    if tri_method == 'LS':
        [X, X_cov, time_args, plt_args, kept_obs] = main_using_table(ObsTables, kwargs)
        file_list = [file for file, kept in zip(file_list, kept_obs) if kept]
        cam_names = [name for name, kept in zip(cam_names, kept_obs) if kept]
        CameraTables = [ct for ct, kept in zip(CameraTables, kept_obs) if kept]

    elif tri_method == 'UKF':
        from UKF_Triangulation import main_using_table as main_ukf
        [state_args, time_args, plt_args] = main_ukf(ObsTables, kwargs)
        [X_filter, X_cov_filter, X, X_cov] = state_args

    elif tri_method == 'IMM':
        from IMM_Triangulation import main_using_table as main_imm
        [state_args, time_args, plt_args] = main_imm(ObsTables, kwargs)
        [X_filter, X_cov_filter, X, X_cov] = state_args

    [T_rel, t_counts, t0, timing] = time_args
    [LOS_errors, slls_args] = plt_args
    print_output(X, X_cov)

    # Save to file =========================================================================
    # Decode the state vector
    Pos_ECI = X[:3]; Vel_ECI = X[3:6]; beta = X[6]; sigma = X[7]

    # Decode the state covariance
    X_cov_diag = np.hstack([np.vstack(np.diag(cov)) for cov in X_cov])
    Pos_ECI_err = np.sqrt(X_cov_diag[0:3]); pos_eci_err = norm(Pos_ECI_err, axis=0)
    Vel_ECI_err = np.sqrt(X_cov_diag[3:6]); vel_eci_err = norm(Vel_ECI_err, axis=0)
    beta_err = np.sqrt(X_cov_diag[6]); sigma_err = np.sqrt(X_cov_diag[7])

    # Define the universal beta-to-mass constant
    sphere_ass = (1.21 * 0.92)**3 / 3500**2 # Assume a sphere
    mass_est = sphere_ass * beta**3
    mass_err = 3 * sphere_ass * beta**2 * beta_err
    # TODO: fix up the mass errors to quote +ve and -ve errors separately #FIXME
    
    if tri_method == 'UKF' or tri_method == 'IMM':
        vel_filter = norm(X_filter[3:6], axis=0)
        beta_filter = X_filter[6]
        mass_filter = sphere_ass * X_filter[6]**3

        X_cov_diag_f = np.hstack([np.vstack(np.diag(cov)) for cov in X_cov_filter])
        Pos_filter_err = np.sqrt(X_cov_diag_f[0:3]); pos_filter_err = norm(Pos_filter_err, axis=0)
        Vel_filter_err = np.sqrt(X_cov_diag_f[3:6]); vel_filter_err = norm(Vel_filter_err, axis=0)
        beta_filter_err = np.sqrt(X_cov_diag_f[6])
        mass_filter_err = sphere_ass * ((beta_filter + beta_filter_err)**3 - beta_filter**3)

    # Coordinate transforms
    T_jd = t0 + T_rel / (24*60*60)
    [Pos_ECEF, Vel_ECEF] = ECI2ECEF(Pos_ECI, Vel_ECI, T_jd)
    Pos_LLH = ECEF2LLH(Pos_ECEF)
    vel_geo = norm(Vel_ECEF, axis=0)
    vel_eci = norm(Vel_ECI, axis=0)

    # Construct the triangulation table with all observations
    TriTable = Table()
    TriTable['datetime'] = Time(T_jd, format='jd', scale='utc').isot
    TriTable['t_rel'] = T_rel*u.second
    TriTable['no_cams'] = t_counts
    TriTable['latitude'] = np.rad2deg(Pos_LLH[0])*u.deg
    TriTable['longitude'] = np.rad2deg(Pos_LLH[1])*u.deg
    TriTable['height'] = Pos_LLH[2]*u.m
    TriTable['X_geo'] = Pos_ECEF[0]*u.m
    TriTable['Y_geo'] = Pos_ECEF[1]*u.m
    TriTable['Z_geo'] = Pos_ECEF[2]*u.m
    TriTable['X_eci'] = Pos_ECI[0]*u.m
    TriTable['Y_eci'] = Pos_ECI[1]*u.m
    TriTable['Z_eci'] = Pos_ECI[2]*u.m
    TriTable['DX_DT_geo'] = Vel_ECEF[0]*u.m/u.second
    TriTable['DY_DT_geo'] = Vel_ECEF[1]*u.m/u.second
    TriTable['DZ_DT_geo'] = Vel_ECEF[2]*u.m/u.second
    TriTable['DX_DT_eci'] = Vel_ECI[0]*u.m/u.second
    TriTable['DY_DT_eci'] = Vel_ECI[1]*u.m/u.second
    TriTable['DZ_DT_eci'] = Vel_ECI[2]*u.m/u.second
    TriTable['D_DT_geo'] = vel_geo*u.m/u.second
    TriTable['D_DT_eci'] = vel_eci*u.m/u.second
    TriTable['beta'] = beta*u.kg/u.m**2.
    TriTable['sigma'] = sigma*u.s**2./u.m**2.
    TriTable['mass_est'] = mass_est*u.kg
    # Error columns
    TriTable['pos_err'] = pos_eci_err*u.m
    TriTable['D_DT_err'] = vel_eci_err*u.m/u.second
    TriTable['beta_err'] = beta_err*u.kg/u.m**2.
    TriTable['sigma_err'] = sigma_err*u.s**2./u.m**2.
    TriTable['mass_err'] = mass_err*u.kg
    # Add the covariance terms
    TriTable['c11'] = X_cov[:,0,0]; TriTable['c36'] = X_cov[:,2,5]
    TriTable['c12'] = X_cov[:,0,1]; TriTable['c37'] = X_cov[:,2,6]
    TriTable['c13'] = X_cov[:,0,2]; TriTable['c38'] = X_cov[:,2,7]
    TriTable['c14'] = X_cov[:,0,3]; TriTable['c44'] = X_cov[:,3,3]
    TriTable['c15'] = X_cov[:,0,4]; TriTable['c45'] = X_cov[:,3,4]
    TriTable['c16'] = X_cov[:,0,5]; TriTable['c46'] = X_cov[:,3,5]
    TriTable['c17'] = X_cov[:,0,6]; TriTable['c47'] = X_cov[:,3,6]
    TriTable['c18'] = X_cov[:,0,7]; TriTable['c48'] = X_cov[:,3,7]
    TriTable['c22'] = X_cov[:,1,1]; TriTable['c55'] = X_cov[:,4,4]
    TriTable['c23'] = X_cov[:,1,2]; TriTable['c56'] = X_cov[:,4,5]
    TriTable['c24'] = X_cov[:,1,3]; TriTable['c57'] = X_cov[:,4,6]
    TriTable['c25'] = X_cov[:,1,4]; TriTable['c58'] = X_cov[:,4,7]
    TriTable['c26'] = X_cov[:,1,5]; TriTable['c66'] = X_cov[:,5,5]
    TriTable['c27'] = X_cov[:,1,6]; TriTable['c67'] = X_cov[:,5,6]
    TriTable['c28'] = X_cov[:,1,7]; TriTable['c68'] = X_cov[:,5,7]
    TriTable['c33'] = X_cov[:,2,2]; TriTable['c77'] = X_cov[:,6,6]
    TriTable['c34'] = X_cov[:,2,3]; TriTable['c78'] = X_cov[:,6,7]
    TriTable['c35'] = X_cov[:,2,4]; TriTable['c88'] = X_cov[:,7,7]

    # Add some metadata
    TriTable.meta['triangulation_method'] = tri_method
    TriTable.meta['triangulation_frame'] = 'ECI'
    TriTable.meta['triangulation_t0'] = Time(t0, format='jd', scale='utc').isot

    # Create the radiant dictonary with radiant angular arguments
    TriTable.meta.update(radiant2dict(-Vel_ECI, X_cov[:,3:6,3:6], T_jd, Pos_ECI, True, True))

    for i, file in enumerate(file_list):
        TriTable.meta['triangulation_file_' + str(i)] = os.path.basename(file)
    TriTable.meta['telescope'] = 'multiple_telescopes_0'

    from astropy import __version__ as ap_version
    from numpy import __version__ as np_version
    from scipy import __version__ as sp_version
    TriTable.meta['astropy_version'] = ap_version
    TriTable.meta['numpy_version'] = np_version
    TriTable.meta['scipy_version'] = sp_version

    PointDir = os.path.dirname(file_list[0])
    event_codename = data.meta['event_codename']
    TriTable.meta['event_codename'] = event_codename
    ofile = os.path.join(PointDir, event_codename+'_triangulation_all_timesteps.ecsv')#'+'_'.join(cam_numbers)+'.ecsv')

    TriTable.sort(['datetime'])
    TriTable.write( ofile, format='ascii.ecsv', delimiter=',')

    logger.info('Output has been written to: ' + ofile)

    # Save to individual camera trajectories ======================================
    for obs in np.unique(LOS_errors[0,:]):

        T_rel_cam = LOS_errors[1,LOS_errors[0,:]==obs]
        cam_index = [np.where(T_rel==t)[0][0] for t in T_rel_cam]
        CamTable = CameraTables[int(obs)]

        # Add the triangulation columns
        CamTable['datetime'] = Time(t0 + T_rel_cam/(24*60*60), format='jd', scale='utc').isot
        CamTable['t_rel'] = T_rel_cam*u.second
        CamTable['latitude'] = np.rad2deg(Pos_LLH[0][cam_index])*u.deg
        CamTable['longitude'] = np.rad2deg(Pos_LLH[1][cam_index])*u.deg
        CamTable['height'] = Pos_LLH[2][cam_index]*u.m
        CamTable['X_geo'] = Pos_ECEF[0][cam_index]*u.m
        CamTable['Y_geo'] = Pos_ECEF[1][cam_index]*u.m
        CamTable['Z_geo'] = Pos_ECEF[2][cam_index]*u.m
        CamTable['X_eci'] = Pos_ECI[0][cam_index]*u.m
        CamTable['Y_eci'] = Pos_ECI[1][cam_index]*u.m
        CamTable['Z_eci'] = Pos_ECI[2][cam_index]*u.m
        CamTable['pos_err'] = pos_eci_err[cam_index]*u.m
        CamTable['D_DT_geo'] = vel_geo[cam_index]*u.m/u.second
        CamTable['D_DT_eci'] = vel_eci[cam_index]*u.m/u.second
        CamTable['D_DT_err'] = vel_eci_err[cam_index]*u.m/u.second

        obs_res = LOS_errors[:,LOS_errors[0] == obs] 
        CamTable['along_track_error'] = obs_res[3] * obs_res[2] * u.m
        CamTable['cross_track_error'] = obs_res[5] * obs_res[2] * u.m
        CamTable['range'] = obs_res[2] * u.m

        # Remove some columns
        CamTable.remove_columns(['obs_lat', 'obs_lon', 'obs_hei'])

        # Add the metadata
        CamTable.meta['triangulation_method'] = tri_method
        CamTable.meta['triangulation_frame'] = 'ECI'
        CamTable.meta['triangulation_t0'] = Time(t0, format='jd', scale='utc').isot
        CamTable.meta.update(radiant2dict(-Vel_ECI[:,cam_index], 
            X_cov[cam_index,3:6,3:6], T_jd[cam_index], Pos_ECI[:,cam_index], True, True))

        for i, file in enumerate(file_list):
            CamTable.meta['triangulation_file_' + str(i)] = os.path.basename(file)

        CamTable.meta['astropy_version'] = ap_version
        CamTable.meta['numpy_version'] = np_version
        CamTable.meta['scipy_version'] = sp_version

        # Write the results to file
        CamTable.sort(['datetime'])
        CamTable.write(file_list[int(obs)], format='ascii.ecsv', delimiter=',', overwrite=True)
        logger.debug('Table updated: ' + file_list[int(obs)])

    # Plot cross-track errors =====================================================
    colour = plt.cm.hsv(np.linspace(0,1,len(file_list)+1))

    fig = plt.figure(figsize=(16,9))
    ax1 = plt.subplot(2,1,1); plt.grid()
    plt.xlabel('Relative time [s]'); plt.ylabel('Cross Track Error [arcmin]')
    ax2 = plt.subplot(2,1,2); plt.grid()
    plt.xlabel('Relative time [s]'); plt.ylabel('Cross Track Error [meters]')
    for obs in np.unique(LOS_errors[0]):
        obs_res = LOS_errors[:,LOS_errors[0] == obs]; cam_c = colour[int(obs)]; 
        cam_l = '{0:s} [{1:.0f}-{2:.0f}km]'.format(
            cam_names[int(obs)], obs_res[2,0]/1000, obs_res[2,-1]/1000)

        ax1.errorbar(obs_res[1], np.rad2deg(obs_res[5])*60, fmt='-o',
            yerr=np.rad2deg(obs_res[6])*60, c=cam_c, label=cam_l)
        ax2.errorbar(obs_res[1], obs_res[5] * obs_res[2], fmt='-o',
            yerr=obs_res[6] * obs_res[2], c=cam_c, label=cam_l)

    plt.legend(loc=0)

    ofile_pic1 = os.path.join(PointDir, event_codename+'_cross_track_errors.png')
    plt.savefig(ofile_pic1, format='png')

    logger.debug('Cross-track errors have been saved to: ' + ofile_pic1)

    # # Plot along-track errors ======================================================
    fig = plt.figure(figsize=(16,9))
    ax1 = plt.subplot(2,1,1); plt.grid()
    plt.xlabel('Relative time [s]'); plt.ylabel('Along Track Error [arcmin]')
    ax2 = plt.subplot(2,1,2); plt.grid()
    plt.xlabel('Relative time [s]'); plt.ylabel('Along Track Error [meters]')
    for obs in np.unique(LOS_errors[0]):
        obs_res = LOS_errors[:,LOS_errors[0] == obs]; cam_c = colour[int(obs)]; 
        cam_l = '{0:s} [{1:.0f}-{2:.0f}km]'.format(
            cam_names[int(obs)], obs_res[2,0]/1000, obs_res[2,-1]/1000)

        ax1.errorbar(obs_res[1], np.rad2deg(obs_res[3])*60, fmt='-o',
            yerr=np.rad2deg(obs_res[4])*60, c=cam_c, label=cam_l)
        ax2.errorbar(obs_res[1], obs_res[3] * obs_res[2], fmt='-o',
            yerr=obs_res[4] * obs_res[2], c=cam_c, label=cam_l)

    plt.legend(loc=0)

    ofile_pic2 = os.path.join(PointDir, event_codename+'_along_track_errors.png')
    plt.savefig(ofile_pic2, format='png')

    logger.debug('Along-track errors have been saved to: ' + ofile_pic2)

    # Plot velocities (Second order) ===============================================
    fig = plt.figure(figsize=(16,9)); plt.grid()
    
    if tri_method == 'UKF' or tri_method == 'IMM':
        plt.errorbar(T_rel, vel_filter, yerr=vel_filter_err, fmt='-o', c='0.5')

    plt.errorbar(T_rel, vel_eci, yerr=vel_eci_err, fmt='-o', c='k')
    plt.plot(T_rel[0], vel_eci[0], '.k', label=r'$v_0$ = '
        +str(int(round(vel_eci[0])))+r'$\pm$'+str(int(round(vel_eci_err[0])))+' m/s')
    plt.plot(T_rel[-1], vel_eci[-1], '.k', label=r'$v_f$ = '
        +str(int(round(vel_eci[-1])))+r'$\pm$'+str(int(round(vel_eci_err[-1])))+' m/s')

    # Plot the individual camera's velocities determined by slls
    # index_timing = np.where(timing==1)[0]
    t_rel_slls = LOS_errors[1, timing] 
    t_slls = t0 + t_rel_slls / (24*60*60); 
    obs_slls = LOS_errors[0, timing]

    Pos_ECEF_slls = slls_args[0][:,timing] 
    Pos_ECI_slls = ECEF2ECI_pos(Pos_ECEF_slls, t_slls)
    for obs in np.unique(obs_slls):
        t_cam = t_rel_slls[obs_slls == obs]
        Pos_ECI_cam = Pos_ECI_slls[:, obs_slls==obs]
        Vel_ECI_cam = (Pos_ECI_cam[:,2:] - Pos_ECI_cam[:,:-2]) / (t_cam[2:] - t_cam[:-2])

        cam_c = colour[int(obs)]; cam_l = cam_names[int(obs)]
        plt.plot(t_cam[1:-1], norm(Vel_ECI_cam, axis=0), '.-', c=cam_c, label=cam_l)

    plt.xlabel('Relative time [s]')#\nNote: The errors are 10x actual for visualisation purposes'); 
    plt.ylabel('ECI Velocity Magnitude [m/s]'); plt.legend(loc=0)

    ofile_pic3 = os.path.join(PointDir, event_codename+'_modelled_velocity.png')
    plt.savefig(ofile_pic3, format='png')

    logger.debug('Modelled velocity has been saved to: ' + ofile_pic3)

    # Plot beta ====================================================================
    fig = plt.figure(figsize=(16,9)); plt.grid()

    if tri_method == 'UKF' or tri_method == 'IMM':
        plt.errorbar(T_rel, beta_filter, yerr=beta_filter_err, fmt='-o', c='0.5')
    
    plt.errorbar(T_rel, beta, yerr=beta_err, fmt='-o', c='k')
    plt.plot(T_rel[0], beta[0], '.k', label=r'$\beta_0$ = '
        +str(int(round(beta[0])))+r'$\pm$'+str(int(round(beta_err[0])))+r' $kg/m^2$')
    plt.plot(T_rel[-1], beta[-1], '.k', label=r'$\beta_f$ = '
        +str(int(round(beta[-1])))+r'$\pm$'+str(int(round(beta_err[-1])))+r' $kg/m^2$')
    plt.xlabel('Relative time [s]'); plt.ylabel(r'$\beta$ [kg/m$^2$]')

    ofile_pic4a = os.path.join(PointDir, event_codename+'_modelled_beta.png')
    plt.legend(loc=0); plt.savefig(ofile_pic4a, format='png')

    logger.debug('Modelled beta has been saved to: ' + ofile_pic4a)

    # Plot mass estimation =========================================================
    fig = plt.figure(figsize=(16,9)); plt.grid()

    if tri_method == 'UKF' or tri_method == 'IMM':
        plt.errorbar(T_rel, mass_filter, yerr=mass_filter_err, fmt='-o', c='0.5')
    
    plt.errorbar(T_rel, mass_est, yerr=mass_err, fmt='-o', c='k')
    plt.plot(T_rel[0], mass_est[0], '.k', label=r'$mass_0$ = '
        + r'{0:.3f} $\pm$ {1:.3f} kg'.format(mass_est[0], mass_err[0]))
    plt.plot(T_rel[-1], mass_est[-1], '.k', label=r'$mass_f$ = '
        + r'{0:.3f} $\pm$ {1:.3f} kg'.format(mass_est[-1], mass_err[-1]))
    plt.xlabel('Relative time [s]'); plt.ylabel(r'$mass_{est}$ [kg]'
        +'\n(Assuming a sphere of '+r'3500$kg/m^3$ density)')

    ofile_pic4b = os.path.join(PointDir, event_codename+'_modelled_mass_est.png')
    plt.legend(loc=0); plt.savefig(ofile_pic4b, format='png')

    logger.debug('Modelled mass has been saved to: ' + ofile_pic4b)

    # Plot SLLS timing =============================================================
    fig = plt.figure(figsize=(16,9)); plt.grid()
    plt.plot(T_rel, Pos_LLH[2]/1000, '-o', c='k')
    plt.plot(T_rel[0], Pos_LLH[2,0]/1000, '.k', label=r'$H_0$ = '
        +str(round(Pos_LLH[2,0]/1000,3))+' km')
    plt.plot(T_rel[-1], Pos_LLH[2,-1]/1000, '.k', label=r'$H_f$ = '
        +str(round(Pos_LLH[2,-1]/1000,3))+' km')

    # Plot the individual camera's timing determined by slls
    height_cam = ECEF2LLH(Pos_ECEF_slls)[2]
    for obs in np.unique(obs_slls):
        cam_c = colour[int(obs)]; cam_l = cam_names[int(obs)]
        plt.plot(t_rel_slls[obs_slls == obs], height_cam[obs_slls == obs]/1000, 
            '.-', c=cam_c, label=cam_l)

    plt.xlabel('Relative time [s]'); plt.legend(loc=0)
    plt.ylabel('Straight Line Least Squares Height [km]')

    ofile_pic5 = os.path.join(PointDir, event_codename+'_SLLS_timing_consistancy_check.png')
    plt.savefig(ofile_pic5, format='png')

    logger.debug('Timing consistancy check has been saved to: ' + ofile_pic5)

    # Plot position lags =============================================================
    fig = plt.figure(figsize=(16,9)); plt.grid()
    pos_lag = norm(Pos_ECI - Pos_ECI[:,:1], axis=0) - vel_eci[0] * (T_rel - T_rel[0])
    
    # if tri_method == 'UKF' or tri_method == 'IMM':
    #     plt.errorbar(T_rel, vel_filter, yerr=vel_filter_err, fmt='-o', c='0.5')

    plt.plot(T_rel, pos_lag/1000, '-o', c='k')
    plt.plot(T_rel[0], pos_lag[0]/1000, '.k', label=r'$lag_0$ = '
        +str(round(pos_lag[0]/1000,3))+' km')
    plt.plot(T_rel[-1], pos_lag[-1]/1000, '.k', label=r'$lag_f$ = '
        +str(round(pos_lag[-1]/1000,3))+' km')

    # Plot the individual camera's timing determined by slls
    for obs in np.unique(obs_slls):

        t_cam = t_rel_slls[obs_slls == obs]
        Pos_ECI_cam = Pos_ECI_slls[:, obs_slls==obs]
        pos_lag_slls = norm(Pos_ECI_cam - Pos_ECI[:,:1], axis=0) - vel_eci[0] * (t_cam - T_rel[0]) 

        cam_c = colour[int(obs)]; cam_l = cam_names[int(obs)]
        plt.plot(t_cam, pos_lag_slls/1000, '.-', c=cam_c, label=cam_l)

    plt.xlabel('Relative time [s]'); plt.legend(loc=0)
    plt.ylabel('Position Lag assuming constant velocity [km]'
        +'\n(...and a straight line)')

    ofile_pic6 = os.path.join(PointDir, event_codename+'_position_lag_check.png')
    plt.savefig(ofile_pic6, format='png')

    logger.debug('Position lag has been saved to: ' + ofile_pic6)

    # Create the KML's =============================================================
    if kwargs['create_kmls']: 
        path_dict = Path(ofile)
        path_dict['camera'] = 'Brightflight_trajectory'
        points_dict = Points(ofile, colour='ff1400ff') # red points
        points_dict['camera'] = 'Brightflight_trajectory'
        kml_files = [points_dict, path_dict]

        for file in file_list:
            kml_files.append( Rays(file) )

        # Merge KMLs into KMZ
        out_kmz = os.path.join(PointDir, event_codename + '_trajectory.kmz')
        merge_trajectory_KMLs(kml_files, out_kmz)

    # ==============================================================================

import glob
if __name__ == '__main__':

    parser = argparse.ArgumentParser(description='Modelled Triangulation of Meteors')
    parser.add_argument("-d", "--eventdirectory", type=str,
            help="The event directory for images with extension .ecsv" )
    parser.add_argument("-o", "--overwrite", action="store_true", default=False, 
            help="use this option if you want to allow a second trajectory run on the same day")
    args = parser.parse_args()

    if args.eventdirectory and os.path.isdir(args.eventdirectory):
        eventdir = args.eventdirectory
    else:
        raise FileNotFoundError('Event does not exist')
    
    # Create the file list for the dtf algorithm
    file_list = glob.glob(os.path.join(eventdir,"*.ecsv"))

    # Run the LS code
    kwargs = {'frag_times' : [],
              'create_kmls' : True}
    main(file_list, kwargs)
