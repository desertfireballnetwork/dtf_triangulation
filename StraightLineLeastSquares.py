"""
=============== Straight Line Least Squares ===============
Created on Mon Jun 29 15:13:05 2015
@author: Trent Jansen-Sturgeon

Straight line least squares estimate (or the Borovicka method).
Inputs: Two or more alt/azi files for triangulation.
Output: The triangulated data file in the form of time, LLH, ECEF.

"""

#import modules
import multiprocessing
import datetime
import os
import logging
import time

# import science modules
import numpy as np
from numpy.linalg import inv, norm
from scipy.linalg import block_diag
from astropy.table import Table, Column, hstack
from astropy.time import Time
from astropy import units as u

# import custom modules
import OptimalNormalFit as ONF
import trajectory_utilities as tu

__author__ = "Trent Jansen-Sturgeon"
__copyright__ = "Copyright 2017, Desert Fireball Network"
__license__ = ""
__version__ = "2.3"
__scriptName__ = "StraightLineLeastSquares.py"


def f(state, args):
    '''
    The azimuth and elevation assuming the meteor trajectory of x.
    Input: The state variable, x=[px,py,vx,vy], where pz = const and |v|=1.
    Output: The estimated azimuth and elevation for all data points from every 
    camera along the meteor's trajectory, y=dim(2*#cam*#t,1).
    '''

    # Reapply the scaling factor
    [UV_obs, ang_err, Camera_pos, Times_jd, Times_err, 
        NumberTimePts, pz, v_loc, v_sign] = args

    # Define the start of the trajectory and the radiant
    pos0 = np.vstack((state[0], state[1], pz)) #[3,1]

    v_mag = np.sqrt(1 - state[2]**2 - state[3]**2)
    UV_rad = np.hstack((state[2:v_loc+2],v_sign*v_mag,
        state[v_loc+2:])).reshape((3,1)) #[3,1]

    # Normal vector to UV_obs and UV_est
    n = np.cross(np.cross(UV_obs, UV_rad, axis=0), UV_obs, axis=0) #[3,n]

    # Distance along radient to get to pos
    t = np.sum((Camera_pos - pos0) * n, axis=0) / np.sum(UV_rad * n, axis=0) #[n]
    
    # Define ALL the positions of the fireball in ECEF
    Pos_rel = (pos0 + t * UV_rad) - Camera_pos #[3,n]

    return Pos_rel

def residuals(state, args):

    [UV_obs, ang_err, v_loc, v_sign] = [args[0], args[1], args[7], args[8]]
    Pos_est = f(state, args)
    UV_est = Pos_est / norm(Pos_est, axis=0)

    v_mag = np.sqrt(1 - state[2]**2 - state[3]**2)
    UV_rad = np.hstack((state[2:v_loc+2],v_sign*v_mag,
        state[v_loc+2:])).reshape((3,1)) #[3,1]

    n = np.cross(UV_obs, UV_est, axis=0) #[3,n]
    ang_sep = np.arctan2(norm(n, axis=0), 
        np.sum(UV_obs * UV_est, axis=0)) #[n]
    sign = n.T.dot(UV_rad) / np.abs(n.T.dot(UV_rad)) #[n]
    dir_ang_sep = sign.flatten() * ang_sep

    return dir_ang_sep / ang_err


# def parallelJacobian(ARGS):
#     ''' Evaluates one column of the Jacobian '''
#     i, epsilon, args, x = ARGS

#     # Unit vector in x_i direction
#     e = np.zeros((len(x), 1))
#     e[i] = 1

#     # Generate the first derivatives
#     f_pos = f(x + epsilon * e, args)
#     f_neg = f(x - epsilon * e, args)

#     return (f_pos - f_neg) / (2 * epsilon[i])


# def df_dx(x, args):
#     '''
#     The Jacobian as calculated using numerical central differencing.
#     Inputs: Current state estimate, camera locations and #points per camera.
#     Outputs: The Jacobian used within the least squares iterations.
#     '''
        
#     # Perturbation step lengths [1m for positions, 0.1deg for angles]
#     epsilon = np.ones((len(x), 1))
#     epsilon[3:5] = np.deg2rad(0.1) * np.ones((2,1))

#     Jacobian = np.zeros(((len(x) - 4) * 2, len(x)))

#     jac_column_pool = multiprocessing.Pool()
#     result_arr = jac_column_pool.map(parallelJacobian,
#                                      [(i, epsilon, args, x)
#                                       for i in range(len(x))],
#                                      chunksize=10)
#     jac_column_pool.close()
#     jac_column_pool.join()

#     for i in range(len(result_arr)):
#         Jacobian[:, i:i + 1] = result_arr[i]

#     return Jacobian


# def df_dx_NonParallel(x, args):
#     '''
#     The Jacobian as calculated using numerical central differencing.
#     Inputs: Current state estimate, camera locations and #points per camera.
#     Outputs: The Jacobian used within the least squares iterations.
#     '''
        
#     # Perturbation step lengths [1m for positions, 0.1deg for angles]
#     epsilon = np.ones(len(x))
#     epsilon[3:5] = np.deg2rad(0.1) * np.ones(2)

#     Jacobian = np.zeros(((len(x) - 4) * 2, len(x)))
#     for i in range(len(x)):
#         # Unit vector in x_i direction
#         e = np.zeros((len(x), 1))
#         e[i] = 1

#         # Generate the first derivatives
#         f_pos = f(x + epsilon * e, args)
#         f_neg = f(x - epsilon * e, args)
#         Jacobian[:, i:i + 1] = (f_pos - f_neg) / (2 * epsilon[i])

#     return Jacobian

from scipy.optimize import leastsq, least_squares
def SLLS(state0, args, eci_bool=False):
    """experimental version using scipy optimise instead of custom equations"""

    [ang_err, Camera_pos, Times_jd, Times_err, NumberTimePts] = args[1:6]
    
    # Create the first approximation of x, where X0=[x1,y1,z1,ra_geo,dec_geo,L2,L3,...,Ln]
    # Include a scaling factor, x_SF, so the NLLS can handle x.
    state_SF = np.ones(4); state_SF[:2] *= 1e-4
    
    t = time.time()
    print('Normalised residuals = %.3f deg' % 
        np.rad2deg(norm(residuals(state0, args) * ang_err)))
    results = leastsq(residuals, state0, diag=state_SF, args=args, full_output=True)
    
    state = results[0]

    f_eval = results[2]['nfev']
    jac_f = results[2]['fjac'].T
    res_f = ang_err * results[2]['fvec']

    # # Don't trust the errors produced by leastsq()- estimate using **
    # cov_reduced = results[1]
    # reduced_chi_sq = np.sum(res_vector**2) / (len(res_vector) - len(state))
    # cov = cov_reduced * reduced_chi_sq; cov_diag = np.diag(cov)

    # state_lb = np.array([state0[0]-1e4, state0[1]-1e4,-1,-1])
    # state_ub = np.array([state0[0]+1e4, state0[1]+1e4, 1, 1])
    # result = least_squares(residuals, state0, bounds=(state_lb, state_ub),
    #     x_scale='jac', args=(args,), method='trf', verbose=2)

    # state = result.x
    # res_f = result.fun * ang_err
    # f_eval = result.nfev
    # jac_f = result.jac
    
    # Estimate of the covariance matrix **
    # https://au.mathworks.com/help/stats/nlinfit.html?requestedDomain=www.mathworks.com
    mse = res_f.T.dot(res_f) / (len(res_f) - len(state))
    cov = mse * np.linalg.inv( jac_f.T.dot(jac_f) )
    cov_diag = np.diag(cov)
    
    print('\nNormalised residuals = %.3f deg' % np.rad2deg(norm(res_f)))
    print('Function Evaluations = %d' % f_eval)
    print('Calculation time (s) = %.2f\n' % (time.time() - t))
    
    # print('px_error = ', np.sqrt(cov_diag[0]), 'm')
    # print('py_error = ', np.sqrt(cov_diag[1]), 'm')
    # print('UV_x_error = ', np.sqrt(cov_diag[2]))
    # print('UV_y_error = ', np.sqrt(cov_diag[3]))
    
    # Uncode the state vector, x, into ECEF positions, where
    # Position@Tj=Position@T0+(radiant vector)*(length)
    Pos = f(state, args) + Camera_pos
    [v_loc, v_sign] = args[7:]
    v_mag = np.sqrt(1 - state[2]**2 - state[3]**2)
    UV_rad = -np.hstack((state[2:v_loc+2],v_sign*v_mag,
        state[v_loc+2:])).reshape((3,1)) #[3,1]

    # Calculate the radiant errors
    v_err = np.array([np.sqrt((state[2]/v_mag)**2 * cov_diag[2] 
                  + (state[3]/v_mag)**2 * cov_diag[3])])
    UV_err = np.hstack((np.sqrt(cov_diag[2:v_loc+2]),
                    v_err, np.sqrt(cov_diag[v_loc+2:])))
    UV_cov = [np.diag(UV_err**2)]

    radiant_dic = {'triangulation_method': 'SLLS'}
    if eci_bool:
        radiant_dic['triangulation_frame'] = 'ECI'
    else:
        radiant_dic['triangulation_frame'] = 'ECEF'

    # Generate the radiant metadata
    radiant_dic.update( generate_radiant_dictionary(UV_rad, 
        UV_cov, Pos, Times_jd, Times_err, eci_bool) )

    # Distances to the triangulated points
    dist = np.zeros(sum(NumberTimePts))
    for cam, N_tp in enumerate(NumberTimePts):
        
        # Number of previous data points
        N_i = sum(NumberTimePts[:cam]); N_f = N_i + N_tp
        dist[N_i:N_f] = norm(Pos[:, N_i:N_f] - Camera_pos[:, N_i:N_f], axis=0)
    
    # Cross track error
    CTE = res_f * dist

    logger = logging.getLogger('trajectory')
    logger.info('Normalised residuals = %6.4e deg' % np.rad2deg(norm(res_f)))
    # if norm(res_f) > 0.1:
    #     logger.info('Normalised residuals > 0.1 degrees; skipping')
    #     raise tu.PoorTriangulationResiduals

    return Pos, CTE, radiant_dic

def generate_radiant_dictionary(UV_rad, UV_cov, Pos, Times_jd, Times_err, eci_bool=False):

    # Update the radiant dictonary with radiant angular arguments
    # index = [np.argmin(Times_jd[Times_err < 1]), np.argmax(Times_jd[Times_err < 1])]
    # t_rad = Times_jd[Times_err < 1][index]; pos_rad = Pos[:,Times_err < 1][:,index]
    sorted_args = np.argsort(Times_jd[Times_err < 1])
    T_rad = Times_jd[Times_err < 1][sorted_args]
    if len(T_rad) != 0:
        Pos_rad = Pos[:,Times_err < 1][:,sorted_args]
        radiant_dic = radiant2dict(UV_rad, UV_cov, T_rad, Pos_rad, eci_bool)
    else:
        print('\nThere is no reliable timing with this event,'
            + 'but we can still do a geometric fit...')
        radiant_dic = radiant2dict_notiming(UV_rad, UV_cov, Pos)

    return radiant_dic

def radiant2dict(UV_rad, UV_cov, T_jd, Pos, eci_bool=False, ls_method=False):

    radiant_dic = {} # Create the radiant dictionary
    for i, suffix in enumerate(['inf', 'end']):

        uv_rad = np.vstack((UV_rad[:,-i])); uv_cov = UV_cov[-i]

        # Convert the radiant from cartesian to pointing angles
        angles = radiant_conversion(uv_rad, T_jd, Pos, i, eci_bool, ls_method) #[1,6]
        angles_jac = radiant_conversion_jac(uv_rad, T_jd, Pos, i, eci_bool, ls_method) #[3,6]
        angles_cov = angles_jac.T.dot(uv_cov).dot(angles_jac) #[6,6]
        angles_err = np.sqrt(np.diag(angles_cov)) #[6]

        [[ra_eci, dec_eci, ra_ecef, dec_ecef, slope, azimuth]] = angles
        [ra_eci_err, dec_eci_err, ra_ecef_err, 
        dec_ecef_err, slope_err, azimuth_err] = angles_err

        # The errors cannot be calculated through a frame change
        if eci_bool:
            ra_ecef_err = ra_eci_err
            dec_ecef_err = dec_eci_err
        else:
            ra_eci_err = ra_ecef_err
            dec_eci_err = dec_ecef_err

        radiant_dic['triangulation_radiant_'+suffix+'_time'] = \
            Time(T_jd[-i], format='jd', scale='utc').isot

        radiant_dic['triangulation_ra_eci_'+suffix] = np.rad2deg(ra_eci)
        radiant_dic['triangulation_ra_eci_'+suffix+'_err'] = np.rad2deg(ra_eci_err)
        radiant_dic['triangulation_dec_eci_'+suffix] = np.rad2deg(dec_eci)
        radiant_dic['triangulation_dec_eci_'+suffix+'_err'] = np.rad2deg(dec_eci_err)

        radiant_dic['triangulation_ra_ecef_'+suffix] = np.rad2deg(ra_ecef)
        radiant_dic['triangulation_ra_ecef_'+suffix+'_err'] = np.rad2deg(ra_ecef_err)
        radiant_dic['triangulation_dec_ecef_'+suffix] = np.rad2deg(dec_ecef)
        radiant_dic['triangulation_dec_ecef_'+suffix+'_err'] = np.rad2deg(dec_ecef_err)

        radiant_dic['triangulation_slope_'+suffix] = np.rad2deg(slope)
        radiant_dic['triangulation_slope_'+suffix+'_err'] = np.rad2deg(slope_err)
        radiant_dic['triangulation_azimuth_'+suffix] = np.rad2deg(azimuth)
        radiant_dic['triangulation_azimuth_'+suffix+'_err'] = np.rad2deg(azimuth_err)

    return radiant_dic

def radiant_conversion(uv_rad, T_jd, Pos, idx, eci_bool=False, ls_method=False):

    # Create a time mask with all points within 0.5sec of t0
    time_mask = (np.abs(T_jd[idx] - T_jd) < 0.5 / (24*60*60)) #[n]

    if sum(time_mask) == 1:
        print("\nSorry, but it looks like you've got one badly picked point somewhere...\n")

    if ls_method:
        UV_rad_eci = uv_rad #[3,1]
        UV_rad_ecef = -tu.ECI2ECEF(np.vstack((Pos[:,-idx])), 
                    -np.vstack((uv_rad)), T_jd[-idx])[1] #[3,1]
    elif eci_bool:
        UV_rad_eci = uv_rad #[3,1]
        Pos_ECEF = tu.ECI2ECEF_pos(Pos[:,time_mask], T_jd[time_mask]) #[3,n]
        UV_rad_ecef = fit_line_to_pos(Pos_ECEF) #[3,1]
    else:
        UV_rad_ecef = uv_rad #[3,1]
        Pos_ECI = tu.ECEF2ECI_pos(Pos[:,time_mask], T_jd[time_mask]) #[3,n]
        UV_rad_eci = fit_line_to_pos(Pos_ECI) #[3,1]

    # Calculate the ra/dec of the radiant
    ra_eci = np.arctan2(UV_rad_eci[1], UV_rad_eci[0]) #[1]
    dec_eci = np.arcsin(UV_rad_eci[2] / norm(UV_rad_eci)) #[1]
    ra_ecef = np.arctan2(UV_rad_ecef[1], UV_rad_ecef[0]) #[1]
    dec_ecef = np.arcsin(UV_rad_ecef[2] / norm(UV_rad_ecef)) #[1]

    # Calculate the slope and azimuth of the radiant
    [lat_i, lon_i] = tu.ECEF2LLH(np.vstack((Pos[:,idx])))[:2]
    UV_rad_enu = tu.ECEF2ENU(float(lon_i), float(lat_i)).dot(uv_rad) #[3,1]
    slope = np.arcsin(UV_rad_enu[2] / norm(UV_rad_enu)) #[1]
    azimuth = np.arctan2(UV_rad_enu[0], UV_rad_enu[1]) #[1]

    return np.vstack((ra_eci, dec_eci, ra_ecef, dec_ecef, slope, azimuth)).T #[1,6]


def radiant2dict_notiming(UV_rad, UV_cov, Pos):
    T_jd = 2451545.0 * np.ones(len(Pos[0])) #dummy

    radiant_dic = {} # Create the radiant dictionary
    for i, suffix in enumerate(['inf', 'end']):

        uv_rad = np.vstack((UV_rad[:,-i])); uv_cov = UV_cov[-i]

        # Convert the radiant from cartesian to pointing angles
        angles = radiant_conversion(uv_rad, T_jd, Pos, i) #[1,6]
        angles_jac = radiant_conversion_jac(uv_rad, T_jd, Pos, i) #[3,6]
        angles_cov = angles_jac.T.dot(uv_cov).dot(angles_jac) #[6,6]
        angles_err = np.sqrt(np.diag(angles_cov)) #[6]

        [[ra_eci, dec_eci, ra_ecef, dec_ecef, slope, azimuth]] = angles
        [ra_eci_err, dec_eci_err, ra_ecef_err, 
        dec_ecef_err, slope_err, azimuth_err] = angles_err

        radiant_dic['triangulation_ra_ecef_'+suffix] = np.rad2deg(ra_ecef)
        radiant_dic['triangulation_ra_ecef_'+suffix+'_err'] = np.rad2deg(ra_ecef_err)
        radiant_dic['triangulation_dec_ecef_'+suffix] = np.rad2deg(dec_ecef)
        radiant_dic['triangulation_dec_ecef_'+suffix+'_err'] = np.rad2deg(dec_ecef_err)

        radiant_dic['triangulation_slope_'+suffix] = np.rad2deg(slope)
        radiant_dic['triangulation_slope_'+suffix+'_err'] = np.rad2deg(slope_err)
        radiant_dic['triangulation_azimuth_'+suffix] = np.rad2deg(azimuth)
        radiant_dic['triangulation_azimuth_'+suffix+'_err'] = np.rad2deg(azimuth_err)

    return radiant_dic

def fit_line_to_pos(Pos):
    # There is probably a smarter way to do this...

    def perp_distance(x_line, Pos):

        ra = x_line[0]; dec = x_line[1]
        uv_line = np.vstack((np.cos(ra) * np.cos(dec),
            np.sin(ra) * np.cos(dec), np.sin(dec))) #[3,1]
        x0_line = np.vstack((x_line[2], x_line[3], Pos[2,0])) #[3,1]

        return norm(np.cross(x0_line - Pos, uv_line, axis=0), axis=0) / norm(uv_line)

    # Perform a least squares line fit to the given positions
    pos_rel = Pos[:,0] - Pos[:,-1] #[3]
    x_line_est = np.array([np.arctan2(pos_rel[1], pos_rel[0]),
        np.arcsin(pos_rel[2] / norm(pos_rel)), Pos[0,0], Pos[1,0]])
    x_line = leastsq(perp_distance, x_line_est, args=(Pos,))[0]

    # Convert to cartesian
    ra = x_line[0]; dec = x_line[1]
    uv_rad = np.vstack((np.cos(ra) * np.cos(dec),
        np.sin(ra) * np.cos(dec), np.sin(dec))) #[3,1]

    return uv_rad

def radiant_conversion_jac(uv_rad, t_jd, Pos, idx, eci_bool=False, ls_method=False):
    
    # Perturbation step lengths
    epsilon = 1e-6
    eps_diag = epsilon * np.eye(3)

    # Generate the first derivatives
    jac = np.zeros((3,6))
    for i, e in enumerate(eps_diag):
        ev = np.vstack((e))
        angles_pos = radiant_conversion(uv_rad + ev, t_jd, Pos, idx, eci_bool, ls_method) #[3,6]
        angles_neg = radiant_conversion(uv_rad - ev, t_jd, Pos, idx, eci_bool, ls_method) #[3,6]
        jac[i] = tu.angular_difference(angles_pos, angles_neg) / (2 * epsilon) #[3,6]

    return jac

def FirstApprox(*CamFiles, eci_bool=False):
    """Convert the camera files into tables"""
    NumberCams = len(CamFiles)
    astrometry_tables = []
    for a in range(NumberCams):
        astrometry_tables.append( Table.read(CamFiles[a],
                            format='ascii.ecsv', guess=False, delimiter=',') )
    return FirstApprox_using_tables( astrometry_tables, eci_bool)
    
def FirstApprox_using_tables( astrometry_tables, eci_bool=False):
    ''' Create the first approximate state vector, X0, where X0=[px,py,vx,vy]'''

    if eci_bool:
        print('Removing datapoints without accurate timing...')
        for i, table in reversed(list(enumerate(astrometry_tables))):
            timing_err = (table['time_err_minus'] + table['time_err_plus']) / 2
            astrometry_tables[i] = table[timing_err < 1]
            if len(astrometry_tables[i]) == 0: del astrometry_tables[i]
    
    # Extracting the camera location from table and determine best fit plane
    NumberCams = len(astrometry_tables)
    Cameras_LLH = np.zeros((3, NumberCams))
    Cameras_ECEF = np.zeros((3, NumberCams))
    n_ECEF = np.zeros((3, NumberCams))
    NumberTimePts = [0] * NumberCams
    UV_obs, ang_err, Camera_pos, Times_jd, Times_err = [], [], [], [], []
    for cam in range(NumberCams):
        
        # Extract the camera's position
        astrometry_table = astrometry_tables[cam]
        obs_lon = astrometry_table.meta['obs_longitude']
        obs_lat = astrometry_table.meta['obs_latitude']
        obs_ele = astrometry_table.meta['obs_elevation']
        
        # Define the camera locations
        Lat0 = np.deg2rad(float(obs_lat))  # Latitude [rad]
        Long0 = np.deg2rad(float(obs_lon))  # Longitude [rad]
        H0 = float(obs_ele)  # Height [m]
        Cameras_LLH[:, cam:cam + 1] = np.vstack((Lat0, Long0, H0))  # In radians
        Cameras_ECEF[:, cam:cam + 1] = tu.LLH2ECEF(Cameras_LLH[:, cam:cam + 1])

        # Fetch the az/el data from the table
        Az = np.deg2rad(astrometry_table['azimuth'] % 360)  # Always between 0 and 2pi
        El = np.deg2rad(astrometry_table['altitude'])
        Az_error = np.deg2rad(astrometry_table['err_plus_azimuth']) # FIXME error control
        El_error = np.deg2rad(astrometry_table['err_plus_altitude']) # FIXME error control
        NumberTimePts[cam] = len(Az)

        UV_enu = np.vstack((np.sin(Az) * np.cos(El),
                            np.cos(Az) * np.cos(El),
                            np.sin(El))) #[3,n]
        UV_ecef = tu.ENU2ECEF(Long0, Lat0).dot(UV_enu) #[3,n]

        # If errors are just a relative measure, this should be fine...
        ang_err.extend([np.sqrt(Az_error**2 + El_error**2)])

        T_isot = astrometry_table['datetime']
        T_jd = Time(T_isot, format='isot', scale='utc').jd
        Times_jd.extend([T_jd])
        T_err = np.array((astrometry_table['time_err_minus']
            +astrometry_table['time_err_plus']) / 2)
        Times_err.extend([T_err])

        if eci_bool:
            temp = Cameras_ECEF[:,cam:cam+1] * np.ones((3,len(T_jd)))
            Camera_pos.extend([tu.ECEF2ECI_pos(temp, T_jd)])
            
            UV_eci = tu.ECEF2ECI_pos(UV_ecef, T_jd)
            UV_obs.extend([UV_eci / norm(UV_eci, axis=0)]) #[3,n]
        else:
            Camera_pos.extend([Cameras_ECEF[:, cam:cam + 1]*np.ones((3,len(Az)))])
            UV_obs.extend([UV_ecef]) #[3,n]


        # Use MOP to get a plane best fit
        try:
            [n, n_ECEF[:,cam:cam+1]] = ONF.n_fit_using_tables(astrometry_table)
        except:
            print("There is a problem with ", astrometry_table.meta['telescope'])
            raise

    # Reshape the arrays into column vectors
    UV_obs = np.hstack(UV_obs); ang_err = np.hstack(ang_err)
    Camera_pos = np.hstack(Camera_pos); Times_jd = np.hstack((Times_jd))
    Times_err = np.hstack((Times_err))

    # Define the radiant direction
    if eci_bool:
        n1 = tu.ECEF2ECI_pos(n_ECEF[:,0:1], Times_jd[0])
        n2 = tu.ECEF2ECI_pos(n_ECEF[:,1:2], Times_jd[0])
    else:
        n1 = n_ECEF[:,0:1]; n2 = n_ECEF[:,1:2]
    V0 = np.cross(n1, n2, axis=0) / norm(np.cross(n1, n2, axis=0)) #[3,1]

    # Find the unit vector in optimised plane
    UV_opt = UV_obs[:,:1] - UV_obs[:,:1].T.dot(n1) * n1 #[3,1]
    UV_opt = UV_opt / norm(UV_opt)

    # Add the xyz positions to the state vector
    cam2 = Camera_pos[:,NumberTimePts[0]:NumberTimePts[0]+1]
    d = (cam2 - Camera_pos[:,:1]).T.dot(n2) / UV_opt.T.dot(n2)
    X0 = UV_opt * d + Camera_pos[:,:1] #[3,1]

    if X0.T.dot(V0) > 0:
        V0 = -V0

    pz = X0[2]; v_loc = np.argmax(np.abs(V0)); 
    v_sign = V0[v_loc] / np.abs(V0[v_loc])
    state0 = np.vstack((X0[:2], V0[:v_loc], V0[v_loc+1:])).flatten()
    
    args = [UV_obs, ang_err, Camera_pos, Times_jd, 
        Times_err, NumberTimePts, pz, v_loc, v_sign]
    return state0, args


def main(CamFiles, eci_bool=False):

    astrometry_tables = []
    for camfile in CamFiles:
        t = Table.read(camfile, format='ascii.ecsv')
        t.meta['self_disk_path'] = camfile
        astrometry_tables.append(t)

    # if eci_bool:
    #     timing_list = [dfn_utils.has_reliable_timing(at) for at in astrometry_tables]
    #     astrometry_tables = [at[t] for at, t in 
    #             zip(astrometry_tables, timing_list) if np.any(t)]

    NumberCams = len(astrometry_tables)

                                        
    astrometry_tables = main_using_tables( astrometry_tables, CamFiles, eci_bool)
    
    for t in astrometry_tables:
        t.write(t.meta['self_disk_path'], format='ascii.ecsv', delimiter=',', overwrite=True)

    return

def main_using_tables(astrometry_tables, CamFiles, eci_bool=False):
    """CamFiles is passed so that names can be written into the tables, not for 
    file i/o"""

    NumberCams = len(astrometry_tables)
    # Calculate a first radient approximation
    print('\nCalculating initial approximation using MOP...')
    [state0, args] = FirstApprox_using_tables( astrometry_tables, eci_bool)

    # Triangulate using the straight line least squares technique
    print('\nBeginning Straight Line Least Squares...')
    [Pos, CTE, radiant_dic] = SLLS(state0, args, eci_bool)
    print('\nAlgorithm Finished!')

    # Convert the ECEF into LLH coordinates
    if eci_bool:
        Pos_ECEF = tu.ECI2ECEF_pos(Pos, np.array(args[3]))
    else:
        Pos_ECEF = Pos
    Pos_LLH = tu.ECEF2LLH(Pos_ECEF)  # In radians
    Pos_LLH_deg = np.vstack((np.rad2deg(Pos_LLH[0:2]), Pos_LLH[2]))
    # In degrees and METERS!
    
    star_pos = 0
    new_astrometry_tables = []
    for cam, astrometry_table in enumerate(astrometry_tables):

        # Get the data
        NumberTimePts = len(astrometry_table)
#        print("astrometry table length: ", NumberTimePts)
        # Get the relevant index range in the points
        end_pos = star_pos + NumberTimePts
        #update the data columns
        fire_longitude_col = 'longitude'
        fire_latitude_col = 'latitude'
        fire_height_col = 'height'
        fire_x_col = 'X_geo'
        fire_y_col = 'Y_geo'
        fire_z_col = 'Z_geo'
        fire_x_eci_col = 'X_eci'
        fire_y_eci_col = 'Y_eci'
        fire_z_eci_col = 'Z_eci'
        fire_cte_col = 'cross_track_error'

        possible_newcols = [fire_longitude_col,
                            fire_latitude_col,
                            fire_height_col,
                            fire_x_col,
                            fire_y_col,
                            fire_z_col,
                            fire_x_eci_col,
                            fire_y_eci_col,
                            fire_z_eci_col,
                            fire_cte_col]
                            
        for c in possible_newcols:
            if c in astrometry_table.colnames:
                astrometry_table.remove_columns(c)

#        print("start and end pos", star_pos, "  ", end_pos)
#        print("number of points about to be written: ", end_pos - star_pos + 1)

        lon_col = Column(name=fire_longitude_col, 
                        data=Pos_LLH_deg[1, star_pos:end_pos], unit=u.deg)
        lat_col = Column(name=fire_latitude_col,
                        data=Pos_LLH_deg[0, star_pos:end_pos], unit=u.deg)
        hei_col = Column(name=fire_height_col,
                        data=Pos_LLH_deg[2, star_pos:end_pos], unit=u.meter)
        x_col = Column(name=fire_x_col,
                        data=Pos_ECEF[0, star_pos:end_pos], unit=u.meter)
        y_col = Column(name=fire_y_col, 
                        data=Pos_ECEF[1, star_pos:end_pos], unit=u.meter)
        z_col = Column(name=fire_z_col,
                        data=Pos_ECEF[2, star_pos:end_pos], unit=u.meter)            
        cte_col = Column(name=fire_cte_col,
                        data=CTE[star_pos:end_pos], unit=u.meter)

        # Create write-time string for session meta
        now_timestamp = datetime.datetime.now()
        writetime = now_timestamp.strftime('%Y-%m-%dT%H:%M:%S')
        sessionmeta = {'triangulation_write_time': writetime,
                       'triangulation_software': __scriptName__ + " " + __version__}
        
        sessionmeta.update(radiant_dic)
                    
        for c, i in zip(CamFiles, list(range(1, NumberCams + 1))):
            sessionmeta['triangulation_file_' + str(i)] = os.path.basename(c)

        # update table
        triangulation_table = Table(data=[lon_col, lat_col, hei_col, x_col, y_col, 
                                            z_col, cte_col], meta=sessionmeta)
        if eci_bool:
            triangulation_table[fire_x_eci_col] = Pos[0, star_pos:end_pos]*u.m
            triangulation_table[fire_y_eci_col] = Pos[1, star_pos:end_pos]*u.m
            triangulation_table[fire_z_eci_col] = Pos[2, star_pos:end_pos]*u.m

        astrometry_table = hstack([astrometry_table, triangulation_table],
                                  join_type='exact')
        print('Table updated: ' + CamFiles[cam] )
        star_pos += NumberTimePts
        new_astrometry_tables.append( astrometry_table)
        
    return new_astrometry_tables


def MartinExpt(*CamFiles):
    """experimental version that builds repeated triangulations on shorter and
    shorter datasets by dropping the last point. An attempt to highlight any
    breaks of slope that might be fragmentation events"""
    from shutil import copyfile

    FileSuffix = []
    NewFileNames = []
    for j in range(len(CamFiles)):
        FileSuffix.append('_Modified_' + CamFiles[j])
        NewFileNames.append(str(0).zfill(3) + FileSuffix[j])
        copyfile(CamFiles[j], NewFileNames[j])

    # Keep producing files until one camera reaches two time points
    Iter = 0
    num = [10, 10]
    while min(np.array(num)) >= 2:  # Check there is still enough info to triangulate

        # Run SLLS as usual: mb
        main(*NewFileNames)

        # Calculate the next max time
        mintime = []
        astrometry_table = []
        for j in range(len(NewFileNames)):
            astrometry_table.append( Table.read(NewFileNames[j],
                                     format='ascii.ecsv', guess=False, 
                                     delimiter=','))
            times = astrometry_table[j]['datetime']
            mintime.append(min(times))

        MinTime = min(mintime)

        # Delete all rows with max time and save
        num = []
        Iter = Iter + 1
        for j in range(len(NewFileNames)):
            newtable = astrometry_table[j][astrometry_table[j]['datetime'] > MinTime]
            newtable.write( str(Iter).zfill(3) + FileSuffix[j],
                            format='ascii.ecsv', delimiter=',')
            num.append(len(newtable['datetime']))
            NewFileNames[j] = str(Iter).zfill(3) + FileSuffix[j]

    for File in NewFileNames:
        os.remove(File)

