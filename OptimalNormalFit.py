"""
=============== Optimal Normal Fit ===============
Created on Mon Jun 29 16:27:36 2015
@author: Trent Jansen-Sturgeon
mct, 2018-05 add logging, remove prints
"""
from __future__ import absolute_import, division, print_function

import logging

import numpy as np
from numpy.linalg import inv, norm
from astropy.table import Table
from scipy.optimize import leastsq

def g(n, y_obs):
    '''    
    The azimuth and elevation assuming the plane with normal n.
    Input: The current normal estimate, n=dim(2,1).
    Output: The estimated azimuth and elevation in the plane for all data 
    points along the meteor's trajectory for a given camera, y=dim(2*#t,1).
    Note the normal and y_obs angles must be in the same coordinates.
    '''

    # Define a constant
    NumberTimePts = int(np.shape(y_obs)[0] / 2)

    y_est = np.zeros((NumberTimePts * 2, 1))
    for j in range(NumberTimePts):

        # Extract the measured angular values
        az = y_obs[2 * j]
        el = y_obs[2 * j + 1]

        # Find the unit vector given by these angles (in ENU coords)
        UV = np.vstack((np.cos(el) * np.sin(az),
                        np.cos(el) * np.cos(az),
                        np.sin(el)))

        n_cart = np.vstack((np.cos(n[1]) * np.sin(n[0]),
                            np.cos(n[1]) * np.cos(n[0]),
                            np.sin(n[1])))

        # Take the measured unit vector and project it to the plane
        UV_est = UV - np.dot(UV.T, n_cart) / (norm(n_cart)**2) * n_cart

        # Extract the angular information
        az_est = np.arctan2(UV_est[0], UV_est[1]) % (2 * np.pi)  # Always between 0 and 2pi
        el_est = np.arctan2(UV_est[2], np.sqrt(UV_est[0]**2 + UV_est[1]**2))

        # Construct the estimated line of sights
        y_est[2 * j:2 * j + 2] = np.vstack((az_est, el_est))

    return y_est


def dg_dn(n, y_obs):
    '''
    The Jacobian as calculated using numerical central differencing.
    Inputs: Current normal estimate and the observational angles, az and el.
    Outputs: The Jacobian used within the least squares iterations.

    '''

    # Setup some empty variables
    Jacobian = np.zeros((len(y_obs), len(n)))

    # Perturbation step lengths for the unit vector n
    epsilon = 0.01 * np.ones((len(n), 1))

    for i in range(len(n)):

        # Basis unit vector
        e = np.zeros((len(n), 1))
        e[i] = 1

        # Generate the first derivatives
        g_pos = g(n + epsilon * e, y_obs)
        g_neg = g(n - epsilon * e, y_obs)
        Jacobian[:, i:i + 1] = (g_pos - g_neg) / (2 * epsilon[i])

    return Jacobian
    
def residuals(n, y_obs, Sigma2):
    y_est = g(n, y_obs)
    return np.hstack((y_obs - y_est) / np.sqrt(Sigma2))

class TrajectoryGeometryError(Exception):

    def __init__(self, value):
        self.value = value

    def __str__(self):
        return repr(self.value)


def __init__(self, prev, next, msg):
    self.prev = prev
    self.next = next
    self.msg = msg


def n_fit(CamFile):
    '''
    Calculates the normal to the best fit plane for a particular camera.
    Inputs: One camera file that includes the camera's location and the
    observed meteor data in azimuth and elevation.
    Outputs: The optimal plane for the given data, expressed in a normal
    vector in both ENU angles, n, and ECEF vector, n_ECEF. '''
    
    astrometry_table = Table.read(CamFile, format='ascii.ecsv', guess=False, delimiter=',')
    
    return n_fit_using_tables( astrometry_table)
def n_fit_using_tables( astrometry_table):
    """copy of n_fit that works on a table, rather than a camfile    
    
    Calculates the normal to the best fit plane for a particular camera.
    Inputs: One camera file that includes the camera's location and the
    observed meteor data in azimuth and elevation.
    Outputs: The optimal plane for the given data, expressed in a normal
    vector in both ENU angles, n, and ECEF vector, n_ECEF."""

    logger = logging.getLogger()
    logger.debug( 'start n_fit using tables')
    # Extract the camera's name and position
    obs_lon = astrometry_table.meta['obs_longitude']
    obs_lat = astrometry_table.meta['obs_latitude']
    obs_location_name = astrometry_table.meta['telescope'] + " " + astrometry_table.meta['location']

    Lat0 = np.deg2rad(float(obs_lat))  # Latitude [rad]
    Long0 = np.deg2rad(float(obs_lon))  # Longitude [rad]

    # print astrometry_table

    # Fetch the az/el data from the file
    Az = np.deg2rad(astrometry_table['azimuth'] % 360)  # Always between 0 and 2pi
    El = np.deg2rad(astrometry_table['altitude'])
    Az_error = np.deg2rad(astrometry_table['err_plus_azimuth'])  # FIXME error control
    El_error = np.deg2rad(astrometry_table['err_plus_altitude'])  # FIXME error control

    # Construct a vector of the observed angular lines of sight(y_obs)
    NumberTimePts = np.shape(Az)[0]
    y_obs = np.zeros((NumberTimePts*2,1))
    Sigma2 = np.zeros((NumberTimePts*2,1))
    y_obs[::2] = np.vstack((Az)); y_obs[1::2] = np.vstack((El))
    Sigma2[::2] = np.vstack((Az_error**2)); Sigma2[1::2] = np.vstack((El_error**2))

    # Approximate the plane normal using the first and last lines of sight
    UV1_ENU = np.vstack((np.cos(El[0]) * np.sin(Az[0]),
                         np.cos(El[0]) * np.cos(Az[0]),
                         np.sin(El[0])))

    UV2_ENU = np.vstack((np.cos(El[-1]) * np.sin(Az[-1]),
                         np.cos(El[-1]) * np.cos(Az[-1]),
                         np.sin(El[-1])))

    # Calculate the cartesian normal to the plane (in ENU)
    n_cart = np.cross(UV1_ENU, UV2_ENU, axis=0)

    # Convert the cartesian coords to angular ones (effectively az/el)
    n0 = np.vstack((np.arctan2(n_cart[0], n_cart[1]),
                   np.arctan2(n_cart[2], np.sqrt(n_cart[0]**2 + n_cart[1]**2))))

    # Begin the NLLS process to improve the n approximation
    results = leastsq(residuals, n0, args=(y_obs, Sigma2), full_output=True)
    
    n = np.vstack(results[0])
#    f_eval = results[2]['nfev']
#    residual = norm(np.vstack(results[2]['fvec']) * np.sqrt(Sigma2))
#    print(('Normalised residuals = %6.4e' % residual))
#    print(('Function Evaluations = %d' % f_eval))
    logger.info( obs_location_name + ", plane is now optimised.")

    # Convert the normal angles back to cartesian form (in ENU)
    n_cart = np.vstack((np.cos(n[1]) * np.sin(n[0]),
                        np.cos(n[1]) * np.cos(n[0]),
                        np.sin(n[1])))

    # Compute the other transformation matrix
    C_ENU2ECEF = np.array([[-np.sin(Long0), -np.sin(Lat0) * np.cos(Long0), np.cos(Lat0) * np.cos(Long0)],
                           [np.cos(Long0), -np.sin(Lat0) * np.sin(Long0), np.cos(Lat0) * np.sin(Long0)],
                           [0, np.cos(Lat0), np.sin(Lat0)]])

    # Cocrdinate transform from ENU to ECEF and output
    n_ECEF = C_ENU2ECEF.dot(n_cart)
    return (n, n_ECEF)


def angle_vectors(v1, v2):
    '''
    Returns the angle in radians between vectors 'v1' and 'v2'
    '''
    cosang = np.dot(v1, v2)
    sinang = norm(np.cross(v1, v2))
    return np.arctan2(sinang, cosang)

def angle_between_planes(n_ECEF_1, n_ECEF_2):
    '''
    Calculates the unsigned angle between 2 vectors ([0,90] deg)
    '''
    
    # Compute the angle between the two normal vectors
    angle = np.degrees(angle_vectors(n_ECEF_1, n_ECEF_2))
    # mirror everything back to [0,90]
    angle = (angle + 180.0) % 180.0
    angle = 90.0 - abs(angle - 90.0)
    
    return angle
