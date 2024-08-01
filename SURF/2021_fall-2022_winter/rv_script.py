import pickle
import radvel
import rebound as rb
import pandas as pd
import numpy as np
import matplotlib.pyplot as pl
from scipy import optimize, stats
import corner
import reboundx
import emcee
import h5py
from tqdm import tqdm
import os
import multiprocessing
from getdist import plots, MCSamples

MAY_1_2015 = 57143.5  # barycentric julian date for May 1, 2015 (the date of the HARPS instrument upgrade as per trifonov et al 2020)
# 57143.5 is BJD for May 1, 2015
# 57173.5 is BJD for May 31, 2015

fit_params = [2.27859008e+02, 7.20396587e+00,  5.39386707e+04, -7.17270858e-03, -2.13670237e-01,
              3.44028221e+02, 1.82216479e+01,  5.47055869e+04, 1.14530821e-01,  3.81765820e-02,
              -1.38087163e-01, -2.89290650e+00, 1.70788055e+00, 
              1.00000000e+00,
              2.15025156e+00, 1.48605174e+00, 4.42809302e+00] 

# harps
hd_data_harps = pd.read_csv('hd45364_rvs.csv', sep = ';')
# giant outlier at position 116 in the data (found manually earlier) which we remove
hd_data_harps.drop(116, inplace=True)  # drop the row and keep the df in place
# subtract 2.4e6 from all the rows in the data
hd_data_harps.BJD -= 2.4e6
# rename target to HARPS1 or HARPS2
hd_data_harps['target'] = hd_data_harps.apply(lambda row: 'HARPS1' if row.BJD < MAY_1_2015 else 'HARPS2', axis = 1)
# hires
hd_data_hires = pd.read_csv('../hires_rvs.txt', sep = '\t', index_col=False, header='infer', dtype=np.float64)
hd_data_hires['BJD - 2,450,000'] += 50000.  # adding 50000 to have the same units as harps
hd_data_hires['target'] = 'HIRES'
hd_data_hires.columns = ['BJD', 'RV_mlc_nzp', 'e_RV_mlc_nzp', 'target']
# concatenate two data sets one on top of the other
hd_data = pd.concat((hd_data_harps, hd_data_hires), axis=0)  # matching BJD, RV_mlc_nzp and e_RV_mlc_nzp columns
# reset index
hd_data.reset_index(drop=True, inplace=True)

## CONSTANTS:

STAR_MASS = 920  # 920 jupiter masses
G = 2.825e-7  # converting G to jupiter masses, au, and days
AUDAY_MS = 1.731e6  # conversion factor for au/day to m/s

obs_time_base = np.median(hd_data_harps.BJD)

# print(f'nbody_params:{nbody_params}\n fit_params:{fit_params}')

def mass_to_semiamp(planet_mass, star_mass, period, eccentricity, inclination):
    """
    planet mass (jupiter masses) to semi amplitude (in au/day)
    """
    return ((2 * np.pi * G/period) ** (1/3) * (planet_mass * np.sin(inclination) / star_mass ** (2/3)) * (1/np.sqrt(1 - eccentricity ** 2)))


def semiamp_to_mass(semiamp, star_mass, period, eccentricity, inclination):
    """
    semi amplitude (in au/day) to planet mass (jupiter masses)
    """
    return (((2 * np.pi * G/period) ** (-1/3)) * (semiamp / np.sin(inclination)) * np.sqrt(1 - eccentricity ** 2) * (star_mass ** (2/3)))


def get_sim_from_params(params, integrator, time_base, star_mass = STAR_MASS, auday_ms = AUDAY_MS):
    """
    takes in params array, returns a rebound Simulation object with those parameters
    
    param params: numpy array of params:
    
    for i in range(0, num_planets):
    
    params[i + 0] is period
    params[i + 1] is semiamp
    params[i + 2] is tc (time of conjunction)
    params[i + 3] is sqrt(e) * cos(omega)
    params[i + 4] is sqrt(e) * sin(omega)
    
    params[5 * num_planets] is rv offset for HARPS1
    params[5 * num_planets + 1] is rv offset for HARPS2
    params[5 * num_planets + 2] is rv offset for HIRES
    params[5 * num_planets + 3] is sin(i)
    params[5 * num_planets + 4] is jitter for HARPS1
    params[5 * num_planets + 5] is jitter for HARPS2
    params[5 * num_planets + 6] is jitter for HIRES
    
    param integrator: integrator to use, one of 'whfast' or 'ias15'
    param time_base: base time (to begin integration from) in the simulation
    """
    
    num_planets = 2 # 2 planets
    
    sim = rb.Simulation()
    sim.integrator = integrator
    sim.t = time_base  # keplerian and n-body models initialized at the same time offset
    # print(sim.t)
    if integrator == 'whfast':  # if using whfast integrator, set timestep
        sim.dt = 1/50 * np.min([params[0], params[5]])  # timestep is 1/20th of the shortest orbital period of any planet
        # print(sim.dt)
    sim.units = ('AU', 'Mjupiter', 'day')
    sim.add(m = star_mass)  # star mass as a constant
    
    inclination = np.arcsin(params[-4])  # sin(i) is fourth from the back of the array
        
    for i in range (0, num_planets):
        # print(i)
        # planet parameters
        period = params[5*i]  # in days
        semiamp = params[5*i + 1] / auday_ms # divide by auday_ms because semiamp given in m/s
        eccentricity = params[5*i + 3] ** 2 + params[5*i + 4] ** 2  # eccentricity from secos, sesin
        omega = np.arctan2(params[5*i + 4], params[5*i + 3])  # omega from arctan of sesin, secos  (in that order!)
        # get tp by converting from tc
        tp = radvel.orbit.timetrans_to_timeperi(tc = params[5*i + 2], per = period, ecc = eccentricity, omega = omega)
        
        # mass
        mass = semiamp_to_mass(semiamp = semiamp, star_mass = star_mass, period = period, eccentricity = eccentricity, inclination = inclination)
        
        # adding to simulation
        sim.add(m = mass, P = period, e = eccentricity, T = tp, omega = omega, inc = inclination)
        
    sim.move_to_com()  # move to center of mass
    
    return sim

def get_simple_sim(masses, integrator = 'ias15', period_ratio = 3/2, epsilon=0.01):
    """
    gets simple sim (for eccentricity track stuff)
    param masses: array of planet masses
    param integrator: integrator
    param epsilon: amount by which the resonant period ratio should be offset from the equilibrium in the simulation
    """
    sim = rb.Simulation()
    sim.integrator = integrator
    # central star
    sim.add(m = 1)
    
    sim.add(m = masses[0], P = 1)
    sim.add(m = masses[1], P = period_ratio * (1 + epsilon))

    sim.move_to_com()
    if integrator == 'whfast':
        sim.dt = 1/50 * 1  # dy default use 1/50th of the inner planet's orbital period for the timestep if using whfast
    return sim


def get_rvs(params, instrument, times, integrator, time_base, auday_ms = AUDAY_MS):
    
    """
    Gets RVs from a Numpy array of planet params
    
    param params:     for i in range(0, num_planets):
    
    params[i + 0] is period
    params[i + 1] is semiamp
    params[i + 2] is tc (time of conjunction)
    params[i + 3] is sqrt(e) * cos(omega)
    params[i + 4] is sqrt(e) * sin(omega)
    
    params[5 * num_planets] is rv offset for HARPS1
    params[5 * num_planets + 1] is rv offset for HARPS2
    params[5 * num_planets + 2] is rv offset for HIRES
    params[5 * num_planets + 3] is sin(i) (also params[-4])
    params[5 * num_planets + 4] is jitter for HARPS1 (also params[-3])
    params[5 * num_planets + 5] is jitter for HARPS2 (also params[-2])
    params[5 * num_planets + 6] is jitter for HIRES (also params[-1])

    param instrument: instrument (HARPS1, HARPS2, or HIRES)
    param times: array of times to integrate over
    param integrator: integrator to use, one of 'whfast' or 'ias15'
    
    """
    
    sim = get_sim_from_params(params, integrator, time_base = time_base)
    
    sim_backwards = sim.copy()
    sim_backwards.dt *= -1  # set timestep to be negative if integrating backwards

    times = pd.Series(times)  # convert to series if not already
    
    forward_times = times[times - obs_time_base >= 0]
    backward_times = times[times - obs_time_base < 0]
    forward_indices = forward_times.index
    backward_indices = backward_times.index
    
    # initialize rvs
    rv_forward = np.zeros(len(forward_times))
    rv_backward = np.zeros(len(backward_times))
    
    num_planets = 2  # find number of planets in params passed
    
    # get the rvs (z velocity, assuming 90 deg inclination) from the rebound simulation to compare with the actual simulation
    for j, it in enumerate(zip(forward_indices, forward_times)):
        i, t = it  # forward index, forward time
        sim.integrate(t, exact_finish_time = 1)
        # integrate to the specified time, exact_finish_time = 1 for ias15, 
        # sim.status()
        star = sim.particles[0]
        # print(instrument[i])
        # use one of 3 different radial velocity offsets depending on whether the data is from HARPS1, HARPS2 or HIRES
        if instrument[i] == 'HARPS1':
            rv_offset = params[5 * num_planets]
        elif instrument[i] == 'HARPS2':
            rv_offset = params[5 * num_planets + 1]
        elif instrument[i] == 'HIRES':
            rv_offset = params[5 * num_planets + 2]
        else:
            rv_offset = 0.
        rv_forward[j] = (-star.vz * auday_ms) + rv_offset  # use x-velocity of the star as the radial velocity, convert to m/s
    
    for j, it in enumerate(zip(backward_indices, backward_times)):
        i, t = it  # backward index, backward time
        sim_backwards.integrate(t, exact_finish_time = 1)
        star = sim_backwards.particles[0]
        # use one of 3 different radial velocity offsets depending on whether the data is from HARPS1, HARPS2 or HIRES
        # print(instrument[i])
        if instrument[i] == 'HARPS1':
            rv_offset = params[5 * num_planets]
        elif instrument[i] == 'HARPS2':
            rv_offset = params[5 * num_planets + 1]
        elif instrument[i] == 'HIRES':
            rv_offset = params[5 * num_planets + 2]
        else:
            rv_offset = 0.
        rv_backward[j] = (-star.vz * auday_ms) + rv_offset
    
    return np.concatenate((rv_backward, rv_forward))

def get_resonant_angles(params, times, integrator, time_base, auday_ms = AUDAY_MS):
    
    """
    Gets resonant angles theta_1, theta_2 from an array of parameters
    
    param params:     for i in range(0, num_planets):
    
    params[i + 0] is period
    params[i + 1] is semiamp
    params[i + 2] is tc (time of conjunction)
    params[i + 3] is sqrt(e) * cos(omega)
    params[i + 4] is sqrt(e) * sin(omega)
    
    params[5 * num_planets] is rv offset for HARPS1
    params[5 * num_planets + 1] is rv offset for HARPS2
    params[5 * num_planets + 2] is rv offset for HIRES
    params[5 * num_planets + 3] is sin(i) (also params[-4])
    params[5 * num_planets + 4] is jitter for HARPS1 (also params[-3])
    params[5 * num_planets + 5] is jitter for HARPS2 (also params[-2])
    params[5 * num_planets + 6] is jitter for HIRES (also params[-1])

    param instrument: instrument (HARPS1, HARPS2, or HIRES)
    param times: array of times to integrate over
    param integrator: integrator to use, one of 'whfast' or 'ias15'
    
    """
    
    sim = get_sim_from_params(params, integrator, time_base = time_base)
    
    sim_backwards = sim.copy()
    sim_backwards.dt *= -1  # set timestep to be negative if integrating backwards

    times = pd.Series(times)  # convert to series if not already
    
    forward_times = times[times - obs_time_base >= 0]
    backward_times = times[times - obs_time_base < 0]
    forward_indices = forward_times.index
    backward_indices = backward_times.index
    
    # initialize libration angle for inner and outer planets
    resonant_angle1_forward = np.zeros(len(forward_times))
    resonant_angle1_backward = np.zeros(len(backward_times))
    resonant_angle2_forward = np.zeros(len(forward_times))
    resonant_angle2_backward = np.zeros(len(backward_times))
    
    num_planets = 2  # find number of planets in params passed

    inner = sim.particles[1]
    outer = sim.particles[2]

    for i, t in enumerate(forward_times):
        sim.integrate(t, exact_finish_time = 1)
        # integrate to the specifided time
        resonant_angle1_forward[i] = np.mod(3 * outer.l - 2 * inner.l - inner.pomega, 2 * np.pi)  # theta1
        resonant_angle2_forward[i] = np.mod(3 * outer.l - 2 * inner.l - outer.pomega, 2 * np.pi)  # theta1
    
    for i, t in enumerate(backward_times):
        sim.integrate(t, exact_finish_time = 1)
        # integrate to the specified time
        resonant_angle1_backward[i] = np.mod(3 * outer.l - 2 * inner.l - inner.pomega, 2 * np.pi)  # theta1
        resonant_angle2_backward[i] = np.mod(3 * outer.l - 2 * inner.l - outer.pomega, 2 * np.pi)  # theta1

    # concatentate forward and backward libration angle to get the libration angle for every timestep, for both inner and outer planet
    resonant_angles_inner = np.concatenate((resonant_angle1_backward, resonant_angle1_forward))
    resonant_angles_outer = np.concatenate((resonant_angle2_backward, resonant_angle2_forward))
    
    return resonant_angles_inner, resonant_angles_outer


# import data
# cluster_data = h5py.File('../mcmc_hd45364_everything_with_libration_penalty_variable_1.h5', 'r')  # import data
cluster_data = h5py.File('../mcmc_hd45364_cluster_everything.h5', 'r')  # import data
accepted, samples, log_prob = np.array(cluster_data['mcmc']['accepted']), np.array(cluster_data['mcmc']['chain']), np.array(cluster_data['mcmc']['log_prob'])

burnin = 200  # number of sampels to discard for burn-in

accepted = accepted[burnin:]
samples = samples[burnin:]
log_prob = log_prob[burnin:]

# Compute and plot RV signal with 1, 2 and 3 sigma for all the samples, following Fig. 2 of H&P
n_rv_samples = len(samples)//1
rv_sample_indices = np.random.randint(0, len(samples), size=n_rv_samples)

times = np.linspace(np.min(hd_data.BJD), np.max(hd_data.BJD), int(1e4))
instruments = np.linspace(np.min(hd_data.BJD), np.max(hd_data.BJD), int(1e4))

# multiprocessing
pool = multiprocessing.Pool()
rv_samples = np.array(pool.starmap(get_rvs, zip(samples[rv_sample_indices][:,0], 
    (np.repeat(instruments, n_rv_samples).reshape(-1, n_rv_samples)).T, 
    (np.repeat(times, n_rv_samples).reshape(-1, n_rv_samples)).T, 
    np.repeat('ias15', n_rv_samples), 
    np.repeat(obs_time_base, n_rv_samples))))

# close and join
pool.close()
pool.join()

# save
np.save('rv_samples2', rv_samples)