import h5py
import rebound as rb
import numpy as np
import os
import multiprocessing

# Parameters:
obs_time_base = 54422.789665  # Epoch minus 2.4e6 BJD
integrator = 'IAS15'  # Integrator to use by default
star_mass = 859.  # mass of the star in Jupiter masses
auday_ms = 1.731e6  # conversion factor for au/day to m/s
G = 2.825e-7  # G in units of jupiter masses, au, and days

# Import posteriors
uninf_posteriors = h5py.File('./hd45364_mcmc_uninformative_model_posteriors.h5', 'r')
rw_posteriors = h5py.File('./hd45364_mcmc_resonance_weighted_model_posteriors.h5', 'r')
# Get samples
uninf_samples = np.array(uninf_posteriors['mcmc']['chain']) # dimensions of (n_chains, n_steps, n_parameters)
rw_samples = np.array(rw_posteriors['mcmc']['chain'])

# Convert the parameters to rebound simulations:
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

# Function taken from the RadVel package (Fulton et al. 2018)
# https://doi.org/10.5281/zenodo.580821

def timetrans_to_timeperi(tc, per, ecc, omega):
    """
    Convert Time of Transit to Time of Periastron Passage

    Args:
        tc (float): time of transit
        per (float): period [days]
        ecc (float): eccentricity
        omega (float): longitude of periastron (radians)

    Returns:
        float: time of periastron passage

    """
    try:
        if ecc >= 1:
            return tc
    except ValueError:
        pass

    f = np.pi/2 - omega
    ee = 2 * np.arctan(np.tan(f/2) * np.sqrt((1-ecc)/(1+ecc)))  # eccentric anomaly
    tp = tc - per/(2*np.pi) * (ee - ecc*np.sin(ee))      # time of periastron

    return tp


def get_sim_from_params(params, path_to_save, chain_num, step_num, integrator=integrator, time_base=obs_time_base, star_mass=star_mass, auday_ms=auday_ms, num_planets=2):
    """
    Returns a rebound.Simulation object from a given array of parameters
    
    param params: Numpy array of params:
    
    For the ith planet:
    params[i + 0] is period 
    params[i + 1] is semiamp
    params[i + 2] is tc (time of conjunction)
    params[i + 3] is sqrt(e) * cos(omega)
    params[i + 4] is sqrt(e) * sin(omega)
    
    For the system:
    params[5 * num_planets] is rv offset for HARPS1
    params[5 * num_planets + 1] is rv offset for HARPS2
    params[5 * num_planets + 2] is rv offset for HIRES
    params[5 * num_planets + 3] is sin(I)
    params[5 * num_planets + 4] is jitter for HARPS1
    params[5 * num_planets + 5] is jitter for HARPS2
    params[5 * num_planets + 6] is jitter for HIRES
    
    param chain_num: chain number of sample
    param step_num: step number of sample

    Default parameters:
    param integrator: integrator to use, one of 'whfast' or 'ias15'
    param time_base: epoch to begin integration from in the simulation
    param star_mass: mass of star in Jupiter masses
    param auday_ms: conversion factor for AU/day to m/s
    param num_planets: number of planets in the system
    """
    
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
        tp = timetrans_to_timeperi(tc = params[5*i + 2], per = period, ecc = eccentricity, omega = omega)
        
        # mass
        mass = semiamp_to_mass(semiamp = semiamp, star_mass = star_mass, period = period, eccentricity = eccentricity, inclination = inclination)
        
        # adding to simulation
        sim.add(m = mass, P = period, e = eccentricity, T = tp, omega = omega, inc = inclination)
        
    sim.move_to_com()  # move to center of mass
    sim.save_to_file(path_to_save + 'sim_chain_' + str(chain_num) + '_step_' + str(step_num) + '.bin') # save simulation as binary file
    return None

# Convert to REBOUND simulations and save using multiprocessing
# Uninformative priors:
uninf_path_to_save = './rebound_simulations/uninf_prior/'
if not os.path.exists(uninf_path_to_save):
    os.makedirs(uninf_path_to_save)
uninf_pool = multiprocessing.Pool(multiprocessing.cpu_count() - 1) 
for i, uninf_chain in enumerate(np.transpose(uninf_samples, axes=[1, 0, 2])):
    uninf_pool.starmap(get_sim_from_params, zip(uninf_chain, 
                                          np.repeat(uninf_path_to_save, len(uninf_chain)), 
                                          np.repeat(i, len(uninf_chain)), 
                                          np.arange(0, len(uninf_chain))))
uninf_pool.close()
uninf_pool.join()


# Resonance-weighted priors:
rw_pool = multiprocessing.Pool(multiprocessing.cpu_count() - 1) 
rw_path_to_save = './rebound_simulations/rw_prior/'
if not os.path.exists(rw_path_to_save):
    os.makedirs(rw_path_to_save)
for i, rw_chain in enumerate(np.transpose(rw_samples, axes=[1, 0, 2])):
    rw_pool.starmap(get_sim_from_params, zip(rw_chain, 
                                          np.repeat(rw_path_to_save, len(rw_chain)), 
                                          np.repeat(i, len(rw_chain)), 
                                          np.arange(0, len(rw_chain))))
rw_pool.close()
rw_pool.join()