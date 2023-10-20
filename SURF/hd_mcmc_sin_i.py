import pickle
import radvel
import rebound as rb
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import celmech as cm
from tqdm import tqdm
from scipy import optimize, stats
import corner
from radvel.plot import mcmc_plots
import reboundx
import emcee
from multiprocessing import Pool

# DON'T THINK I NEED THIS SINCE I ALREADY HAVE THE ORIGINAL PARAMETERS USED IN HADDEN AND PAYNE, AS WELL AS THE LAST SQUARES FIT:

# with open("init_params_hd45364.bin", 'rb') as file:
#     old_params = pickle.load(file)

# LOAD THE DATA HERE, MAKING SURE TO DROP THE BAD DATA

hd_data = pd.read_csv('hd45364_rvs.csv', sep = ';')
# giant outlier at position 116 in the data (found manually earlier) which we remove
hd_data.drop(116, inplace=True)  # drop the row and keep the df in place
# subtract 2.4e6 from all the rows in the data
hd_data.BJD -= 2.4e6

#Very original parameters used in Hadden and Payne
nbody_params =[ 2.27798546e+02,  7.25405874e+00,  5.39392010e+04,  1.71866112e-01, 
               1.17923823e-01,  3.43881599e+02,  1.87692753e+01,  5.40138425e+04, 
               1.68408461e-01,  5.05903191e-02, -3.28526403e-03, 1]

#Least squares fit: 
fit_params = [ 2.28512793e+02, 7.27736501e+00, 5.39371914e+04, -4.66868256e-02, 
               -1.78080009e-01, 3.43378038e+02, 1.78603341e+01, 5.40186750e+04, 
               9.72945632e-02,  1.32194117e-01, -5.29072002e-01, 1]#-7.68527759e-03] 

# this does not include jitter! the last term is taken from the post params with pickle (nbody_params in the original ipynb)

## SKIPPING ALL THE PLOTTING AS WELL

## CONSTANTS:

STAR_MASS = 920  # 920 jupiter masses
G = 2.825e-7  # converting G to jupiter masses, au, and days
AUDAY_MS = 1.731e6  # conversion factor for au/day to m/s

# use median of time data as the time base:
obs_time_base = np.median(hd_data.BJD)

print(f'nbody_params:{nbody_params}\n fit_params:{fit_params}')

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


def get_sim_from_params(params, integrator, time_base, star_mass=STAR_MASS, auday_ms=AUDAY_MS):
    """
    takes in params array, returns a rebound Simulation object with those parameters

    param params: numpy array of params:

    for i in range(0, num_planets):

    params[i + 0] is period
    params[i + 1] is semiamp
    params[i + 2] is tc (time of conjunction)
    params[i + 3] is sqrt(e) * cos(omega)
    params[i + 4] is sqrt(e) * sin(omega)

    params[5 * num_planets] is rv offset
    params[5 * num_planets + 1] is sin(i)

    param inclination: inclination of system in the observation plane (pi/2 is in the plane of the sky, 0 is edge-on)
    param integrator: integrator to use, one of 'whfast' or 'ias15'
    param time_base: base time (to begin integration from) in the simulation
    """

    num_planets = int((len(params) - 1) / 5)  # -2 because there are rv_offset and jit parameters:

    sim = rb.Simulation()
    sim.integrator = integrator
    sim.t = time_base  # keplerian and n-body models initialized at the same time offset
    # print(sim.t)
    if integrator == 'whfast':  # if using whfast integrator, set timestep
        sim.dt = 1 / 50 * min(params[0::5][:-1])  # timestep is 1/20th of the shortest orbital period of any planet
        # print(sim.dt)
    sim.units = ('AU', 'Mjupiter', 'day')
    sim.add(m=star_mass)  # star mass as a constant

    inclination = np.arcsin(params[-1])  # sin(i) is at the back of the array

    for i in range(0, num_planets):
        # print(i)
        # planet parameters
        period = params[5 * i]  # in days
        semiamp = params[5 * i + 1] / auday_ms  # divide by auday_ms because semiamp given in m/s
        eccentricity = params[5 * i + 3] ** 2 + params[5 * i + 4] ** 2  # eccentricity from secos, sesin
        omega = np.arctan2(params[5 * i + 3], params[5 * i + 4])  # omega from arctan of sesin, secos
        # get tp by converting from tc
        tp = radvel.orbit.timetrans_to_timeperi(params[5 * i + 2], per=period, ecc=eccentricity, omega=omega)

        # mass
        mass = semiamp_to_mass(semiamp=semiamp, star_mass=star_mass, period=period, eccentricity=eccentricity,
                               inclination=inclination)

        # adding to simulation
        sim.add(m=mass, P=period, e=eccentricity, T=tp, omega=omega, inc=inclination)

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


def get_rvs(params, times, integrator, time_base, auday_ms = AUDAY_MS):
    
    """
    Gets RVs from a Numpy array of planet params
    
    param params:     for i in range(0, num_planets):
    
    params[i + 0] is period
    params[i + 1] is semiamp
    params[i + 2] is tc (time of conjunction)
    params[i + 3] is sqrt(e) * cos(omega)
    params[i + 4] is sqrt(e) * sin(omega)
    
    params[5 * num_planets] is rv offset
    params[5 * num_planets + 1] is sin(i) (also params[-2])
    
    param inclination: inclination of system in the observation plane (pi/2 is in the plane of the sky, 0 is edge-on)
    param times: array of times to integrate over
    param integrator: integrator to use, one of 'whfast' or 'ias15'
    
    """
    sim = get_sim_from_params(params, integrator, time_base=time_base)

    sim_backwards = sim.copy()
    sim_backwards.dt *= -1  # set timestep to be negative if integrating backwards

    forward_times = np.array(list(filter(lambda x: x - post.model.time_base >= 0, times)))
    backward_times = np.array(list(filter(lambda x: x - post.model.time_base < 0, times)))

    # initialize rvs
    rv_forward = np.zeros(len(forward_times))
    rv_backward = np.zeros(len(backward_times))

    num_planets = int((len(params) - 1) / 5)  # find number of planets in params passed

    # get the rvs (z velocity, assuming 90 deg inclination) from the rebound simulation to compare with the actual simulation
    for i, t in enumerate(forward_times):
        sim.integrate(t, exact_finish_time=1)
        # integrate to the specified time, exact_finish_time = 1 for ias15,
        # sim.status()
        star = sim.particles[0]
        rv_forward[i] = (-star.vz * auday_ms) + params[5 * num_planets]  # use x-velocity of the star as the radial velocity, convert to m/s

    for i, t in enumerate(backward_times):
        sim_backwards.integrate(t, exact_finish_time=1)
        star = sim_backwards.particles[0]
        rv_backward[i] = (-star.vz * auday_ms) + params[5 * num_planets]

    return np.concatenate((rv_backward, rv_forward))
    

# OPTIMIZE OVER NEGATIVE LOG LIKELIHOOD, THIS IS EQUIVALENT TO GET_NBODY_RESIDS BELOW

def neg_log_likelihood(params, data = hd_data):
    """
    Gets the negative log-likelihood (without a jitter term) for use with scipy.optimize.minimze
    
    Implements the log likelihood using the same method above
    
    """
    # LOG LIKELIHOOD
    obs_y = data.RV_mlc_nzp  # observed RVs
    
    synth_y = get_rvs(params, data.BJD, 'ias15')  # RVs from the rebound simulation
    obs_yerr = data.e_RV_mlc_nzp  # y errors
    # # compute variance
    # variance = obs_yerr ** 2  # + (synth_y * np.exp(log_f)) ** 2  # assuming simply that variance is underestimated by some amount f

    # # # compute log likelihood
    # log_likelihood = -1/2 * np.sum(np.log(variance) + ((obs_y - synth_y) ** 2/variance))  # this is equivalent up to a 2pi constant
    
    # log_likelihood = -1 / 2 * np.sum(np.log(variance) + ((obs_y - synth_y) ** 2 / variance))
    log_likelihood = -1/2 * np.sum(((obs_y - synth_y) ** 2)/(obs_yerr ** 2) 
                                   + np.log(np.sqrt(2 * np.pi * (obs_yerr ** 2))))

    return -log_likelihood  # negative since we are trying to minimize the negative log likelihood

def get_nbody_resids(params, integrator, data = hd_data):
    """
    Gets the normalized residuals for the n-body fit with REBOUND
    """
    obs_y = data.RV_mlc_nzp  # observed RVs
    synth_y = get_rvs(params, data.BJD, integrator, time_base=obs_time_base)  # RVs from the rebound simulation
    obs_yerr = data.e_RV_mlc_nzp  # y errors
    return (obs_y - synth_y) / obs_yerr  # return normalized residuals


# OPTIMIZE USING OPTIMIZE.MINIMIZE WITH THE LOG LIKELIHOOD INSTEAD OF JUST NORMAL LEAST-SQUARES FIRST AND COMPARE

# bounds of (0, 1) for sin(i), everything else can vary however: bounds format for optimize.minimize()
bounds = ((None, None), (None, None), (None, None), (None, None), (None, None), (None, None), (None, None), (None, None), (None, None), (None, None), (None, None), (0, 1) ,(None, None))

best_fit = optimize.minimize(neg_log_likelihood, x0=np.array(fit_params), method='Nelder-Mead', bounds=bounds, options={'maxiter': int(1e5), 'maxfev': int(1e5)})  # optimization

# bounds format for optimize.least_squaares()
bounds2 = ([-np.inf, -np.inf, -np.inf, -np.inf, -np.inf, -np.inf, -np.inf, -np.inf, -np.inf, -np.inf, -np.inf, 0],
          [np.inf, np.inf, np.inf, np.inf, np.inf, np.inf, np.inf, np.inf, np.inf, np.inf, np.inf, 1])  # bounds of [0, 1] for sin(i)
fit_params_chi2 = optimize.least_squares(lambda params: get_nbody_resids(params, integrator = 'ias15'), best_fit.x, bounds=bounds2)

print(f'original guess:\n{np.array(fit_params)}\n\noptimization without jitter using negloglike:\n{best_fit.x}\n\optimization using ls and ias15:\n{best_fit.x}')

# EMCEE STUFF:
print('MCMC STUFF:')

# LOG PRIOR
def log_prior(params, e_max=0.8, sin_i_min=0.3):
    ps = params[0:-3:5]  # start at 0, the last 3 elements of params are not planet params (rv_offset, sin(i), jitter)
    ks = params[1:-3:5]  # semiamps
    tcs = params[2:-3:5]  # times of conjunction
    # compute e and omega from secos, sesin
    es = params[3:-3:5] ** 2 + params[4:-3:5] ** 2  # eccentricity from secos, sesin
    # omega = np.arctan2(params[3:-3:5], params[4:-3:5])  # omega from arctan of sesin, secos
    sin_i = params[-1]  # sin(i) is the second-to-last item of the array

    # uniform log prior, return 0 if param falls within uniform distribution, -infinity otherwise
    # print(ps, ks, tcs, es, omega)

    if all(p > 0 for p in ps) and all(k > 0 for k in ks) and all(tc > 0 for tc in tcs) and all(0 < e < e_max for e in es) and (sin_i_min < sin_i <= 1.):
        return 0.0  # log prior, so ln(1) = 0
    else:
        return -np.inf  # log prior, so ln(0) = -infinity

# LOG LIKELIHOOD
def log_likelihood(params, data=hd_data):
    """
    Gets the log-likelihood (negative of the negative log likelihood) (including a jitter term!) for use with scipy.optimize.minimze
    
    Implements the log likelihood using the same method as neg_log_likelihood above
    """
    obs_y = data.RV_mlc_nzp  # observed RVs
    synth_y = get_rvs(params, data.BJD, 'ias15')  # RVs from the rebound simulation
    obs_yerr = data.e_RV_mlc_nzp  # y errors

    # compute variance
    variance = obs_yerr ** 2  # + (synth_y * np.exp(log_f)) ** 2  # assuming simply that variance is underestimated by some amount f

    # compute log likelihood
    log_likelihood = -1 / 2 * np.sum(np.log(variance) + ((obs_y - synth_y) ** 2 / variance))

    return log_likelihood

# LOG PROBABILITY
def log_probability(params):
    lp = log_prior(params)
    if not np.isfinite(lp):
        return -np.inf
    return lp + log_likelihood(params)

# COVARIANCE MATRIX CALCULATIONS
j = fit_params_chi2.jac  # jacobian
cov = np.linalg.inv(j.T @ j)  # covariance matrix from jacobian sigma = (X^T * X)^(-1)
best = fit_params_chi2.x  # best-fit solution is our center

# initialize walkers
nwalkers = 50  # number of walkers to use in MCMC
ndim = len(best)  # number of dimensions in parameter space
# gaussian ball of 50 walkers with variance equal to cov * 1/100 and centered on the best-fit solution
pos = np.random.multivariate_normal(best, cov * 1/100, size = nwalkers)

# save MCMC sample chain to a file
filename = "mcmc_hd45364_cluster_sin_i.h5"
backend = emcee.backends.HDFBackend(filename)
backend.reset(nwalkers, ndim)

steps = 50000  # try 50000 steps with multiprocessing on the cluster

# RUNNING MCMC (parallelization):

with Pool() as pool:
    sampler = emcee.EnsembleSampler(nwalkers, ndim, log_probability, pool = pool, backend = backend)
    sampler.run_mcmc(pos, steps, progress=True)  # this takes 80 minutes or so to run on laptop

samples = sampler.get_chain()