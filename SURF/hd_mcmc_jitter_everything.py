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
from emcee.interruptible_pool import InterruptiblePool
from multiprocessing import Pool

# DON'T THINK I NEED THIS SINCE I ALREADY HAVE THE ORIGINAL PARAMETERS USED IN HADDEN AND PAYNE, AS WELL AS THE LAST SQUARES FIT:

# with open("init_params_hd45364.bin", 'rb') as file:
#     old_params = pickle.load(file)

# LOAD THE DATA HERE, MAKING SURE TO DROP THE BAD DATA

MAY_1_2015 = 57143.5  # barycentric julian date for May 1, 2015 (the date of the HARPS instrument upgrade as per trifonov et al 2020)
# 57143.5 is BJD for May 1, 2015
# 57173.5 is BJD for May 31, 2015

# harps
hd_data_harps = pd.read_csv('hd45364_rvs.csv', sep = ';')
# giant outlier at position 116 in the data (found manually earlier) which we remove
hd_data_harps.drop(116, inplace=True)  # drop the row and keep the df in place
# subtract 2.4e6 from all the rows in the data
hd_data_harps.BJD -= 2.4e6
# rename target to HARPS1 or HARPS2
hd_data_harps['target'] = hd_data_harps.apply(lambda row: 'HARPS1' if row.BJD < MAY_1_2015 else 'HARPS2', axis = 1)
# hires
hd_data_hires = pd.read_csv('hires_rvs.txt', sep = '\t', index_col=False, header='infer', dtype=np.float64)
hd_data_hires['BJD - 2,450,000'] += 50000.  # adding 50000 to have the same units as harps
hd_data_hires['target'] = 'HIRES'
hd_data_hires.columns = ['BJD', 'RV_mlc_nzp', 'e_RV_mlc_nzp', 'target']
# concatenate two data sets one on top of the other
hd_data = pd.concat((hd_data_harps, hd_data_hires), axis=0)  # matching BJD, RV_mlc_nzp and e_RV_mlc_nzp columns
# reset index
hd_data.reset_index(drop=True, inplace=True)

# NEW STUFF FOR THE JITTER VERSION OF THIS SCRIPT:

#Very original parameters used in Hadden and Payne
nbody_params =[ 2.27798546e+02,  7.25405874e+00,  5.39392010e+04,  1.71866112e-01, 1.17923823e-01,  
               3.43881599e+02,  1.87692753e+01,  5.40138425e+04, 1.68408461e-01,  5.05903191e-02, 
               -3.28526403e-03, 0., 0., 
               1, 
               1.84, 0., 0.]  # inserted 0 for harps2 and hires for both rv offset and jitter

# #Least squares fit: 
# fit_params = [ 2.28512793e+02, 7.27736501e+00, 5.39371914e+04, -4.66868256e-02, 
#                -1.78080009e-01, 3.43378038e+02, 1.78603341e+01, 5.40186750e+04, 
#                9.72945632e-02,  1.32194117e-01, -5.29072002e-01, 0., 0., 1, 2.428]#-7.68527759e-03] 

# Neg log likelihood jitter fit:
prev_params = [ 2.27510047e+02,  7.21459722e+00, 5.39394197e+04, -1.45510376e-02, -1.91998583e-01,
              3.44196007e+02,  1.80943200e+01,  5.47060928e+04, 9.38174624e-02,  1.11054397e-01,
              -1.80048668e-01, -1.44155418e+00, 1.40493043e+00,
              1.00000000e+00,
              1.46046278e+00,  6.96508946e-01, 3.45217643e+00]

# LI ET AL. 2022 PARAMS
li_params = [225.34, 7.26, radvel.orbit.timeperi_to_timetrans(53375, 225.34, 0.07, np.deg2rad(92)), np.sqrt(0.07) * np.cos(np.deg2rad(92)), np.sqrt(0.07) * np.sin(np.deg2rad(92)),
             345.76, 18.17, radvel.orbit.timeperi_to_timetrans(53336, 345.76, 0.01, np.deg2rad(276)), np.sqrt(0.01) * np.cos(np.deg2rad(276)), np.sqrt(0.01) * np.cos(np.deg2rad(276)),
            -1.80048668e-01, -1.44155418e+00, 1.40493043e+00,
              1.00000000e+00,
              1.46046278e+00,  6.96508946e-01, 3.45217643e+00]

# TRIFONOV PARAMS
trifonov_params = [226.57, 7.29, radvel.orbit.timeperi_to_timetrans(52902, 226.57, 0.0796, np.deg2rad(244.44)), np.sqrt(0.0796) * np.cos(np.deg2rad(244.44)), np.sqrt(0.0796) * np.sin(np.deg2rad(244.44)),
              344.66, 18.21, radvel.orbit.timeperi_to_timetrans(52920, 344.66, 0.002, np.deg2rad(20.342)), np.sqrt(0.0002) * np.cos(np.deg2rad(20.342)), np.sqrt(0.0002) * np.sin(np.deg2rad(20.342)),
              0.041, -3.348, 2.708,
              np.sin(np.deg2rad(83.7597)),
              1.437, 0.763, 3.136]

# UPDATED TRIFONOV PARAMS
fit_params = [2.27864638e+02, 7.19443190e+00, 5.27993627e+04, -7.26813509e-03, -2.15682280e-01, 
              3.44040155e+02, 1.82002701e+01, 5.29855398e+04, 1.11463111e-01, 3.12038118e-02, 
              -1.41032815e-01, -2.93573404e+00, 1.65809757e+00, 
              1., 
              1.40629135e+00, 8.26926669e-01, 3.03850222e+00]

# this includes jitter! the last term is taken from the post params with pickle (nbody_params in the original ipynb)

## SKIPPING ALL THE PLOTTING AS WELL

## CONSTANTS:

STAR_MASS = 920  # 920 jupiter masses
G = 2.825e-7  # converting G to jupiter masses, au, and days
AUDAY_MS = 1.731e6  # conversion factor for au/day to m/s

# use median of harps data as base time
obs_time_base = np.median(hd_data_harps.BJD)

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
    
    forward_times = times[times - time_base >= 0]
    backward_times = times[times - time_base < 0]
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

# OPTIMIZE OVER NEGATIVE LOG LIKELIHOOD INSTEAD OF JUST CHI SQUARED

def neg_log_likelihood(params, time_base = obs_time_base, data = hd_data, num_planets = 2):
    """
    Gets the negative log-likelihood (including a jitter term!) for use with scipy.optimize.minimze
    
    Iplements the log likelihood using the same method above
    
    """
    obs_y = data.RV_mlc_nzp  # observed RVs
    
    # inclination not handled sparately
    # inclination = np.arcsin(params[-4])  # inclination is np.arcsin of the second to last parameter
    
    synth_y = get_rvs(params, data.target, data.BJD, 'ias15', time_base = time_base)  # RVs from the rebound simulation
    obs_yerr = data.e_RV_mlc_nzp  # y errors

    conditions = [data.target == 'HARPS1', data.target == 'HARPS2', data.target == 'HIRES']  # conditions are harps1, harps2 or hires

    rv_offsets = params[5 * num_planets:5 * num_planets + 3]  # rv_offsets for HARPS1, HARPS2 and HIRES, in that order
    jitters = params[-3:]  # jitters for HARPS1, HARPS2 and HIRES, in that order
    
    # get the jitter and rv values for the corresponding data points
    rv_offset = np.select(conditions, rv_offsets, default=np.nan)
    jitter = np.select(conditions, jitters, default=np.nan)
    # print(rv_offset, jitter)

    # compute the log-likelihood
    #### OLD
    # log_likelihood = -1/2 * np.sum(((obs_y - synth_y) ** 2)/(obs_yerr ** 2 + jitter ** 2) 
    #                                + np.log(np.sqrt(2 * np.pi * (obs_yerr ** 2 + jitter ** 2))))

    #### LI ET AL. 2022 VERSION (NEW)
    sigma_z2 = 1/(np.sum(1/(obs_yerr ** 2 + jitter ** 2)))
    log_likelihood = -1/2 * np.sum(((obs_y - synth_y) ** 2)/((obs_yerr ** 2 + jitter ** 2))) - np.sum(np.log(np.sqrt(2 * np.pi * (obs_yerr ** 2 + jitter ** 2)))) + np.log(np.sqrt(2 * np.pi * sigma_z2))
    # log_likelihood = -1/2 * np.sum(np.log(variance) + ((obs_y - synth_y) ** 2/variance))    
    # print(-1/2 * np.sum(((obs_y - rv_offset - synth_y) ** 2)/((obs_yerr ** 2 + jitter ** 2))))
    # print(-log_likelihood)
    return -log_likelihood  # negative since we are trying to minimize the negative log likelihood

# AGAIN, OPTIMIZE USING OPTIMIZE.MINIMIZE WITH THE LOG LIKELIHOOD INSTEAD OF JUST NORMAL LEAST-SQUARES

def ecc_con1(params):
    return 1 - (params[3] ** 2 + params[4] ** 2)

def ecc_con2(params):
    return 1 - (params[8] ** 2 + params[9] ** 2)

cons = ({'type': 'ineq', 'fun': ecc_con1}, 
        {'type': 'ineq', 'fun': ecc_con2})

# bounds of (0, 1) for sin(i), everything else can vary however
bounds = ((None, None), (None, None), (None, None), (None, None), (None, None), 
          (None, None), (None, None), (None, None), (None, None), (None, None), 
          (None, None), (None, None), (None, None),
          (0, 1), 
          (None, None), (None, None), (None, None))
#### BFGS and L-BFGS do not work without hacky fixes (returns RV array full of nans (ecc greater than 1))
#### Nelder-Mead seems to work...
best_fit_jitter = optimize.minimize(neg_log_likelihood, x0=np.array(fit_params), method='Nelder-Mead', 
                                    bounds=bounds, constraints=cons, options={'maxiter': 1000000000000, 
                                                                              'maxfev': 1000000000000, 
                                                                              # 'ftol': 1.e-7
                                                                             }
                                   )  # optimization

print(f'original guess:\n{np.array(fit_params)}\n\noptimization with jitter:\n{best_fit_jitter.x}\n\n')

print(best_fit_jitter)

# EMCEE STUFF:
print('MCMC STUFF:')

# LOG PRIOR
def log_prior(params, e_max=0.8, sin_i_min=0.076):
    ps = params[0:-7:5]  # start at 0, the last 7 elements of params are not planet params (rv_offset1, rvoffset2, rvoffset3, sin(i), jitter1, jitter2, jitter3)
    ks = params[1:-7:5]  # semiamps
    tcs = params[2:-7:5]  # times of conjunction
    # compute e and omega from secos, sesin
    es = params[3:-7:5] ** 2 + params[4:-7:5] ** 2  # eccentricity from secos, sesin
    # omega = np.arctan2(params[3:-3:5], params[4:-3:5])  # omega from arctan of sesin, secos
    sin_i = params[-4]  # sin(i) is the fourth-to-last item of the array
    # jitters
    jitters = params[-3:]  # jitters

    # uniform log prior, return 0 if param falls within uniform distribution, -infinity otherwise
    # print(ps, ks, tcs, es, omega)

    if all(p > 0. for p in ps) and all(k > 0. for k in ks) and all(tc > 0. for tc in tcs) and all(0. < e < e_max for e in es) and (sin_i_min < sin_i <= 1.) and all(0. <= jitter <= 10. for jitter in jitters):
        return 0.0  # log prior, so ln(1) = 0
    else:
        return -np.inf  # log prior, so ln(0) = -infinity

# LOG LIKELIHOOD
def log_likelihood(params, time_base = obs_time_base, data = hd_data, num_planets = 2):
    """
    Gets the log-likelihood (negative of the negative log likelihood) (including a jitter term!) for use with scipy.optimize.minimze
    
    Implements the log likelihood using the same method as neg_log_likelihood above
    
    """
    obs_y = data.RV_mlc_nzp  # observed RVs
    
    # inclination not handled sparately
    # inclination = np.arcsin(params[-4])  # inclination is np.arcsin of the second to last parameter
    
    synth_y = get_rvs(params, data.target, data.BJD, 'ias15', time_base = time_base)  # RVs from the rebound simulation
    obs_yerr = data.e_RV_mlc_nzp  # y errors

    conditions = [data.target == 'HARPS1', data.target == 'HARPS2', data.target == 'HIRES']  # conditions are harps1, harps2 or hires

    rv_offsets = params[5 * num_planets:5 * num_planets + 3]  # rv_offsets for HARPS1, HARPS2 and HIRES, in that order
    jitters = params[-3:]  # jitters for HARPS1, HARPS2 and HIRES, in that order
    
    # get the jitter and rv values for the corresponding data points
    rv_offset = np.select(conditions, rv_offsets, default=np.nan)
    jitter = np.select(conditions, jitters, default=np.nan)
    # print(rv_offset, jitter)

    # compute the log-likelihood
    #### OLD
    # log_likelihood = -1/2 * np.sum(((obs_y - synth_y) ** 2)/(obs_yerr ** 2 + jitter ** 2) 
    #                                + np.log(np.sqrt(2 * np.pi * (obs_yerr ** 2 + jitter ** 2))))

    #### LI ET AL. 2022 VERSION (NEW)
    sigma_z2 = 1/(np.sum(1/(obs_yerr ** 2 + jitter ** 2)))
    log_likelihood = -1/2 * np.sum(((obs_y - synth_y) ** 2)/((obs_yerr ** 2 + jitter ** 2))) - np.sum(np.log(np.sqrt(2 * np.pi * (obs_yerr ** 2 + jitter ** 2)))) + np.log(np.sqrt(2 * np.pi * sigma_z2))
    # log_likelihood = -1/2 * np.sum(np.log(variance) + ((obs_y - synth_y) ** 2/variance))    
    # print(-1/2 * np.sum(((obs_y - rv_offset - synth_y) ** 2)/((obs_yerr ** 2 + jitter ** 2))))
    # print(-log_likelihood)
    return log_likelihood  # positive

# LOG PROBABILITY
def log_probability(params):
    lp = log_prior(params)
    if not np.isfinite(lp):
        return -np.inf
    return lp + log_likelihood(params)

# DO LEAST-SQUARES OPTIMIZATION HERE TO GET A JACOBIAN FOR THE NEXT PART
# The actual least-squares "fit" will probably be slightly worse compared to the fit using negative log-likelihodo (because it doesn't take into account jitter), but hopefully the jacobian is close enough to just use with the original best fit (the one computed using negloglikelihood instead of least-squares) to get a decent starting position for the walkers

def get_nbody_resids(params, integrator, time_base = obs_time_base, data = hd_data):
    """
    Gets the normalized residuals for the n-body fit with REBOUND
    """
    obs_y = data.RV_mlc_nzp  # observed RVs
    synth_y = get_rvs(params, data.target, data.BJD, integrator, time_base=time_base)  # RVs from the rebound simulation
    obs_yerr = data.e_RV_mlc_nzp  # y errors
    return (obs_y - synth_y) / obs_yerr  # return normalized residuals

# get the "fit params" to get the jacobian from this:
bounds2 = ([-np.inf, -np.inf, -np.inf, -np.inf, -np.inf, 
            -np.inf, -np.inf, -np.inf, -np.inf, -np.inf, 
            -np.inf, -np.inf, -np.inf, 
            0, 
            -np.inf, -np.inf, -np.inf],
          [np.inf, np.inf, np.inf, np.inf, np.inf, 
           np.inf, np.inf, np.inf, np.inf, np.inf, 
           np.inf, np.inf, np.inf, 
           1, 
           np.inf, np.inf, np.inf])
jacobian_fit_params = optimize.least_squares(lambda params: get_nbody_resids(params, integrator='ias15'), best_fit_jitter.x, bounds=bounds2)

# COVARIANCE MATRIX CALCULATIONS
j = jacobian_fit_params.jac  # jacobian, value for jitter is 0 though so hopefully we can manually patch the jacobian to get something reasonable

# MANUALLY PATCHING JITTER FOR THE JACOBIAN:
j[-1][-1] = 0.01  # empiriclaly determined value of onesigma_jit in the hd45364_jitter_sini notebook, about one sigma value of jitter, roughly divided by 3
j[-2][-2] = 0.01  
j[-3][-3] = 0.01
# manually patching RV offset for the jacobian as well:
j[10][10] = 0.01
j[11][11] = 0.01
j[12][12] = 0.01
# assume covariance with everything else is 0 which is bad but hopefully gives something to work with at least? (the chains will converge (?))
# computing covariance:
cov = np.linalg.inv(j.T @ j)  # covariance matrix from jacobian sigma = (X^T * X)^(-1)
best = best_fit_jitter.x  # best-fit solution is our center

# initialize walkers
nwalkers = 50  # number of walkers to use in MCMC
ndim = len(best)  # number of dimensions in parameter space
# gaussian ball of 50 walkers with variance equal to cov * 1e-5 and centered on the best-fit solution
pos = np.random.multivariate_normal(best, cov * 1.e-5, size = nwalkers)

# save MCMC sample chain to a file
filename = "mcmc_hd45364_cluster_everything.h5"  # this has everything: rv offset, sin(i), and jitter
backend = emcee.backends.HDFBackend(filename)
backend.reset(nwalkers, ndim)

steps = 50000  # try 50000 steps with multiprocessing on the cluster

# RUNNING MCMC (parallelization):
with InterruptiblePool() as pool:
    sampler = emcee.EnsembleSampler(nwalkers, ndim, log_probability, pool = pool, backend = backend)
    sampler.run_mcmc(pos, steps, progress=True)  # this takes 80 minutes or so to run on laptop

samples = sampler.get_chain()
