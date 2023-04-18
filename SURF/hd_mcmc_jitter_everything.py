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

# params = radvel.Parameters(2,basis='per tc e w k')

# for key, val in params.items():
#     if key in old_params.keys():
#         val.value = old_params[key].value

# params['k1'].value *= 1000
# params['k2'].value *= 1000

# params['curv'] = radvel.Parameter(value = 0)  # initialize these params to avoid the list index error
# params['dvdt'] = radvel.Parameter(value = 0)

# params = params.basis.to_any_basis(params, 'per tc secosw sesinw k')
# mod = radvel.RVModel(params, time_base = hd_data.BJD.median())
# like = radvel.RVLikelihood(mod, hd_data.BJD, hd_data.RV_mlc_nzp, hd_data.e_RV_mlc_nzp)
# like.vector.dict_to_vector()

# post = radvel.posterior.Posterior(like)

# res  = optimize.minimize(
#     post.neglogprob_array,     # objective function is negative log likelihood
#     post.get_vary_params(),    # initial variable parameters
#     method='Nelder-Mead',           # Powell also works
#     )

# post.get_vary_params()

# # SKIPPED THE KEPLERIAN RADVEL MCMC STUFF

# params['rv_offset'] = radvel.Parameter(value='0')  # initialize rv_offset as a parameter and set to 0
# # params['sin_i'] = radvel.Parameter(value='1')  # initialize sin(i) (sin of inclination) and set it to 1
# nbody_params = np.array([])

# for key, item in sorted(list(params.items()), key=lambda x: x[0][-1]):
#     if key not in ['curv', 'gamma', 'jit', 'dvdt']:
#         # maybe there is a more elegant way of doing this
#         nbody_params = np.append(nbody_params, item.value)

# # append 1 to the end and cast nbody_params to arrray of floats
# nbody_params = np.append(nbody_params, 1).astype(float)

# NEW STUFF FOR THE JITTER VERSION OF THIS SCRIPT:

#Very original parameters used in Hadden and Payne
nbody_params =[ 2.27798546e+02,  7.25405874e+00,  5.39392010e+04,  1.71866112e-01, 
               1.17923823e-01,  3.43881599e+02,  1.87692753e+01,  5.40138425e+04, 
               1.68408461e-01,  5.05903191e-02, -3.28526403e-03, 1, 1.84]

#Least squares fit: 
fit_params = [ 2.28512793e+02, 7.27736501e+00, 5.39371914e+04, -4.66868256e-02, 
               -1.78080009e-01, 3.43378038e+02, 1.78603341e+01, 5.40186750e+04, 
               9.72945632e-02,  1.32194117e-01, -5.29072002e-01, 1, 2.428]#-7.68527759e-03] 

# this includes jitter! the last term is taken from the post params with pickle (nbody_params in the original ipynb)



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
    
    params[5 * num_planets] is rv offset
    params[5 * num_planets + 1] is sin(i)
    params[5 * num_planets + 2] is jitter (not used in this specific function but used in some other functions that call this one)
    
    param integrator: integrator to use, one of 'whfast' or 'ias15'
    param time_base: base time (to begin integration from) in the simulation
    """
    
    num_planets = int((len(params) - 1) / 5) # -2 because there are rv_offset and jit parameters:
    
    sim = rb.Simulation()
    sim.integrator = integrator
    sim.t = time_base  # keplerian and n-body models initialized at the same time offset
    # print(sim.t)
    if integrator == 'whfast':  # if using whfast integrator, set timestep
        sim.dt = 1/50 * min(params[0::5][:-1])  # timestep is 1/20th of the shortest orbital period of any planet
        # print(sim.dt)
    sim.units = ('AU', 'Mjupiter', 'day')
    sim.add(m = star_mass)  # star mass as a constant
    
    inclination = np.arcsin(params[-2])  # sin(i) is second from the back of the array
        
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
    params[5 * num_planets + 2] is jitter (also params[-1])
    
    param inclination: inclination of system in the observation plane (pi/2 is in the plane of the sky, 0 is edge-on)
    param times: array of times to integrate over
    param integrator: integrator to use, one of 'whfast' or 'ias15'
    
    """
    
    sim = get_sim_from_params(params, integrator, time_base = time_base)
    
    sim_backwards = sim.copy()
    sim_backwards.dt *= -1  # set timestep to be negative if integrating backwards
    
    forward_times = np.array(list(filter(lambda x: x - time_base >= 0, times)))
    backward_times = np.array(list(filter(lambda x: x - time_base < 0, times)))
    
    # initialize rvs
    rv_forward = np.zeros(len(forward_times))
    rv_backward = np.zeros(len(backward_times))
    
    num_planets = int((len(params) - 1) / 5)  # find number of planets in params passed
    
    # get the rvs (z velocity, assuming 90 deg inclination) from the rebound simulation to compare with the actual simulation
    for i, t in enumerate(forward_times):
        sim.integrate(t, exact_finish_time = 1)
        # integrate to the specified time, exact_finish_time = 1 for ias15, 
        # sim.status()
        star = sim.particles[0]
        rv_forward[i] = (-star.vz * auday_ms) + params[5 * num_planets]  # use x-velocity of the star as the radial velocity, convert to m/s
    
    for i, t in enumerate(backward_times):
        sim_backwards.integrate(t, exact_finish_time = 1)
        star = sim_backwards.particles[0]
        rv_backward[i] = (-star.vz * auday_ms) + params[5 * num_planets]
    
    return np.concatenate((rv_backward, rv_forward))

# OPTIMIZE OVER NEGATIVE LOG LIKELIHOOD INSTEAD OF JUST CHI SQUARED

def neg_log_likelihood(params, data = hd_data):
    """
    Gets the negative log-likelihood (including a jitter term!) for use with scipy.optimize.minimze
    
    Iplements the log likelihood using the same method above
    
    """
    obs_y = data.RV_mlc_nzp  # observed RVs
    jitter = params[-1]  # jitter is at the back of the parameter array, and is handeld separately
    # inclination not handled sparately
    # inclination = np.arcsin(params[-2])  # inclination is np.arcsin of the second to last parameter
    
    synth_y = get_rvs(params, data.BJD, 'ias15', time_base = obs_time_base)  # RVs from the rebound simulation
    obs_yerr = data.e_RV_mlc_nzp  # y errors
    
    log_likelihood = -1/2 * np.sum(((obs_y - synth_y) ** 2)/(obs_yerr ** 2 + jitter ** 2) 
                                   + np.log(np.sqrt(2 * np.pi * (obs_yerr ** 2 + jitter ** 2))))
    
    # log_likelihood = -1/2 * np.sum(np.log(variance) + ((obs_y - synth_y) ** 2/variance))
    
    return -log_likelihood  # negative since we are trying to minimize the negative log likelihood
    
    
# def get_nbody_resids(params, integrator, data = hd_data):
#     """
#     Gets the normalized residuals for the n-body fit with REBOUND
#     """
#     obs_y = data.RV_mlc_nzp  # observed RVs
#     synth_y = get_rvs(params, data.BJD, integrator)  # RVs from the rebound simulation
#     obs_yerr = data.e_RV_mlc_nzp  # y errors
#     return (obs_y - synth_y) / obs_yerr  # return normalized residuals

# # bounds:
# bounds = ([-np.inf, -np.inf, -np.inf, -np.inf, -np.inf, -np.inf, -np.inf, -np.inf, -np.inf, -np.inf, -np.inf, 0],
#           [np.inf, np.inf, np.inf, np.inf, np.inf, np.inf, np.inf, np.inf, np.inf, np.inf, np.inf, 1])  # bounds of [0, 1] for sin(i)
# fit_params = optimize.least_squares(lambda params: get_nbody_resids(params, integrator = 'whfast'), nbody_params, bounds=bounds)

# AGAIN, OPTIMIZE USING OPTIMIZE.MINIMIZE WITH THE LOG LIKELIHOOD INSTEAD OF JUST NORMAL LEAST-SQUARES

# bounds of (0, 1) for sin(i), everything else can vary however
bounds = ((None, None), (None, None), (None, None), (None, None), (None, None), (None, None), (None, None), (None, None), (None, None), (None, None), (None, None), (0, 1) ,(None, None))

best_fit_jitter = optimize.minimize(neg_log_likelihood, x0=np.array(fit_params), method='Nelder-Mead', bounds=bounds, options={'maxiter': int(1e5), 'maxfev': int(1e5)})  # optimization

print(f'original guess:\n{np.array(fit_params)}\n\noptimization with jitter:\n{best_fit_jitter.x}\n\n')

print(best_fit_jitter)


# def get_rvs_dvdt(params, times, integrator, auday_ms=AUDAY_MS):
#     #     # gets rvs using dvdt as a parameter
#     #     if integrator == 'ias15':
#     #         time_base = post.model.time_base
#     #     elif integrator == 'whfast':
#     #         time_base = times[0]
#
#     sim = get_sim_from_params(params, integrator, time_base=post.model.time_base)
#
#     sim_backwards = sim.copy()
#     sim_backwards.dt *= -1  # set timestep to be negative if integrating backwards
#
#     forward_times = np.array(list(filter(lambda x: x - post.model.time_base >= 0, times)))
#     backward_times = np.array(list(filter(lambda x: x - post.model.time_base < 0, times)))
#
#     num_planets = int((len(params) - 2) / 5)  # find number of planets in params passed
#
#     # initialize rvs
#     rv_forward = np.zeros(len(forward_times))
#     rv_backward = np.zeros(len(backward_times))
#
#     # get the rvs (z velocity, assuming 90 deg inclination) from the rebound simulation to compare with the actual simulation
#     for i, t in enumerate(forward_times):
#         sim.integrate(t, exact_finish_time=1)
#         # integrate to the specified time, exact_finish_time = 1 for ias15,
#         # sim.status()
#         star = sim.particles[0]
#         rv_forward[i] = (-star.vz * auday_ms) + params[5 * num_planets] + ((t - post.model.time_base) * params[
#             5 * num_planets + 1])  # use z-velocity of the star as the radial velocity, convert to m/s  # use x-velocity of the star as the radial velocity, convert to m/s
#
#     for i, t in enumerate(backward_times):
#         sim_backwards.integrate(t, exact_finish_time=1)
#         star = sim_backwards.particles[0]
#         rv_backward[i] = (-star.vz * auday_ms) + params[5 * num_planets] + ((t - post.model.time_base) * params[
#             5 * num_planets + 1])  # use z-velocity of the star as the radial velocity, convert to m/s
#
#     return np.concatenate((rv_backward, rv_forward))
#
# def get_nbody_resids_dvdt(params, integrator, inclination = np.pi/2, data = hd_data):
#     """
#     Gets the normalized residuals for the n-body fit with REBOUND
#     """
#     obs_y = data.RV_mlc_nzp  # observed RVs
#     synth_y = get_rvs_dvdt(params, inclination, data.BJD, integrator)  # RVs from the rebound simulation
#     obs_yerr = data.e_RV_mlc_nzp  # y errors
#     return (obs_y - synth_y) / obs_yerr  # return normalized residuals
#
#
# nbody_params_dvdt = np.append(nbody_params, params['dvdt'].value)
# fit_params_dvdt = optimize.least_squares(lambda params: get_nbody_resids_dvdt(params, integrator = 'whfast', inclination = np.pi/2), nbody_params_dvdt)

# print(f'Least squares fit: \n{fit_params_dvdt.x}', '\n\n', f'Original guess: \n {nbody_params_dvdt}')

# CHI2 sin(i) grid search

# sin_i = np.linspace(1, 0.1, 10)  # sini being 0 doesn't work for RV so go to 0.1
# sin_i_chisq_array = []
# k = []
#
# fit_params_chi2 = nbody_params
#
# for inc in sin_i:
#     result = optimize.least_squares(lambda params: get_nbody_resids(params, integrator = 'ias15', inclination = np.arcsin(inc)), fit_params_chi2)
#     # sini_param_array.append([inc, optimize.least_squares(lambda params: get_nbody_resids(params, inclination = np.arcsin(inc)), nbody_params).x])
#     fit_params_chi2 = result.x
#     sin_i_chisq_array.append(result.cost)
#     k.append([fit_params_chi2[1], fit_params_chi2[6]])
#
# minchisq = np.min(sin_i_chisq_array)
# # print(f'min chi2 without dvdt: {minchisq}')
#
# sin_i_chisq_array = []
# k = []

# fit_params_chi2_dvdt = nbody_params_dvdt
#
# for inc in sin_i:
#     result = optimize.least_squares(lambda params: get_nbody_resids_dvdt(params, integrator = 'ias15', inclination = np.arcsin(inc)), fit_params_chi2_dvdt)
#     # sini_param_array.append([inc, optimize.least_squares(lambda params: get_nbody_resids(params, inclination = np.arcsin(inc)), nbody_params).x])
#     fit_params_chi2_dvdt = result.x
#     sin_i_chisq_array.append(result.cost)
#     k.append([fit_params_chi2_dvdt[1], fit_params_chi2_dvdt[6]])
#
# minchisq = np.min(sin_i_chisq_array)
# # print(f'min chi2 with dvdt: {minchisq}')

# optimize again 
jitter_min_neg_log_likelihood = 2.42242242242242  # jitter producing the minimum negative log likelihood when doing the grid search, found manually in the ipynb

best_fit_jitter2 = optimize.minimize(neg_log_likelihood, x0=np.append(best_fit_jitter.x[:-1], jitter_min_neg_log_likelihood), method='Nelder-Mead', bounds=bounds, options={'maxiter': int(1e5), 'maxfev': int(1e5)})  # optimization

# MIGRATION/ECCENTRICITY DAMPING

# nsims = int(1e4)
# sim = get_sim_from_params(nbody_params, inclination = np.pi/2, integrator = 'whfast', time_base = 0)
#
# inner = sim.particles[1]  # inner and outer planets in our simulation
# outer = sim.particles[2]
#
# a_inner, a_outer, p_inner, p_outer = np.zeros(nsims), np.zeros(nsims), np.zeros(nsims), np.zeros(nsims)
# e_inner, e_outer = np.zeros(nsims), np.zeros(nsims)
#
# rebx = reboundx.Extras(sim)
# rebx.add_force(rebx.load_force('modify_orbits_forces'))
# rebx.add_operator(rebx.load_operator('modify_orbits_direct'))
#
# t_end = int(3e4)
#
# outer.params['tau_a'] = -10 * t_end
# inner.params['tau_e'] = -t_end/5  # eccentricity damping, these stay constant
# outer.params['tau_e'] = -t_end/10

# sim.automateSimulationArchive("hd_migration.bin",interval=10,deletefile=True)
#
# migration_times = np.linspace(0, t_end, nsims)
# for ind, time in enumerate(migration_times):
#     sim.integrate(time)
#     a_inner[ind] = inner.a
#     a_outer[ind] = outer.a
#     p_inner[ind] = inner.P
#     p_outer[ind] = outer.P
#     e_inner[ind] = inner.e
#     e_outer[ind] = outer.e
#
# period_ratio = p_outer/p_inner
#
# a_inner, a_outer, p_inner, p_outer = np.zeros(nsims), np.zeros(nsims), np.zeros(nsims), np.zeros(nsims)
# e_inner, e_outer = np.zeros(nsims), np.zeros(nsims)

# COMMENTING THIS OUT BECAUSE I DON'T THINK I NEED IT?

# def get_tau_alphas(tau_alpha, m_inner, m_outer, period_ratio):
#     # use Kepler's third law to compute the ratio of semi-major axes in resonance from the period ratio in resonance
#     sma_ratio = period_ratio ** (2 / 3)  # ratio of outer planet's semi-major axis to inner
#     # define matrix A
#     A = np.array([[-1, 1],
#                   [m_outer, m_inner * sma_ratio]])
#     # compute gamma_1 and gamma_2
#     gammas = np.matmul(np.linalg.inv(A), np.array([-1 / tau_alpha, 0]))
#     # gamma = 1/tau
#     taus = 1 / gammas

#     return tuple(taus)  # returns (tau_alpha_outer, tau_alpha_inner) as a tuple

# # t_end = 3e4
# # nsims = int(100)

# def get_simple_sim(masses):
#     sim = rb.Simulation()
#     # central star
#     sim.add(m=1)

#     sim.add(m=masses[0], P=1)
#     sim.add(m=masses[1], P=1.5 * 1.01)

#     sim.move_to_com()
#     sim.integrator = 'whfast'
#     sim.dt = 1 / 50 * 1

#     return sim

#
# sim = get_sim_from_params(fit_params_chi2.x, inclination = np.pi/2, integrator = 'whfast', time_base = 0)
# masses = np.array([sim.particles[1].m, sim.particles[2].m])/sim.particles[0].m
#
# Ks = np.logspace(0, 2.5, 10)
# for ind, K in enumerate(Ks):
#     sim = get_simple_sim(masses)
#
#     inner = sim.particles[1]  # inner and outer planets in our simulation
#     outer = sim.particles[2]
#
#     rebx = reboundx.Extras(sim)
#     rebx.add_force(rebx.load_force('modify_orbits_forces'))
#     rebx.add_operator(rebx.load_operator('modify_orbits_direct'))
#
#     # outer.params['tau_a'] = -10 * t_end
#     inner.params['tau_e'] = -t_end / 3  # eccentricity damping, these stay constant
#     outer.params['tau_e'] = -t_end / 3
#
#     tau_e = 1 / (1 / inner.params['tau_e'] + 1 / outer.params['tau_e'])
#     ####
#
#     # set the semi-major axis damping for inner and outer planets
#     outer.params['tau_a'], inner.params['tau_a'] = get_tau_alphas(K * tau_e, inner.m, outer.m,
#                                                                   period_ratio=3 / 2)  # 3/2 period ratio
#     # print(outer.params['tau_a'], inner.params['tau_a'])
#     sim.integrate(t_end)
#
#     e_inner[ind] = inner.e
#     e_outer[ind] = outer.e
#
#     # integrate to t_end to find the equilibrium eccentricities
#
# sim = get_sim_from_params(fit_params_chi2.x, inclination = np.pi/2, integrator = 'whfast', time_base = 0)
# es = np.array([sim.particles[1].e, sim.particles[2].e])

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
    sin_i = params[-2]  # sin(i) is the second-to-last item of the array

    # uniform log prior, return 0 if param falls within uniform distribution, -infinity otherwise
    # print(ps, ks, tcs, es, omega)

    if all(p > 0 for p in ps) and all(k > 0 for k in ks) and all(tc > 0 for tc in tcs) and all(0 < e < e_max for e in es) and (sin_i_min < sin_i <= 1.):
        return 0.0  # log prior, so ln(1) = 0
    else:
        return -np.inf  # log prior, so ln(0) = -infinity

# LOG LIKELIHOOD
def log_likelihood(params, data = hd_data):
    """
    Gets the log-likelihood (negative of the negative log likelihood) (including a jitter term!) for use with scipy.optimize.minimze
    
    Implements the log likelihood using the same method as neg_log_likelihood above
    
    """
    obs_y = data.RV_mlc_nzp  # observed RVs
    jitter = params[-1]  # jitter is at the back of the parameter array, and is handeld separately
    # inclination not handled sparately
    # inclination = np.arcsin(params[-2])  # inclination is np.arcsin of the second to last parameter
    
    synth_y = get_rvs(params, data.BJD, 'ias15', time_base = obs_time_base)  # RVs from the rebound simulation
    obs_yerr = data.e_RV_mlc_nzp  # y errors
    
    log_likelihood = -1/2 * np.sum(((obs_y - synth_y) ** 2)/(obs_yerr ** 2 + jitter ** 2) 
                                   + np.log(np.sqrt(2 * np.pi * (obs_yerr ** 2 + jitter ** 2))))
    
    # log_likelihood = -1/2 * np.sum(np.log(variance) + ((obs_y - synth_y) ** 2/variance))
    
    return log_likelihood  # positive in this case, unlike the other


# LOG PROBABILITY
def log_probability(params):
    lp = log_prior(params)
    if not np.isfinite(lp):
        return -np.inf
    return lp + log_likelihood(params)

# THIS IS THE "BEST" SOLUTION TO INITIALIZE THE MCMC FROM:
print(best_fit_jitter2)


# DO LEAST-SQUARES OPTIMIZATION HERE TO GET A JACOBIAN FOR THE NEXT PART
# The actual least-squares "fit" will probably be slightly worse compared to the fit using negative log-likelihodo (because it doesn't take into account jitter), but hopefully the jacobian is close enough to just use with the original best fit (the one computed using negloglikelihood instead of least-squares) to get a decent starting position for the walkers

def get_nbody_resids(params, integrator, data = hd_data):
    """
    Gets the normalized residuals for the n-body fit with REBOUND
    """
    obs_y = data.RV_mlc_nzp  # observed RVs
    synth_y = get_rvs(params, data.BJD, integrator, time_base=obs_time_base)  # RVs from the rebound simulation
    obs_yerr = data.e_RV_mlc_nzp  # y errors
    return (obs_y - synth_y) / obs_yerr  # return normalized residuals


# get the "fit params" to get the jacobian from this:
bounds2 = ([-np.inf, -np.inf, -np.inf, -np.inf, -np.inf, -np.inf, -np.inf, -np.inf, -np.inf, -np.inf, -np.inf, 0, -np.inf],
          [np.inf, np.inf, np.inf, np.inf, np.inf, np.inf, np.inf, np.inf, np.inf, np.inf, np.inf, 1, np.inf])
jacobian_fit_params = optimize.least_squares(lambda params: get_nbody_resids(params, integrator='ias15'), best_fit_jitter2.x, bounds=bounds2)

# COVARIANCE MATRIX CALCULATIONS
j = jacobian_fit_params.jac  # jacobian, value for jitter is 0 though so hopefully we can manually patch the jacobian to get something reasonable

# MANUALLY PATCHING JITTER FOR THE JACOBIAN:
j[-1][-1] = 0.321  # empiriclaly determined value of onesigma_jit in the hd45364_jitter_sini notebook, about one sigma value of jitter 
# assume covariance with everything else is 0 which is bad but hopefully gives something to work with at least? (the chains will converge (?))

# computing covariance:
cov = np.linalg.inv(j.T @ j)  # covariance matrix from jacobian sigma = (X^T * X)^(-1)
best = best_fit_jitter2.x  # best-fit solution is our center

# initialize walkers
nwalkers = 50  # number of walkers to use in MCMC
ndim = len(best)  # number of dimensions in parameter space
# gaussian ball of 50 walkers with variance equal to cov * 1/100 and centered on the best-fit solution
pos = np.random.multivariate_normal(best, cov * 1/100, size = nwalkers)

# save MCMC sample chain to a file
filename = "mcmc_hd45364_cluster_everything.h5"  # this has everything: rv offset, sin(i), and jitter
backend = emcee.backends.HDFBackend(filename)
backend.reset(nwalkers, ndim)

steps = 50000  # try 50000 steps with multiprocessing on the cluster

# RUNNING MCMC (parallelization):

with Pool() as pool:
    sampler = emcee.EnsembleSampler(nwalkers, ndim, log_probability, pool = pool, backend = backend)
    sampler.run_mcmc(pos, steps, progress=True)  # this takes 80 minutes or so to run on laptop

samples = sampler.get_chain()

# e1 = samples[:, :, 3] ** 2 + samples[:, :, 4] ** 2
# e2 = samples[:, :, 8] ** 2 + samples[:, :, 9] ** 2
#
# flat_samples = sampler.get_chain(discard = 200, flat = True)  # arbitrary discard numbers
#
# # best fit of mcmc by log probability:
#
# best_fit_loc = np.argmax(sampler.flatlnprobability)
# best_params_mcmc = sampler.flatchain[best_fit_loc]
#
# resids1 = get_nbody_resids(best_params_mcmc, 'ias15')  # chi squared from mcmc best fit params
# resids2 = get_nbody_resids(best, 'ias15')  # chi squared from starting guess (using least squares and chi square optimization)
# resids3 = get_nbody_resids(fit_params.x, 'ias15')  # chi squared from least squares without chi square optimization
#
# # [(resids @ resids)/2 for resids in [resids1, resids2, resids3]]
#
# sim1 = get_sim_from_params(best_params_mcmc, np.pi/2, 'ias15', post.model.time_base)
# angle_times = np.linspace(0, 5000 * sim1.particles[1].P, 200) + sim1.t
# e1, e2 = np.zeros(len(angle_times)), np.zeros(len(angle_times))
# angle1, angle2 = np.zeros(len(angle_times)), np.zeros(len(angle_times))
#
# for i, t in enumerate(angle_times):
#     sim1.integrate(t)
#     e1[i] = sim1.particles[1].e
#     e2[i] = sim1.particles[2].e
#     resonant_angle = 3 * sim1.particles[2].l - 2 * sim1.particles[1].l
#     angle1[i] = resonant_angle - sim1.particles[1].pomega
#     angle2[i] = resonant_angle - sim1.particles[2].pomega
#
# angle1 = np.mod(angle1, 2 * np.pi)
# angle2 = np.mod(angle2, 2 * np.pi)