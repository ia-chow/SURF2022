import pickle
import radvel
import rebound as rb
import pandas as pd
import numpy as np
import matplotlib.pyplot as pl
from scipy import optimize, stats
import corner
from radvel.plot import mcmc_plots
import reboundx
import emcee
from multiprocessing import Pool
import h5py
import time
from celmech.nbody_simulation_utilities import align_simulation
from tqdm import tqdm
from rebound.interruptible_pool import InterruptiblePool


with open("../init_params_hd45364.bin", 'rb') as file:
    old_params = pickle.load(file)

hd_data = pd.read_csv('../hd45364_rvs.csv', sep =';')
hd_data.BJD -= 2.4e6

params = radvel.Parameters(2,basis='per tc e w k')

for key, val in params.items():
    if key in old_params.keys():
        val.value = old_params[key].value

params['k1'].value *= 1000
params['k2'].value *= 1000

params['curv'] = radvel.Parameter(value = 0)  # initialize these params to avoid the list index error
params['dvdt'] = radvel.Parameter(value = 0)

params = params.basis.to_any_basis(params, 'per tc secosw sesinw k')
mod = radvel.RVModel(params, time_base = hd_data.BJD.median())
like = radvel.RVLikelihood(mod, hd_data.BJD, hd_data.RV_mlc_nzp, hd_data.e_RV_mlc_nzp)
like.vector.dict_to_vector()

post = radvel.posterior.Posterior(like)

res  = optimize.minimize(
    post.neglogprob_array,     # objective function is negative log likelihood
    post.get_vary_params(),    # initial variable parameters
    method='Nelder-Mead',           # Powell also works
    )

post.get_vary_params()

# SKIPPED THE KEPLERIAN RADVEL MCMC STUFF

params['rv_offset'] = radvel.Parameter(value='0')  # initialize rv_offset as a parameter and set to 0
nbody_params = np.array([])

for key, item in sorted(list(params.items()), key=lambda x: x[0][-1]):
    if key not in ['curv', 'gamma', 'jit', 'dvdt']:
        # maybe there is a more elegant way of doing this
        nbody_params = np.append(nbody_params, item.value)

# cast nbody_params to arrray of floats
nbody_params = nbody_params.astype(float)

## SKIPPING ALL THE PLOTTING AS WELL

## CONSTANTS:

STAR_MASS = 920  # 920 jupiter masses
G = 2.825e-7  # converting G to jupiter masses, au, and days
AUDAY_MS = 1.731e6  # conversion factor for au/day to m/s


def mass_to_semiamp(planet_mass, star_mass, period, eccentricity, inclination):
    """
    planet mass to semi amplitude
    """
    return ((2 * np.pi * G / period) ** (1 / 3) * (planet_mass * np.sin(inclination) / star_mass ** (2 / 3)) * (
                1 / np.sqrt(1 - eccentricity ** 2)))


def semiamp_to_mass(semiamp, star_mass, period, eccentricity, inclination):
    """
    semi amplitude to planet mass
    """
    return (((2 * np.pi * G / period) ** (-1 / 3)) * (semiamp / np.sin(inclination)) * np.sqrt(
        1 - eccentricity ** 2) * (star_mass ** (2 / 3)))


def get_sim_from_params(params, inclination, integrator, time_base, star_mass=STAR_MASS, auday_ms=AUDAY_MS):
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
    params[5 * num_planets + 1] is jit

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


def get_rvs(params, inclination, times, integrator, auday_ms=AUDAY_MS):
    """
    Gets RVs from a Numpy array of planet params

    param params:     for i in range(0, num_planets):

    params[i + 0] is period
    params[i + 1] is semiamp
    params[i + 2] is tc (time of conjunction)
    params[i + 3] is sqrt(e) * cos(omega)
    params[i + 4] is sqrt(e) * sin(omega)

    params[5 * num_planets] is rv offset

    param inclination: inclination of system in the observation plane (pi/2 is in the plane of the sky, 0 is edge-on)
    param times: array of times to integrate over
    param integrator: integrator to use, one of 'whfast' or 'ias15'

    """

    sim = get_sim_from_params(params, inclination, integrator, time_base=post.model.time_base)

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
        rv_forward[i] = (-star.vz * auday_ms) + params[
            5 * num_planets]  # use x-velocity of the star as the radial velocity, convert to m/s

    for i, t in enumerate(backward_times):
        sim_backwards.integrate(t, exact_finish_time=1)
        star = sim_backwards.particles[0]
        rv_backward[i] = (-star.vz * auday_ms) + params[5 * num_planets]

    return np.concatenate((rv_backward, rv_forward))


def get_nbody_resids(params, integrator, inclination = np.pi/2, data = hd_data):
    """
    Gets the normalized residuals for the n-body fit with REBOUND
    """
    obs_y = data.RV_mlc_nzp  # observed RVs
    synth_y = get_rvs(params, inclination, data.BJD, integrator)  # RVs from the rebound simulation
    obs_yerr = data.e_RV_mlc_nzp  # y errors
    return (obs_y - synth_y) / obs_yerr  # return normalized residuals

fit_params = optimize.least_squares(lambda params: get_nbody_resids(params, integrator = 'whfast', inclination = np.pi/2), nbody_params)


def get_rvs_dvdt(params, inclination, times, integrator, auday_ms=AUDAY_MS):
    #     # gets rvs using dvdt as a parameter
    #     if integrator == 'ias15':
    #         time_base = post.model.time_base
    #     elif integrator == 'whfast':
    #         time_base = times[0]

    sim = get_sim_from_params(params, inclination, integrator, time_base=post.model.time_base)

    sim_backwards = sim.copy()
    sim_backwards.dt *= -1  # set timestep to be negative if integrating backwards

    forward_times = np.array(list(filter(lambda x: x - post.model.time_base >= 0, times)))
    backward_times = np.array(list(filter(lambda x: x - post.model.time_base < 0, times)))

    num_planets = int((len(params) - 2) / 5)  # find number of planets in params passed

    # initialize rvs
    rv_forward = np.zeros(len(forward_times))
    rv_backward = np.zeros(len(backward_times))

    # get the rvs (z velocity, assuming 90 deg inclination) from the rebound simulation to compare with the actual simulation
    for i, t in enumerate(forward_times):
        sim.integrate(t, exact_finish_time=1)
        # integrate to the specified time, exact_finish_time = 1 for ias15,
        # sim.status()
        star = sim.particles[0]
        rv_forward[i] = (-star.vz * auday_ms) + params[5 * num_planets] + ((t - post.model.time_base) * params[
            5 * num_planets + 1])  # use z-velocity of the star as the radial velocity, convert to m/s  # use x-velocity of the star as the radial velocity, convert to m/s

    for i, t in enumerate(backward_times):
        sim_backwards.integrate(t, exact_finish_time=1)
        star = sim_backwards.particles[0]
        rv_backward[i] = (-star.vz * auday_ms) + params[5 * num_planets] + ((t - post.model.time_base) * params[
            5 * num_planets + 1])  # use z-velocity of the star as the radial velocity, convert to m/s

    return np.concatenate((rv_backward, rv_forward))

def get_nbody_resids_dvdt(params, integrator, inclination = np.pi/2, data = hd_data):
    """
    Gets the normalized residuals for the n-body fit with REBOUND
    """
    obs_y = data.RV_mlc_nzp  # observed RVs
    synth_y = get_rvs_dvdt(params, inclination, data.BJD, integrator)  # RVs from the rebound simulation
    obs_yerr = data.e_RV_mlc_nzp  # y errors
    return (obs_y - synth_y) / obs_yerr  # return normalized residuals

# nbody_params_dvdt = np.append(nbody_params, params['dvdt'].value)
# fit_params_dvdt = optimize.least_squares(lambda params: get_nbody_resids_dvdt(params, integrator = 'whfast', inclination = np.pi/2), nbody_params_dvdt)
# # print(f'Least squares fit: \n{fit_params_dvdt.x}', '\n\n', f'Original guess: \n {nbody_params_dvdt}')

# CHI2 sin(i) grid search

sin_i = np.linspace(1, 0.1, 10)  # sini being 0 doesn't work for RV so go to 0.1
sin_i_chisq_array = []
k = []

fit_params_chi2 = nbody_params

for inc in sin_i:
    result = optimize.least_squares(lambda params: get_nbody_resids(params, integrator = 'ias15', inclination = np.arcsin(inc)), fit_params_chi2)
    # sini_param_array.append([inc, optimize.least_squares(lambda params: get_nbody_resids(params, inclination = np.arcsin(inc)), nbody_params).x])
    fit_params_chi2 = result.x
    sin_i_chisq_array.append(result.cost)
    k.append([fit_params_chi2[1], fit_params_chi2[6]])

# minchisq = np.min(sin_i_chisq_array)
# print(f'min chi2 without dvdt: {minchisq}')


# dvdt stuff

# sin_i_chisq_array = []
# k = []
#
# fit_params_chi2_dvdt = nbody_params_dvdt
#
# for inc in sin_i:
#     result = optimize.least_squares(lambda params: get_nbody_resids_dvdt(params, integrator = 'ias15', inclination = np.arcsin(inc)), fit_params_chi2_dvdt)
#     # sini_param_array.append([inc, optimize.least_squares(lambda params: get_nbody_resids(params, inclination = np.arcsin(inc)), nbody_params).x])
#     fit_params_chi2_dvdt = result.x
#     sin_i_chisq_array.append(result.cost)
#     k.append([fit_params_chi2_dvdt[1], fit_params_chi2_dvdt[6]])


minchisq = np.min(sin_i_chisq_array)
# print(f'min chi2 with dvdt: {minchisq}')
fit_params_chi2 = optimize.least_squares(lambda params: get_nbody_resids(params, integrator = 'ias15', inclination = np.arcsin(1)), fit_params.x)

# MIGRATION/ECCENTRICITY DAMPING

nsims = int(1e3)
sim = get_sim_from_params(nbody_params, inclination = np.pi/2, integrator = 'whfast', time_base = 0)

inner = sim.particles[1]  # inner and outer planets in our simulation
outer = sim.particles[2]

a_inner, a_outer, p_inner, p_outer = np.zeros(nsims), np.zeros(nsims), np.zeros(nsims), np.zeros(nsims)
e_inner, e_outer = np.zeros(nsims), np.zeros(nsims)

rebx = reboundx.Extras(sim)
rebx.add_force(rebx.load_force('modify_orbits_forces'))
rebx.add_operator(rebx.load_operator('modify_orbits_direct'))

t_end = int(3e4)

outer.params['tau_a'] = -10 * t_end
inner.params['tau_e'] = -t_end/5  # eccentricity damping, these stay constant
outer.params['tau_e'] = -t_end/10

# sim.automateSimulationArchive("hd_migration.bin",interval=10,deletefile=True)

migration_times = np.linspace(0, t_end, nsims)
for ind, time in enumerate(migration_times):
    sim.integrate(time)
    a_inner[ind] = inner.a
    a_outer[ind] = outer.a
    p_inner[ind] = inner.P
    p_outer[ind] = outer.P
    e_inner[ind] = inner.e
    e_outer[ind] = outer.e

period_ratio = p_outer/p_inner

a_inner, a_outer, p_inner, p_outer = np.zeros(nsims), np.zeros(nsims), np.zeros(nsims), np.zeros(nsims)
e_inner, e_outer = np.zeros(nsims), np.zeros(nsims)


def get_tau_alphas(tau_alpha, m_inner, m_outer, period_ratio):
    # use Kepler's third law to compute the ratio of semi-major axes in resonance from the period ratio in resonance
    sma_ratio = period_ratio ** (2 / 3)  # ratio of outer planet's semi-major axis to inner
    # define matrix A
    A = np.array([[-1, 1],
                  [m_outer, m_inner * sma_ratio]])
    # compute gamma_1 and gamma_2
    gammas = np.matmul(np.linalg.inv(A), np.array([-1 / tau_alpha, 0]))
    # gamma = 1/tau
    taus = 1 / gammas

    return tuple(taus)  # returns (tau_alpha_outer, tau_alpha_inner) as a tuple

t_end = 3e4
# nsims = int(100)


def get_simple_sim(masses, integrator='ias15'):
    sim = rb.Simulation()
    # central star
    sim.add(m=1)
    # planets
    sim.add(m=masses[0], P=1)
    sim.add(m=masses[1], P=1.5 * 1.01)

    sim.move_to_com()

    # integrator
    if integrator == 'whfast':
        sim.dt = 1 / 50 * 1

    return sim

# migratino stuff
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

# LOG PRIOR
def log_prior(params):
    ps = params[0:-3:5]  # start at 0, the last 3 elements of params are not planet params (gamma, jitter, rv offset)
    ks = params[1:-3:5]
    tcs = params[2:-3:5]
    # compute e and omega from secos, sesin
    es = params[3:-3:5] ** 2 + params[4:-3:5] ** 2  # eccentricity from secos, sesin
    omega = np.arctan2(params[3:-3:5], params[4:-3:5])  # omega from arctan of sesin, secos
    # uniform log prior, return 0 if param falls within uniform distribution, -infinity otherwise
    # print(ps, ks, tcs, es, omega)

    if all(p > 0 for p in ps) and all(k > 0 for k in ks) and all(tc > 0 for tc in tcs) and all(0 < e < 0.8 for e in es):
        return 0.0  # log prior, so ln(1) = 0
    else:
        return -np.inf  # log prior, so ln(0) = -infinity

# LOG LIKELIHOOD
def log_likelihood(params, inclination=np.pi / 2, data=hd_data):
    """
    Gets the log-likelihood
    """
    obs_y = data.RV_mlc_nzp  # observed RVs
    synth_y = get_rvs(params, inclination, data.BJD, 'ias15')  # RVs from the rebound simulation
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
# j = fit_params_chi2.jac  # jacobian
# cov = np.linalg.inv(j.T @ j)  # covariance matrix from jacobian sigma = (X^T * X)^(-1)
# best = fit_params_chi2.x  # best-fit solution is our center
#
# # initialize walkers
# nwalkers = 50  # number of walkers to use in MCMC
# ndim = len(fit_params_chi2.x)  # number of dimensions in parameter space
# # gaussian ball of 50 walkers with variance equal to cov * 1/100 and centered on the best-fit solution
# pos = np.random.multivariate_normal(best, cov * 1/100, size = nwalkers)

# # save MCMC sample chain to a file
# filename = "mcmc_hd45364.h5"
# backend = emcee.backends.HDFBackend(filename)
# backend.reset(nwalkers, ndim)
#
# steps = 50000  # try 50000 stes with multiprocessing on the cluster
#
# # RUNNING MCMC (parallelization):
#
# with Pool() as pool:
#     sampler = emcee.EnsembleSampler(nwalkers, ndim, log_probability, pool = pool, backend = backend)
#     sampler.run_mcmc(pos, steps, progress=True)  # this takes 80 minutes or so to run on laptop
#
# samples = sampler.get_chain()
#
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


# STARTING MULTIPROCESSING FOR MAX MASSES:
print('STARTING MULTIPROCESSING FOR MAX MASSES:')

ecc_sim = get_sim_from_params(fit_params_chi2.x, inclination = np.pi/2, integrator = 'whfast', time_base = 0)
planet_masses = np.array([ecc_sim.particles[1].m, ecc_sim.particles[2].m])/ecc_sim.particles[0].m
nsims = 100  # 100 simulations for eccentricity tracks/max masses

best_fit_es = np.array([ecc_sim.particles[1].e, ecc_sim.particles[2].e])

# Ks = np.logspace(0, 5, nsims)


def get_eccentricity_tracks(params, period_ratio=3/2, t_end=3e4, tau_e_factor=3):
    """
    Gets planet eccentricity tracks for a simulation with two planets. Assumes that planets are coplanar
    and are in MMR with ratio period_ratio

    params: tuple of (masses, Ks, filename):
    masses is an array of planet masses
    K is the tau_a/tau_e ratio for this eccentricity track value (e1, e2)
    filename is the filename to save the simulation archive to
    """

    masses, K, filename = params
    # K = params[1]
    # filename = params[2]

    print(f'On ecc sim for {filename}')  # debug print statement to see which sim it's on

    sim = get_simple_sim(masses)

    inner = sim.particles[1]  # inner and outer planets in our simulation
    outer = sim.particles[2]

    rebx = reboundx.Extras(sim)  # eccentricity and migration forces
    rebx.add_force(rebx.load_force('modify_orbits_forces'))
    rebx.add_operator(rebx.load_operator('modify_orbits_direct'))

    # outer.params['tau_a'] = -10 * t_end
    inner.params['tau_e'], outer.params[
        'tau_e'] = -t_end / tau_e_factor, -t_end / tau_e_factor  # eccentricity damping, these stay constant

    tau_e = 1 / (1 / inner.params['tau_e'] + 1 / outer.params['tau_e'])

    # set the semi-major axis damping for inner and outer planets
    outer.params['tau_a'], inner.params['tau_a'] = get_tau_alphas(K * tau_e, inner.m, outer.m,
                                                                  period_ratio=period_ratio)  # 3/2 period ratio

    sim.integrate(t_end)  # integrate to t_end to find the equilibrium eccentricities
    sim.save(filename)  # save the simulations each to their own file

    e1_track_val = inner.e
    e2_track_val = outer.e

    return e1_track_val, e2_track_val  # arrays of length Ks

# e1_track, e2_track = np.zeros(len(Ks)), np.zeros(len(Ks))

# par_list_ecc = []
# for i, k in enumerate(Ks):
#     filename_ecc = f'ecc_track_archive_cluster_{i}.bin'  # each value of k gets its own file
#     par_list_ecc.append((planet_masses, k, filename_ecc))
#
# # parallelize it
# pool_ecc = InterruptiblePool()
# ecc_track_results = pool_ecc.map(get_eccentricity_tracks, par_list_ecc)  # get max m1s, m2x


# for i in range(0, len(ecc_archive)):
#     ecc_sim = ecc_archive[i]
#     inner = ecc_sim.particles[1]
#     outer = ecc_sim.particles[2]
#     e1_track[i] = inner.e
#     e2_track[i] = outer.e

# get eccentricities
# e1_track, e2_track = get_eccentricity_tracks(planet_masses, Ks, period_ratio = 3/2)  # takes about 3 minutes to do

# pl.figure(figsize = (10, 7))
# pl.plot(e1_track, e2_track, '-o', label = 'eccentricity track')  # track
# pl.scatter(best_fit_es[0], best_fit_es[1], c = 'r', label = 'best fit')  # best fit eccentricity
# pl.xlabel('e1'), pl.ylabel('e2')
# pl.legend(loc='upper left')

# MAXIMUM MASSES:


def get_max_masses(params, period_ratio=3/2, t_end=3e5, tau_e_factor=3):
    """
    params: tuple of params (masses, K, filename)
    masses is an array of planet masses
    K is the tau_a/tau_e ratio for this case
    filename is the file to save the simulationarchive to
    """
    # unpack params:
    masses, K, filename = params # [0]
    # K = params[1]
    # filename = params[2]

    print(f'On mass sim for {filename}')  # another debug print statement osee which sim it's on

    # first do migration/damping to get them in equilibrium
    sim = get_simple_sim(masses)

    inner = sim.particles[1]  # inner and outer planets in our simulation
    outer = sim.particles[2]

    rebx = reboundx.Extras(sim)  # eccentricity and migration forces
    rebx.add_force(rebx.load_force('modify_orbits_forces'))
    rebx.add_operator(rebx.load_operator('modify_orbits_direct'))

    # tau_e
    inner.params['tau_e'], outer.params[
        'tau_e'] = -t_end / tau_e_factor, -t_end / tau_e_factor  # eccentricity damping, these stay constant
    tau_e = 1 / (1 / inner.params['tau_e'] + 1 / outer.params['tau_e'])

    # set the semi-major axis damping for inner and outer planets

    # tau_a
    outer.params['tau_a'], inner.params['tau_a'] = get_tau_alphas(K * tau_e, inner.m, outer.m,
                                                                  period_ratio=period_ratio)  # 3/2 period ratio

    sim.integrate(t_end)  # integrate to t_end to find the equilibrium eccentricities

    # now create a new sim with the equlibirum positions and turn up the masses until there is a close encounter
    new_sim = sim.copy()
    sim = None  # free up memory

    rebx = reboundx.Extras(new_sim)
    rebx.add_operator(rebx.load_operator("modify_mass"))

    inner = new_sim.particles[1]
    outer = new_sim.particles[2]

    inner.params['tau_mass'], outer.params['tau_mass'] = 1e5, 1e5
    new_sim.exit_min_distance = 5 * inner.rhill  # 5 times the hill distance as a close encounter

    try:
        new_sim.integrate(
            1e7)  # doesn't matter what this number is since eventually there will be a close encounter
        # as long as it's large enough so the encounter happens eventually
    except rb.Encounter as error:
        pass

    new_sim.save(filename)  # again save each simulation to its own file

    # max masses
    max_m1s_k = inner.m * STAR_MASS
    max_m2s_k = outer.m * STAR_MASS
    # since masses are jupiter masses/star masses
    # to get the max masses in jupiter masses

    return max_m1s_k, max_m2s_k


# save sims again
# par_list = []
# for i, k in enumerate(Ks):
#     filename_mass = f'max_mass_archive_cluster_{i}.bin'
#     par_list.append((planet_masses, k, filename_mass))
#
# pool = InterruptiblePool()
# max_mass_results = pool.map(get_max_masses, par_list)  # get max m1s, m2x

# and for 10^3:

# Ks_102 = np.linspace(65, 90, nsims)
# filename_102 = 'max_mass_archive_102_cluster.bin'
#

Ks_103 = np.logspace(0, 3, nsims)
par_list_103 = []
for i, k in enumerate(Ks_103):
    filename_mass_103 = f'max_mass_archive_cluster_103_{i}.bin'
    par_list_103.append((planet_masses, k, filename_mass_103))

pool = InterruptiblePool()
max_mass_results = pool.map(get_max_masses, par_list_103)  # get max m1s, m2x

# par_list_102 = []
# for i, k in enumerate(Ks_102):
#     par_list_102.append((planet_masses, k, filename_102))
#
# pool_102 = InterruptiblePool()
# max_mass_results_102 = pool_102.map(get_max_masses, par_list_102)
