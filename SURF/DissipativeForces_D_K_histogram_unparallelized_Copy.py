# IMPORTS

import rebound as rb
from matplotlib import pyplot as plt
import celmech as cm
import numpy as np
import sympy as sp
import radvel
import pandas as pd
import h5py
from multiprocessing import Pool
from rebound.interruptible_pool import InterruptiblePool

plt.rc('font', size=18)  # fontsize larger

# get the data and convert it to flat samples first:
hd_data = pd.read_csv('hd45364_rvs.csv', sep = ';')
# giant outlier at position 116 in the data (found manually earlier) which we remove
hd_data.drop(116, inplace=True)  # drop the row and keep the df in place
# subtract 2.4e6 from all the rows in the data
hd_data.BJD -= 2.4e6
cluster_data = h5py.File('2021_fall-2022_winter/mcmc_hd45364_cluster_everything.h5', 'r')  # import the posterior distribution data
accepted, samples, log_prob = np.array(cluster_data['mcmc']['accepted']), np.array(cluster_data['mcmc']['chain']), np.array(cluster_data['mcmc']['log_prob'])
n_burn_in = 200  # discard the first 200 samples as burn-in time
# reshape the chain to flatten it out
flat_samples = samples[n_burn_in:].reshape(-1, samples[n_burn_in:].shape[-1])

# functions for the d/k value computations and other setup

#Least squares fit: 
fit_params = [ 2.28512793e+02, 7.27736501e+00, 5.39371914e+04, -4.66868256e-02, 
               -1.78080009e-01, 3.43378038e+02, 1.78603341e+01, 5.40186750e+04, 
               9.72945632e-02,  1.32194117e-01, -5.29072002e-01, 1, 2.428]#-7.68527759e-03] 

# star mass, g and auday to m/s conversion factor
STAR_MASS = 920  # 920 jupiter masses
G = 2.825e-7  # converting G to jupiter masses, au, and days
AUDAY_MS = 1.731e6  # conversion factor for au/day to m/s

# use median of time data as the time base:
obs_time_base = np.median(hd_data.BJD)

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
    param masses: array of planet masses [in units of Mstar]
    param integrator: integrator
    param epsilon: amount by which the resonant period ratio should be offset from the equilibrium in the simulation
    """
    sim = rb.Simulation()
    sim.integrator = integrator
    # central star
    sim.add(m = 1)  # central star has mass of 1, so the planet masses are in units of Mstar
    
    sim.add(m = masses[0], P = 1)
    sim.add(m = masses[1], P = period_ratio * (1 + epsilon))

    sim.move_to_com()
    if integrator == 'whfast':
        sim.dt = 1/50 * 1  # dy default use 1/50th of the inner planet's orbital period for the timestep if using whfast
    return sim


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

# set up the cts
gsim = get_sim_from_params(fit_params, integrator= 'ias15', time_base = obs_time_base)
masses = np.array([gsim.particles[1].m, gsim.particles[2].m])/gsim.particles[0].m  # divide by star mass!
sim = get_simple_sim(masses, integrator = 'ias15')
pvars = cm.Poincare.from_Simulation(sim)
pham = cm.PoincareHamiltonian(pvars)
pham.add_MMR_terms(3, 1, max_order = 1, inclinations=False)  # try adding max order up to 3
A = np.eye(pham.N_dof,dtype = int)
A[0,:4] = [-2,3,1,0]
A[1,:4] = [-2,3,0,1]
A[2,:4] = [1,-1,0,0]
A[3,:4] = [-2,3,0,0]
angvars=sp.symbols("theta1,theta2,psi,l,phi1,phi2",real=True)
actions=sp.symbols("p1,p2,Psi,L,Phi1,Phi2",positive=True)
_,_,Psi,L,_,_=actions
ct1 = cm.CanonicalTransformation.from_poincare_angles_matrix(pvars,A,new_qp_pairs=list(zip(angvars,actions)))
ham1=ct1.old_to_new_hamiltonian(pham,do_reduction=True)
ct2 = cm.CanonicalTransformation.polar_to_cartesian(ham1.full_qp_vars,indices=[0,1])
ham2=ct2.old_to_new_hamiltonian(ham1)
# ct_all here
ct_all = cm.CanonicalTransformation.composite([ct1,ct2])
oldvars = ct_all.old_qp_vars
_,eta1,_,_,eta2,_,L1,kappa1,_,L2,kappa2,_= oldvars
newvars = ct_all.new_qp_vars
y1,y2,_,_,_,_,x1,x2,Psi,L,_,_ = newvars

L_Psi_indices = np.array([ct_all.new_qp_vars.index(v) for v in [L, Psi]])
xy_indices = np.array([ct_all.new_qp_vars.index(v) for v in [y1, y2, x1, x2]])

D,L2res,rho = sp.symbols("D,L2,rho")
rho_val = (pham.particles[1].m/pham.particles[2].m) * (2/3)**(1/3)
mtrx = sp.Matrix([[-1,1+rho],[0,3*rho + 2]])
D_exprn,L2res_exprn = mtrx.inv() * sp.Matrix([L,Psi])
Dval,L2val = [float(s.subs(ham2.full_qp).subs({rho:rho_val})) for s in (D_exprn,L2res_exprn)]
newvars = [y1,y2,x1,x2,D_exprn]
subsrule = dict(zip([L,Psi],mtrx * sp.Matrix([D,L2res])))

tau_a1,tau_a2,tau_e1,tau_e2 = sp.symbols("tau_a1,tau_a2,tau_e1,tau_e2")
taus = [tau_a1,tau_a2,tau_e1,tau_e2]
dyvars = [y1, y2, x1, x2, D]

# flow matrices
tau_a1,tau_a2,tau_e1,tau_e2 = sp.symbols("tau_a1,tau_a2,tau_e1,tau_e2")
disflow = sp.Matrix(np.zeros(pham.N_dim))
for hk,tau in zip([(eta1,kappa1),(eta2,kappa2)],(tau_e1,tau_e2)):
    h,k = hk
    disflow[oldvars.index(h)] = -h/tau
    disflow[oldvars.index(k)] = -k/tau
disflow[oldvars.index(L1)] = -L1/tau_a1/sp.S(2)
disflow[oldvars.index(L2)] = -L2/tau_a2/sp.S(2)

from sympy.simplify.fu import TR10,TR10i,TR11
mtrx_func = lambda i,j: sp.simplify(TR10i(TR10(sp.diff(ct_all.new_to_old(newvars[i]),oldvars[j]))))
jac_old = sp.Matrix(len(newvars),len(oldvars),mtrx_func)
newdisflow = sp.simplify(ct_all.old_to_new(jac_old * disflow))

newdisflow = newdisflow.subs(subsrule)

p1 = (x1*x1 + y1*y1)/sp.S(2)
p2 = (x2*x2 + y2*y2)/sp.S(2)
factor = rho * L2res / (3*rho+2)
tau_alpha = 1/(1/tau_a1 - 1/tau_a2)
Ddot_dis = -1*factor *(1/sp.S(2))* 1/tau_alpha - p1/tau_e1 - p2/tau_e2
newdisflow_approx = sp.Matrix([(newdisflow[i] if i<4 else Ddot_dis) for i in range(5)])

newflow = ham2.flow.subs(subsrule).subs(ham2.H_params)

# full flow
tau_alpha_sym, tau_e = sp.symbols('tau_alpha, tau_e')
newpars = {L2res:L2val,rho:rho_val}
newflow_N = newflow.subs(newpars)
newdisflow_approx_N = newdisflow_approx.subs(newpars)


import warnings
#warnings.filterwarnings("ignore")
from scipy.linalg import solve as lin_solve
def newton_solve(fun,Dfun,guess,max_iter=100,rtol=1e-6,atol=1e-12):
    y = guess.copy()
    for itr in range(max_iter):
        f = fun(y)
        Df = Dfun(y)
        dy = -1 * lin_solve(Df,f)
        y+=dy
        if np.alltrue( np.abs(dy) < rtol * np.abs(y) + atol ):
            break
    else:
        warnings.warn("did not converge")
    return y

# full flow
fullflow = sp.Matrix(list(newflow_N) + [0]) + newdisflow_approx_N
taus = [tau_a1,tau_a2,tau_e1,tau_e2]
dyvars = [y1, y2, x1, x2, D]
flow_fn=sp.lambdify(dyvars+taus,fullflow)
jac_fn=sp.lambdify(
    dyvars+taus,
    sp.Matrix(5,5,lambda i,j : sp.diff(fullflow[i],dyvars[j]) )
)
# negative so it works properly
tauvals = list(get_tau_alphas(-50 * 1e3, masses[0], masses[1], period_ratio=3/2)) + [1e3, 1e3]

from scipy.integrate import solve_ivp

times = np.linspace(0,6e3,200)
f=lambda t,x: flow_fn(*x,*tauvals).reshape(-1)
Df = lambda t,x: jac_fn(*x,*tauvals)
soln=solve_ivp(f,(times[0],times[-1]),[0,0,0,0,float(Dval)],method='Radau',t_eval=times,jac=Df)

f=lambda x: flow_fn(*x,*tauvals).reshape(-1)
Df = lambda x: jac_fn(*x,*tauvals)
root=newton_solve(f,Df,soln.y.T[-1])

xydict = dict(zip(dyvars, root))
# sub with the equilibrium values of d, y1, y2, x1, x2, set ddot = 0 and set tau a1 equal to tau a2
fullflow_N = fullflow.subs(xydict).subs([(-tau_e2, tau_e), (-tau_e1, tau_e)])[4]

# substitute 1/tau_a = 1/tau_a1 - 1/tau_a2 usign the coefficient

ddot_eq = fullflow_N.subs(-fullflow_N.coeff(1/tau_a2)/tau_alpha, -fullflow_N.coeff(1/tau_a2)/tau_alpha_sym)
sp.solve(ddot_eq, tau_alpha_sym)

Ddot_dis = -1*factor *(1/sp.S(2))* 1/tau_alpha - p1/tau_e1 - p2/tau_e2
newdisflow_approx = sp.Matrix([(newdisflow[i] if i<4 else Ddot_dis) for i in range(5)])
fullflow = sp.Matrix(list(newflow) + [0]) + newdisflow_approx# .subs(newpars)


# D to K

def D_to_K(flow_mat, root, dyvars=dyvars):
    """
    Computes K = tau_a/tau_e using the flow matrix and its roots

    Flow matrix is in the form zdot = d/dt[y1, y2, x1, x2, D] and settting D_dot to 0
    Root is vector in the from z = [y1, y2, x1, x2, D]

    Flow matrix should have free symbols for tau_a1, tau_a2, tau_e1, tau_e2, x1, x2, y1, y2, D
    (rho and L2 values should be replaced before passing to the function)
    """
    tau_alpha_sym, tau_e, K = sp.symbols('tau_alpha, tau_e, K')
    # sub with equilibrium values of y1, y2, x1, x2, d, and set tau_e1 = tau_e2 = tau_e
    xydict = dict(zip(dyvars, root))
    # tau_e1, tau_e2, tau_a1, tau_a2 need to be defined
    flow_mat_N = flow_mat.subs(xydict).subs([(-tau_e2, tau_e), (-tau_e1, tau_e)])[
        -1]  # D entry is at the end of the matrix
    # substitute tau_a1 = 1/(1/tau_a + 1/tau_a2) and them sub in tau_a = K * tau_e into the sympy expression
    ddot_eq = flow_mat_N.subs(tau_a1, 1 / (1 / tau_alpha_sym + 1 / tau_a2)).subs(tau_alpha_sym, K * tau_e)
    # return solving for tau_a in terms of tau_e (coefficient is equal to K)
    return sp.solve(ddot_eq, K)[0]  # since sp.solve returns a singleton list

# TRY DOWNSAMPLING FIRST
# flat_samples = flat_samples[::1000]

# compute the d/k values using the for loop
# arrays of D and K values
D_values = np.zeros(len(flat_samples))
K_values = np.zeros(len(flat_samples))

from tqdm import tqdm

for i, sample in tqdm(enumerate(flat_samples)):
    # get rebound simulation from sample:
    samp_sim = get_sim_from_params(sample, integrator='ias15', time_base=obs_time_base)
    # masses of inner, outer planets:
    inner_mass = samp_sim.particles[1].m
    outer_mass = samp_sim.particles[2].m
    # convert to pvars
    samp_pvars = cm.Poincare.from_Simulation(samp_sim)
    # compute rho value using the equation above:
    samp_rho_val = (inner_mass / outer_mass) * ((2 / 3) ** (1 / 3))
    # get L and Psi values, as well as x and y values, by transforming from samp_pvars (eta, kappa) -> (y, x)
    samp_L_Psi_vals = ct_all.old_to_new_array(samp_pvars.values)[L_Psi_indices]  # values of L and Psi in that order
    samp_xy_vals = ct_all.old_to_new_array(samp_pvars.values)[xy_indices]  # values y1, y2, x1, x2 in that order
    # get D and L2 values using L and Psi, as well as the rho value computed earlier, from the D_exprn expression
    samp_D_val = D_exprn.subs(zip([L, Psi], samp_L_Psi_vals)).subs(rho,
                                                                   samp_rho_val)  # use L and Psi to compute the D value
    samp_L2_val = L2res_exprn.subs(zip([L, Psi], samp_L_Psi_vals)).subs(rho,
                                                                        samp_rho_val)  # also use L and Psi to compute L2 value
    # parameterize root as [y1, y2, x1, x2, D]
    samp_root = np.append(samp_xy_vals, samp_D_val)
    # compute rho, L2 for this sample for substituting into fullflow
    samp_rho_L2_pars = dict(zip([rho, L2res], [samp_rho_val, samp_L2_val]))
    # finally, convert from D to K using the D_to_K function:
    D_values[i] = samp_D_val  # record sample D value as well as K value from D
    # make sure to substitute in rho and L2 for this specific(!) sample into the D to K values
    K_values[i] = D_to_K(fullflow.subs(samp_rho_L2_pars),
                         samp_root)  # use the fullflow function from earlier since that doesn't change (two planets and only considering 1st order terms)

# save K value array as numpy array so can use for later
np.save('K_value_array', K_values)  # save the K values

# plot K values
plt.figure(figsize = (12, 8))
plt.hist(K_values)
plt.xlabel(r'$K$'), plt.ylabel('count')
plt.savefig('K_value_histogram_unparallelized.png')

# print a sample of K values to see if it's working...
print(K_values[100:150])
