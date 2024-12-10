================================================================================
Title: Influence of Modeling Assumptions on the Inferred Dynamical State of Resonant Systems: A Case Study of the HD 45364 System
Authors: Chow, I., and Hadden, S. 
================================================================================
Description of Contents: 
This package includes the HDF5 files hd45364_mcmc_uninformative_model_posteriors.h5 
and hd45364_mcmc_resonance_weighted_model_posteriors.h5. The file hd45364_mcmc_uninformative_model_posteriors.h5 
contains the Markov chain Monte Carlo (MCMC) posterior samples computed by N-body simulation using the
uninformative "flat" prior described in Section 2.1 of the associated paper, with log-likelihood 
given by Equation (2). The file hd45364_mcmc_resonance_weighted_model_posteriors.h5 contains the MCMC posterior 
samples computed by N-body simulation using a prior which penalizes dynamical configurations with high libration 
amplitudes described in Section 2.2, with the modified log-likelihood given by Equation (4) for S = 0.1.

System Requirements: 
HDF5 (https://www.hdfgroup.org/HDF5/)

Additional Comments: 
Each file contains data for 50 MCMC chains evolved for 50000 steps, totalling 2.5 million posterior steps each. 
The data is saved in a group named 'mcmc' that contains three datasets: 'accepted', 'samples' and 'log_prob'.

- 'accepted' is a 50 x 1 array containing the number of accepted samples for each chain.
- 'samples' is a 50000 x 50 x 17 array containing all posterior samples.

The parameters of each posterior sample are listed in the order:

        P_b, k_b, T_b, sqrt(e_b) * cos(omega_b), sqrt(e_b) * sin(omega_b), 
        P_c, k_c, T_c, sqrt(e_c) * cos(omega_c), sqrt(e_c) * sin(omega_c), 
        gamma_HARPS1, gamma_HARPS2, gamma_HIRES, 
        sin(I), 
        jit_HARPS1, jit_HARPS2, jit_HIRES

where P_i is the orbital period in days, k_i is the RV semi-amplitude in m s^-1, T_i is the time of conjunction 
in BJD - 2.4 * 10^6 days, e_i is the eccentricity, and omega_i is the argument of pericenter for the ith planet.
gamma_j is the RV offset in m s^-1 and jit_j is the instrumental jitter in m s^-1 for the jth instrument. 
I is the inclination of the planets' orbital planes with respect to the sky plane, assuming they are coplanar.
The planets' orbits are specified at a reference epoch of t_0 = 54422.79 + 2.4 * 10^6 BJD. The host star's mass
is fixed at 0.82 solar masses.

- 'log_prob' is a 50000 x 50 array containing the corresponding log-probability of each posterior sample in 'samples'. 
For the file 'hd45364_mcmc_uninformative_model_posteriors.h5', this is equal to the log-likelihood 
given by Equation (2) of the associated paper. For the file 'hd45364_mcmc_resonance_weighted_model_posteriors.h5', 
this is equal to the modified log-likelihood given by Equation (4) for S = 0.1.
================================================================================
