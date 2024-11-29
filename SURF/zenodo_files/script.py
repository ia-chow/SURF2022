import h5py
import rebound as rb
import numpy as np

posteriors = h5py.File('./hd45364_mcmc_posteriors_resonance_weighted_model_posteriors.h5', 'r')
samples = np.array(posteriors['mcmc']['chain'])

# WIP