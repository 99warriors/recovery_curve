import recovery_curve.prostate_specifics as ps
import matplotlib.pyplot as plt
import recovery_curve.global_stuff as global_stuff
import random
import recovery_curve.get_posterior_fs as gp
import recovery_curve.getting_data as gd
import numpy as np
import importlib
import pdb

data = importlib.import_module('recovery_curve.hard_coded_objects.real_data_no_flats').data
get_pops_f = ps.train_shape_pops_f()
hypers = ps.hypers(1.0, 1.0, 1.0, 15.0, 15.0, 15.0, 'doesnt matter')
num_steps = 1000
num_chains = 1
seed = 1
phi_m = 0.1


posteriors = gp.get_pystan_diffcovs_posterior_truncated_phi_m_fixed_has_test_f(phi_m, get_pops_f, hypers, num_steps, num_chains, seed)(data, data)
#posteriors = gp.get_pystan_diffcovs_posterior_phi_m_fixed_has_test_f(phi_m, get_pops_f, hypers, num_steps, num_chains, seed)(data, data)
