import recovery_curve.hard_coded_objects.hypers as hard_coded_hypers
import recovery_curve.prostate_specifics as ps
import recovery_curve.get_posterior_fs as gp

import recovery_curve.hard_coded_objects.go_two_sim_data as sim_data

num_iter = 5000
num_chains = 5
get_posterior_f_cons = gp.get_pystan_diffcovs_posterior_f
get_pops_f_to_use = ps.returns_whats_given_f(sim_data.pops)
hypers = hard_coded_hypers.default_hyper
get_posterior_f_cons_partial = ps.keyed_partial(get_posterior_f_cons, get_pops_f_to_use, hypers)
get_posterior_f = gp.merged_get_posterior_f(get_posterior_f_cons_partial, num_iter, num_chains)