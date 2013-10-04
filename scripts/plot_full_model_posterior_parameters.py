import recovery_curve.global_stuff
import importlib
import sys
import matplotlib.pyplot as plt
import recovery_curve.prostate_specifics as ps
import itertools
import pdb
from recovery_curve.management_stuff import *

def plot_full_model_posterior_parameters(the_iterable):
    
    for pid_iterator, filtered_data_f, diffcovs_iter, diffcovs_numchains, diffcovs_seed, perf_percentiles, perf_times, get_pops_f, summarize_f, cv_f, ys_f, hypers, x_abc_f, loss_f in the_iterable:

        try:
            get_posterior_f = ps.get_pystan_diffcovs_posterior_f(get_pops_f, hypers, diffcovs_iter, diffcovs_numchains, diffcovs_seed)
            init_f = ps.set_hard_coded_key_dec(ps.s_f, 'init')(ys_f)
            data = ps.get_data_f(x_abc_f, init_f, ys_f)(pid_iterator)
            filtered_data = filtered_data_f(data)

            ps.plot_diffcovs_posterior_f(3, 2, cv_f, get_posterior_f)(filtered_data)
        except:
            pass

if __name__ == '__main__':
    iter_module_name = sys.argv[1]
    iter_module = importlib.import_module(iter_module_name)
    the_iterable = iter_module.the_iterable
    try:
        job_n = int(sys.argv[2])
        log_folder = sys.argv[3]
    except Exception, e:
        plot_full_model_posterior_parameters(the_iterable)
    else:
        ps.run_iter_f_parallel_dec(ps.override_sysout_dec(plot_full_model_posterior_parameters, log_folder), job_n)(the_iterable)
