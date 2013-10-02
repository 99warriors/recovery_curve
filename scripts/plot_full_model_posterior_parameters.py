import importlib
import sys
import matplotlib.pyplot as plt
import recovery_curve.prostate_specifics as ps
import itertools
import pdb
from recovery_curve.management_stuff import *

def plot_full_model_posterior_parameters(the_iter):

    for pid_iterator, filtered_data_f, diffcovs_iter, diffcovs_numchains, diffcovs_seed, perf_percentiles, perf_times, get_pops_f, summarize_f, cv_f, ys_f, hypers, x_abc_f, loss_f in the_iter:

        get_posterior_f = ps.get_diffcovs_posterior_f(get_pops_f, hypers, diffcovs_iter, diffcovs_numchains, diffcovs_seed)
        init_f = ps.set_hard_coded_key_dec(ps.s_f, 'init')(ys_f)
        data = ps.get_data_f(x_abc_f, init_f, ys_f)(pid_iterator)
        filtered_data = filtered_data_f(data)

        ps.plot_diffcovs_posterior_f(3, 2, cv_f, get_posterior_f)(filtered_data)

if __name__ == '__main__':
    iter_module = sys.argv[1]
    the_iter = importlib.import_module(iter_module).the_iter
    plot_full_model_posterior_parameters(the_iter)
