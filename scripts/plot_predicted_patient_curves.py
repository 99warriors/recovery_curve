import importlib
import sys
import matplotlib.pyplot as plt
import recovery_curve.prostate_specifics as ps
import itertools
import pdb
from recovery_curve.management_stuff import *

iter_module = sys.argv[1]
the_iter = importlib.import_module(iter_module).the_iter

def plot_predicted_patient_curves(the_iter):

    for pid_iterator, filtered_data_f, diffcovs_iter, diffcovs_numchains, diffcovs_seed, perf_percentiles, perf_times, get_pops_f, summarize_f, cv_f, ys_f, hypers, x_abc_f, loss_f in the_iter:

        get_posterior_f = ps.get_diffcovs_posterior_f(get_pops_f, hypers, diffcovs_iter, diffcovs_numchains, diffcovs_seed)
        diffcovs_trainer = ps.get_diffcovs_point_predictor_f(get_posterior_f, summarize_f)
        prior_trainer = ps.get_prior_predictor_f(get_pops_f)
        logreg_trainer = ps.get_logreg_predictor_f(perf_times)
        init_f = ps.set_hard_coded_key_dec(ps.s_f, 'init')(ys_f)
        data = ps.get_data_f(x_abc_f, init_f, ys_f)(pid_iterator)
        filtered_data = filtered_data_f(data)

        ps.plot_all_predictions_fig_f(ps.keyed_list([prior_trainer, logreg_trainer, diffcovs_trainer]), cv_f, perf_times)(filtered_data)

if __name__ == '__main__':
    iter_module = sys.argv[1]
    the_iter = importlib.import_module(iter_module).the_iter
    plot_predicted_patient_curves(the_iter)
