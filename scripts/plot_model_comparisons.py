"""
on same plot, for a fixed feature set, plot performance of several models
"""

import importlib
import sys
import matplotlib.pyplot as plt
import recovery_curve.prostate_specifics as ps
import itertools
import pdb
from recovery_curve.management_stuff import *

iter_module = sys.argv[1]

the_iter = importlib.import_module(iter_module).the_iter

def param_to_datasets_and_trainers(pid_iterator, filtered_data_f, diffcovs_iter, diffcovs_numchains, diffcovs_seed, perf_percentiles, perf_times, get_pops_f, summarize_f, cv_f, ys_f, hypers, x_abc_f, loss_f):
    get_posterior_f = ps.get_diffcovs_posterior_f(get_pops_f, hypers, diffcovs_iter, diffcovs_numchains, diffcovs_seed)
    diffcovs_trainer = ps.get_diffcovs_point_predictor_f(get_posterior_f, summarize_f)
    prior_trainer = ps.get_prior_predictor_f(get_pops_f)
    logreg_trainer = ps.get_logreg_predictor_f(perf_times)
    init_f = ps.set_hard_coded_key_dec(ps.s_f, 'init')(ys_f)
    data = ps.get_data_f(x_abc_f, init_f, ps.after_0_ys_f(ys_f))(pid_iterator)
    filtered_data = filtered_data_f(data)
    return ps.keyed_list([prior_trainer, logreg_trainer, diffcovs_trainer]), cv_f, loss_f, perf_percentiles, perf_times, filtered_data

model_data_iter = ps.my_iter_apply(param_to_datasets_and_trainers, the_iter)

"""
for trainers, cv_f, loss_f, perf_percentiles, perf_times, filtered_data in model_data_iter:

    prior_trainer = trainers[0]
    logreg_trainer = trainers[1]
    pdb.set_trace()
    ps.plot_all_predictions_fig_f(ps.keyed_list([prior_trainer, logreg_trainer]), cv_f, perf_times)(filtered_data)

"""
"""
for pid_iterator, filtered_data_f, diffcovs_iter, diffcovs_numchains, diffcovs_seed, perf_percentiles, perf_times, get_pops_f, summarize_f, cv_f, ys_f, hypers, x_abc_f, loss_f in the_iter:
    init_f = ps.set_hard_coded_key_dec(ps.s_f, 'init')(ys_f)
    data = ps.get_data_f(x_abc_f, init_f, ys_f)(pid_iterator)
    filtered_data = filtered_data_f(data)
    for training_data, testing_data in cv_f(filtered_data):
        #print ps.aggregate_shape_f()(training_data)
        print ps.train_shape_pops_f()(training_data)
        print ps.train_better_pops_f()(training_data)
        print ps.train_better_pops_f()(training_data).get_location()
        pdb.set_trace()
"""

for pid_iterator, filtered_data_f, diffcovs_iter, diffcovs_numchains, diffcovs_seed, perf_percentiles, perf_times, get_pops_f, summarize_f, cv_f, ys_f, hypers, x_abc_f, loss_f in the_iter:
    get_posterior_f = ps.get_diffcovs_posterior_f(get_pops_f, hypers, diffcovs_iter, diffcovs_numchains, diffcovs_seed)
    init_f = ps.set_hard_coded_key_dec(ps.s_f, 'init')(ys_f)
    data = ps.get_data_f(x_abc_f, init_f, ps.after_0_ys_f(ys_f))(pid_iterator)
    filtered_data = filtered_data_f(data)
    ps.plot_diffcovs_posterior_f(3, 2, cv_f, get_posterior_f)(filtered_data)


#save_at_specified_path_dec(ps.figure_combiner_f(ps.model_comparer_f, lambda x: (x[0:5], [x[5]])),'some_figs_old.pdf')(model_data_iter)

#for _models, _cv_f, _loss_f, _perf_percentiles, _perf_times, _filtered_data in model_data_iter:
#    diffcovs_trainer = _models[2]
    
#    ps.plot_diffcovs_posterior_f(3,2)()
