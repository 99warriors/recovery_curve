import recovery_curve.prostate_specifics as ps
from recovery_curve.management_stuff import *
import recovery_curve.hard_coded_objects.hypers as hard_coded_hypers
import recovery_curve.hard_coded_objects.feature_sets as hard_coded_feature_sets
import recovery_curve.hard_coded_objects.filter_fs as hard_coded_filter_fs
import itertools
import pdb
import functools


"""
difference between this and mini_iterator_newfiltereddata:
1. post processing includes smoothing the data
2. get_posterior_f_constructor will be model where phi_m is fixed, and set to be really small
"""

class the_iterable_cls(object):

    def __iter__(self):
        phi_m = 0.001
        #get_posterior_f_constructors = [functools.partial(ps.get_pystan_diffcovs_truncated_posterior_f, 0.01)]
        get_posterior_f_constructors = [functools.partial(ps.get_pystan_diffcovs_posterior_phi_m_fixed_f, 0.01)]
        pid_iterators = [ps.filtered_pid_iterator(set_hard_coded_key_dec(ps.filtered_pid_iterator,'surgpids')(ps.all_ucla_pid_iterator(), ps.bin_f(ps.ucla_treatment_f(),ps.equals_bin([ps.ucla_treatment_f.surgery]))), ps.is_good_pid())]
        #pid_iterators = [set_hard_coded_key_dec(ps.filtered_pid_iterator,'surgpids')(ps.all_ucla_pid_iterator(), ps.bin_f(ps.ucla_treatment_f(),ps.equals_bin([ps.ucla_treatment_f.surgery])))]
        #pid_iterators = [ps.all_ucla_pid_iterator()]
        #filter_fs = [hard_coded_filter_fs.old_filter_f]
        filter_fs = [ps.always_true_f()]
        #filtered_data_fs = [ps.generic_filtered_get_data_f(filter_f) for filter_f in filter_fs]
        upscale_vals = [0]
        diffcovs_iters = [5000]
        diffcovs_numchains = [4]
        diffcovs_seeds = [1]
        perf_percentiles = [[0.25, 0.5, 0.75]]
        perf_times = [[1,2,4,8,12,18,24,30,36,42,48]]
        get_pops_fs = [ps.train_better_pops_f()]
        summarize_fs = [ps.get_param_mean_f()]
        #cv_fs = [ps.cv_fold_f(3)]
        cv_fs = [ps.self_test_cv_f()]
        upscale_vals = [0.0] #

        ys_fs = [ps.modified_ys_f(ps.ys_f(ps.ys_f.sexual_function), ps.score_modifier_f(c)) for c in upscale_vals]

        post_process_fs = [composed_factory(ps.normalized_data_f(), ps.smooth_data_f())]

        actual_ys_f_shifts = [0]

        loss_fs = [ps.scaled_logistic_loss_f(10.0)]

        ones_f_list = set_hard_coded_key_dec(ps.keyed_list, 'ones')([ps.ones_f()])

        #feature_sets_iterator = [hard_coded_feature_sets.default_simple_indicators]
        
        feature_sets_iterator = ps.get_feature_set_iterator([ones_f_list], [hard_coded_feature_sets.default_age_categorical_f], [hard_coded_feature_sets.default_initial_categorical_f])

        hypers = [hard_coded_hypers.default_hyper]

        x_abc_fs = ps.keyed_list([set_hard_coded_key_dec(ps.x_abc_fs, feature_set.get_key())(feature_set, feature_set, feature_set) for feature_set in feature_sets_iterator])

        return itertools.product(get_posterior_f_constructors,pid_iterators, filter_fs, diffcovs_iters, diffcovs_numchains, diffcovs_seeds, perf_percentiles, perf_times, get_pops_fs, summarize_fs, cv_fs, ys_fs, hypers, x_abc_fs, loss_fs, actual_ys_f_shifts, post_process_fs)

the_iterable = the_iterable_cls()
