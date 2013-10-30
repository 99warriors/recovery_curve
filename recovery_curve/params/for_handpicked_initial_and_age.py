import recovery_curve.prostate_specifics as ps
from recovery_curve.management_stuff import *
import recovery_curve.hard_coded_objects.hypers as hard_coded_hypers
import recovery_curve.hard_coded_objects.feature_sets as hard_coded_feature_sets
import recovery_curve.hard_coded_objects.filter_fs as hard_coded_filter_fs
import itertools
import pdb

"""
new version of iterator - returns ys_filter_f(s,ys) which can be used by generic_get_data
returns post_process_f(data)

right now, use 4fold cv when working with hand picked , 3 fold with old filtering

"""

class the_iterable_cls(object):

    def __iter__(self):
        
        pid_iterators1 = [ps.filtered_pid_iterator(set_hard_coded_key_dec(ps.filtered_pid_iterator,'surgpids')(ps.all_ucla_pid_iterator(), ps.bin_f(ps.ucla_treatment_f(),ps.equals_bin([ps.ucla_treatment_f.surgery]))), ps.is_good_pid())]
        pid_iterators2 = [set_hard_coded_key_dec(ps.filtered_pid_iterator,'surgpids')(ps.all_ucla_pid_iterator(), ps.bin_f(ps.ucla_treatment_f(),ps.equals_bin([ps.ucla_treatment_f.surgery])))]
        filter_fs1 = [ps.always_true_f()]
        filter_fs2 = [hard_coded_filter_fs.old_filter_f]
        upscale_vals = [0]
        diffcovs_iters = [5000]
        diffcovs_numchains = [4]
        diffcovs_seeds = [1]
        perf_percentiles = [[0.25, 0.5, 0.75]]
        perf_times = [[1,2,4,8,12,18,24,30,36,42,48]]
        get_pops_fs = [ps.train_better_pops_f()]
        summarize_fs = [ps.get_param_mean_f()]
        cv_fs1 = [ps.cv_fold_f(4)]
        cv_fs2 = [ps.cv_fold_f(3)]
        upscale_vals = [0.0] #
        ys_fs = [ps.modified_ys_f(ps.ys_f(ps.ys_f.sexual_function), ps.score_modifier_f(c)) for c in upscale_vals]

        post_process_fs = [ps.normalized_data_f()]

        actual_ys_f_shifts = [0,1]

        loss_fs = [ps.scaled_logistic_loss_f(10.0)]

        ones_f_list = set_hard_coded_key_dec(ps.keyed_list, 'ones')([ps.ones_f()])

        #feature_sets_iterator = ps.get_feature_set_iterator([ones_f_list], hard_coded_feature_sets.default_age_categorical_f, hard_coded_feature_sets.medium_age_categorical_f], [hard_coded_feature_sets.default_initial_categorical_f, hard_coded_feature_sets.medium_initial_categorical_f, hard_coded_feature_sets.highminus_initial_categorical_f])
        feature_sets_iterator = ps.get_feature_set_iterator([ones_f_list], [hard_coded_feature_sets.default_age_categorical_f], [hard_coded_feature_sets.default_initial_categorical_f, hard_coded_feature_sets.highminus_initial_categorical_f])
        
        hypers = [hard_coded_hypers.default_hyper]

        x_abc_fs = ps.keyed_list([set_hard_coded_key_dec(ps.x_abc_fs, feature_set.get_key())(feature_set, feature_set, feature_set) for feature_set in feature_sets_iterator])

        return itertools.chain(\
            itertools.product(pid_iterators1, filter_fs1, diffcovs_iters, diffcovs_numchains, diffcovs_seeds, perf_percentiles, perf_times, get_pops_fs, summarize_fs, cv_fs1, ys_fs, hypers, x_abc_fs, loss_fs, actual_ys_f_shifts, post_process_fs),\
            itertools.product(pid_iterators2, filter_fs2, diffcovs_iters, diffcovs_numchains, diffcovs_seeds, perf_percentiles, perf_times, get_pops_fs, summarize_fs, cv_fs2, ys_fs, hypers, x_abc_fs, loss_fs, actual_ys_f_shifts, post_process_fs)\
                )

        return itertools.chain(\
            itertools.product(pid_iterators1, filter_fs1, diffcovs_iters, diffcovs_numchains, diffcovs_seeds, perf_percentiles, perf_times, get_pops_fs, summarize_fs, cv_fs, ys_fs, hypers, x_abc_fs, loss_fs, actual_ys_f_shifts, post_process_fs), \
            itertools.product(pid_iterators2, filter_fs2, diffcovs_iters, diffcovs_numchains, diffcovs_seeds, perf_percentiles, perf_times, get_pops_fs, summarize_fs, cv_fs, ys_fs, hypers, x_abc_fs, loss_fs, actual_ys_f_shifts, post_process_fs)\
                )

the_iterable = the_iterable_cls()
