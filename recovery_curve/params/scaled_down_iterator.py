import recovery_curve.prostate_specifics as ps
from recovery_curve.management_stuff import *
import recovery_curve.hard_coded_objects.hypers as hard_coded_hypers
import recovery_curve.hard_coded_objects.feature_sets as hard_coded_feature_sets
import itertools
import pdb

"""
iterator used for comparing different models with fixed feature set
10-1 - using to search for config where full_model beats logistic regression
"""

class the_iterable_cls(object):

    def __iter__(self):
        pid_iterators = [set_hard_coded_key_dec(ps.filtered_pid_iterator,'surgpids')(ps.all_ucla_pid_iterator(), ps.bin_f(ps.ucla_treatment_f(),ps.equals_bin([ps.ucla_treatment_f.surgery])))]
        filtered_data_fs = [ps.old_filtered_get_data_f(), ps.medium_filtered_get_data_f(), ps.filtered_get_data_f()]
        upscale_vals = [0]
        diffcovs_iters = [5000]
        diffcovs_numchains = [4]
        diffcovs_seeds = [1]
        perf_percentiles = [[0.25, 0.5, 0.75]]
        perf_times = [[1,2,4,8,12,18,24,30,36,42,48]]
        get_pops_fs = [ps.train_better_pops_f()]
        summarize_fs = [ps.get_param_mean_f()]
        cv_fs = [ps.cv_fold_f(4)]
        upscale_vals = [0.0] #
        ys_fs = [ps.modified_ys_f(ps.ys_f(ps.ys_f.sexual_function), ps.score_modifier_f(c)) for c in upscale_vals]
        loss_fs = [ps.scaled_logistic_loss_f(10.0)]


        feature_sets_iterator = ps.get_feature_set_iterator([hard_coded_feature_sets.default_age_categorical_f, hard_coded_feature_sets.medium_age_categorical_f], [hard_coded_feature_sets.default_initial_categorical_f, hard_coded_feature_sets.medium_initial_categorical_f, hard_coded_feature_sets.highminus_initial_categorical_f])
        
        hypers = [hard_coded_hypers.default_hyper, hard_coded_hypers.medium_hyper]
        x_abc_fs = ps.keyed_list([set_hard_coded_key_dec(ps.x_abc_fs, feature_set.get_key())(feature_set, feature_set, feature_set) for feature_set in feature_sets_iterator])

        return itertools.product(pid_iterators, filtered_data_fs, diffcovs_iters, diffcovs_numchains, diffcovs_seeds, perf_percentiles, perf_times, get_pops_fs, summarize_fs, cv_fs, ys_fs, hypers, x_abc_fs, loss_fs)

the_iterable = the_iterable_cls()
