import prostate_specifics as ps

from management_stuff import *
import recovery_curve.hard_coded_objects.hypers as hard_coded_hypers
import recovery_curve.hard_coded_objects.feature_sets as hard_coded_feature_sets

"""
for use with iterator used for comparing different models with fixed feature set
"""

pid_iterators = [set_hard_coded_key_dec(ps.filtered_pid_iterator,'surgpids')(ps.all_ucla_pid_iterator, ps.bin_f(ps.ucla_treatment_f(),ps.equals_bin([ps.ucla_treatment_f.surgery])))]
filtered_data_fs = [ps.filtered_get_data_f]
upscale_vals = [0, 0.2, 0.4]
diffcovs_iters = [10000]
diffcovs_numchains = [1]
diffcovs_seeds = [1]
perf_percentiles = [[0.25, 0.5, 0.75]]
perf_times = [[1,2,4,8,12,18,24,30,36,42,48]]
get_pops_fs = [ps.train_better_pops_f()]
summarize_fs = [ps.get_param_mean_f()]
loss_fs = [ps.scaled_logistic_loss_f(10)]
cv_fs = [ps.cv_fold_f(3)]
upscale_vals = [0.0,0.2,0.4] #
ys_fs = [ps.modified_ys_f(ps.ys_f(ps.ys_f.sexual_function), ps.score_modifier_f(c)) for c in upscale_vals]

prior_trainer = ps.get_prior_predictor_f(get_pops_f) #
logreg_trainer = ps.get_logreg_predictor_f(perf_times) #

feature_sets = [hard_coded_feature_sets.a_feature_set] #
hypers = [hard_coded_hypers.a_hyper]
x_abc_fs = ps.keyed_list([set_hard_coded_key_dec(ps.x_abc_fs, feature_set.get_key())(feature_set, feature_set, feature_set) for feature_set in feature_sets])

