"""
iterator that iterates over sets of models as to compare their performance
"""
import itertools
import prostate_specifics as ps

# module containing params will be imported into global
p = global_stuff.params

base_iter = itertools.product(p.pid_iterators,\
                             p.filtered_data_fs,\
                             p.diffcovs_iters,\
                             p.diffcovs_numchains,\
                             p.diffcovs_seeds,\
                             p.perf_percentiles,\
                             p.perf_times,\
                             p.get_pops_fs,\
                             p.summarize_fs,\
                             p.loss_fs,\
                             p.cv_fs,\
                             p.feature_sets,\
                             p.hypers,\
                             p.x_abc_fs,\
                             p.ys_fs)

def param_to_datasets_and_trainers(pid_iterator, filtered_data_f, diffcovs_iter, diffcovs_numchains, diffcovs_seed, perf_percentiles, perf_times, get_pops_f, summarize_f, loss_f, cv_f, feature_set, hypers, x_abc_f, ys_f):
    get_posterior_f = ps.get_diffcovs_posterior_f(get_pops_f, hypers, diffcovs_iter, diffcovs_numchains, diffcovs_seed)
    diffcovs_trainer = ps.get_diffcovs_point_predictor_f(get_posterior_f, summarize_f)
    prior_trainer = ps.get_prior_predictor_f(get_pops_f)
    logreg_trainer = ps.get_logreg_predictor_f(perf_times)
    init_f = init = set_hard_coded_key_dec(ps.s_f, 'init')(ys_f)
    data = ps.get_data_f(x_abc_f, init_f, ys_f)(pid_iterator)
    filtered_data = filtered_data_f(data)
    return [prior_trainer, logreg_trainer, diffcovs_trainer], filtered_data

the_iter = itertools.imap(base_iter, param_to_datasets_and_trainers)
