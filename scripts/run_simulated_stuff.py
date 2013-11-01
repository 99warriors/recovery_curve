import recovery_curve.global_stuff
import importlib
import sys
import matplotlib.pyplot as plt
import recovery_curve.prostate_specifics as ps
import itertools
import pdb
from recovery_curve.management_stuff import *

def plot_full_model_posterior_parameters(the_iterable):
    
    for sim_get_data_f_constructor, get_posterior_f_constructor, num_pid, diffcovs_iter, diffcovs_numchains, seed, sim_times, get_pops_f, hypers, params, pops, id_to_x_s_f in the_iterable:

        try:

            get_pops_f_to_use = ps.returns_whats_given_f(pops)

            #get_posterior_f = ps.get_pystan_diffcovs_posterior_f(get_pops_f, hypers, diffcovs_iter, diffcovs_numchains, seed)
            #get_posterior_f = ps.get_pystan_diffcovs_posterior_f(ps.returns_whats_given_f(pops), hypers, diffcovs_iter, diffcovs_numchains, seed)

            get_posterior_f_cons_partial = ps.keyed_partial(get_posterior_f_constructor, get_pops_f_to_use, hypers)
            get_posterior_f = ps.merged_get_posterior_f(get_posterior_f_cons_partial, diffcovs_iter, diffcovs_numchains)




            #get_posterior_f = get_posterior_f_constructor(ps.returns_whats_given_f(pops), hypers, diffcovs_iter, diffcovs_numchains, seed)
            pid_iterator = ps.set_hard_coded_key_dec(ps.keyed_list, 'dum')(range(num_pid))
            data = sim_get_data_f_constructor(params, pops,id_to_x_s_f, sim_times, seed)(pid_iterator)
            #shape = ps.aggregate_shape_f()(data)
            #fit_pops = get_pops_f(data)
            #pdb.set_trace()
            posteriors = get_posterior_f(data)


            #d=ps.gelman_statistic_f()(posteriors)

            ps.plot_single_posterior_f(2,1)(posteriors)

            diffcovs_trainer = ps.get_diffcovs_point_predictor_f(get_posterior_f, ps.get_param_mean_f())
            nonpoint_trainer = ps.get_diffcovs_nonpoints_predictor_f(get_posterior_f)
            trainers = ps.keyed_list([nonpoint_trainer, diffcovs_trainer])
            perf_times = [1,2,4,8,12,18,24,30,36,42,48]
            percentiles = [0.25, 0.5, 0.75]
            cv_f = ps.self_test_cv_f()

            #ps.model_comparer_f(trainers, cv_f, ps.scaled_logistic_loss_f(10.0), percentiles, perf_times)(data)
            ps.plot_all_predictions_fig_f(trainers, cv_f, perf_times)(data)
            """
            scores_getter = ps.cross_validated_scores_f(diffcovs_trainer, cv_f, perf_times)
            distance_fs = ps.set_hard_coded_key_dec(ps.keyed_list, 'somedists')([ps.signed_loss_f(), ps.scaled_logistic_loss_f(10), ps.abs_loss_f()])
            get_true_val_fs = ps.set_hard_coded_key_dec(ps.keyed_list, 'sometrues')([ps.get_true_val(), ps.get_true_val_abc_sim(), ps.get_true_val_abc_fit()])
            ps.loss_comparer_f(scores_getter, distance_fs, get_true_val_fs, percentiles, perf_times)(data)
            """


        except Exception, e:
            import traceback
            for frame in traceback.extract_tb(sys.exc_info()[2]):
                fname,lineno,fn,text = frame
                print "Error in %s on line %d" % (fname, lineno)
            print e
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
        ps.make_folder(log_folder)
        ps.run_iter_f_parallel_dec(ps.override_sysout_dec(plot_full_model_posterior_parameters, log_folder), job_n)(the_iterable)
