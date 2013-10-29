import sys
import pdb
import recovery_curve.global_stuff
import importlib
import sys
import matplotlib.pyplot as plt
import recovery_curve.prostate_specifics as ps
import itertools
import pdb
from recovery_curve.management_stuff import *
import traceback

def plot_logreg_coeffs(the_iterable):

    for pid_iterator, filter_f, diffcovs_iter, diffcovs_numchains, diffcovs_seed, perf_percentiles, perf_times, get_pops_f, summarize_f, cv_f, ys_f, hypers, x_abc_f, loss_f, actual_ys_f_shift, post_process_f in the_iterable:

        try:
            shifted_perf_times = [t - actual_ys_f_shift for t in perf_times]
            logreg_trainer = ps.get_logreg_predictor_f(shifted_perf_times)

            init_f = ps.set_hard_coded_key_dec(ps.s_f, 'init')(ys_f)
            actual_ys_f = ps.actual_ys_f(ys_f, actual_ys_f_shift)
            data = ps.get_data_f(x_abc_f, init_f, actual_ys_f)(pid_iterator)
            filtered_data = ps.generic_filtered_get_data_f(filter_f)(data)
            filtered_data = post_process_f(filtered_data)

            logreg_predictor = ps.get_logreg_predictor_f(shifted_perf_times)(filtered_data)
            logregshape_predictor = ps.get_logregshape_predictor_f(shifted_perf_times)(filtered_data)
            logregshapeunif_predictor = ps.get_logregshapeunif_predictor_f(shifted_perf_times)(filtered_data)
            ps.plot_logreg_coefficients_fig_f()(logreg_predictor)
            ps.plot_logreg_coefficients_fig_f()(logregshape_predictor)
            ps.plot_logreg_coefficients_fig_f()(logregshapeunif_predictor)


        except Exception, e:
            for frame in traceback.extract_tb(sys.exc_info()[2]):
                fname,lineno,fn,text = frame
                print "Error in %s on line %d" % (fname, lineno)
            print e
            pdb.set_trace()
            pass

if __name__ == '__main__':
    iter_module_name = sys.argv[1]
    iter_module = importlib.import_module(iter_module_name)
    the_iterable = iter_module.the_iterable
    try:
        job_n = int(sys.argv[2])
        log_folder = sys.argv[3]
    except Exception, e:
        plot_logreg_coeffs(the_iterable)
    else:
        ps.make_folder(log_folder)
        ps.randomize_iterable_dec(ps.run_iter_f_parallel_dec(ps.override_sysout_dec(plot_logreg_coeffs, log_folder), job_n))(the_iterable)
