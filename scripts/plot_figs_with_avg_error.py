import recovery_curve.global_stuff
import importlib
import sys
import matplotlib.pyplot as plt
import recovery_curve.prostate_specifics as ps
import itertools
import pdb
from recovery_curve.management_stuff import *
#import traceback, sys

def plot_figs_with_avg_error(the_iterable, scalar_fs):

    for pid_iterator, filter_f, diffcovs_iter, diffcovs_numchains, diffcovs_seed, perf_percentiles, perf_times, get_pops_f, summarize_f, cv_f, ys_f, hypers, x_abc_f, loss_f, actual_ys_f_shift, post_process_f in the_iterable:

        try:
            init_f = ps.set_hard_coded_key_dec(ps.s_f, 'init')(ys_f)
            actual_ys_f = ps.actual_ys_f(ys_f, actual_ys_f_shift)
            ps.abc_attributes_scatter_f(scalar_fs, init_f, actual_ys_f)(pid_iterator)
            ps.figs_with_avg_error_f(init_f, actual_ys_f)(pid_iterator)
        except Exception, e:
            print e
            ps.print_traceback()
            """
            for frame in traceback.extract_tb(sys.exc_info()[2]):
                fname,lineno,fn,text = frame
                print "Error in %s on line %d" % (fname, lineno)            
            """
            pass

if __name__ == '__main__':
    iter_module_name = sys.argv[1]
    iter_module = importlib.import_module(iter_module_name)
    the_iterable = iter_module.the_iterable
    try:
        job_n = int(sys.argv[2])
        log_folder = sys.argv[3]
    except Exception, e:
        plot_figs_with_avg_error(the_iterable)
    else:
        ps.make_folder(log_folder)
        ps.run_iter_f_parallel_dec(ps.override_sysout_dec(plot_figs_with_avg_error, log_folder), job_n)(the_iterable)
