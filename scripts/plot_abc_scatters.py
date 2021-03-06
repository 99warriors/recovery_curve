import recovery_curve.global_stuff
import importlib
import sys
import matplotlib.pyplot as plt
import recovery_curve.prostate_specifics as ps
import itertools
import pdb
from recovery_curve.management_stuff import *
#import traceback, sys

def plot_abc_scatters(the_iterable, scalar_fs):

    for pid_iterator, filter_f, diffcovs_iter, diffcovs_numchains, diffcovs_seed, perf_percentiles, perf_times, get_pops_f, summarize_f, cv_f, ys_f, hypers, x_abc_f, loss_f, actual_ys_f_shift, post_process_f in the_iterable:

        try:
            init_f = ps.set_hard_coded_key_dec(ps.s_f, 'init')(ys_f)
            actual_ys_f = ps.actual_ys_f(ys_f, actual_ys_f_shift)
            pid_filter_f = composed_factory(filter_f, lambda pid: (init_f(pid), actual_ys_f(pid)), expand=True)
            filtered_pid_iterator = ps.filtered_pid_iterator(pid_iterator, pid_filter_f)
            ps.abc_vs_attributes_scatter_f(scalar_fs, init_f, actual_ys_f)(filtered_pid_iterator)
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
    scalar_fs_module_name = sys.argv[2]
    scalar_fs_module = importlib.import_module(scalar_fs_module_name)
    scalar_fs = scalar_fs_module.scalar_fs
    try:
        job_n = int(sys.argv[3])
        log_folder = sys.argv[4]
    except Exception, e:
        plot_abc_scatters(the_iterable, scalar_fs)
    else:
        ps.make_folder(log_folder)
        ps.run_iter_f_parallel_dec(ps.override_sysout_dec(plot_abc_scatters, log_folder), job_n)(the_iterable)
