import recovery_curve.global_stuff
import plot_model_performances
import plot_full_model_posterior_parameters
import plot_predicted_patient_curves
import sys
import importlib
import recovery_curve.prostate_specifics as ps

if __name__ == '__main__':
    iter_module_name = sys.argv[1]
    iter_module = importlib.import_module(iter_module_name)
    the_iterable = iter_module.the_iterable
    fs = [plot_model_performances.plot_model_performances, \
          plot_full_model_posterior_parameters.plot_full_model_posterior_parameters, \
          plot_predicted_patient_curves.plot_predicted_patient_curves]
    try:
        job_n = int(sys.argv[2])
        log_folder = sys.argv[3]
    except Exception, e:
        print e
        for f in fs:
            f(the_iterable)
    else:
        ps.make_folder(log_folder)
        for f in fs:
            ps.run_iter_f_parallel_dec(ps.override_sysout_dec(f, log_folder), job_n)(the_iterable)
