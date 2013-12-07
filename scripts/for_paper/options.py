import recovery_curve.global_stuff as global_stuff
import importlib
from recovery_curve.management_stuff import *
import recovery_curve.prostate_specifics as ps
import pandas
import recovery_curve.getting_data as gd
import recovery_curve.get_posterior_fs as gp
import recovery_curve.predictors_and_trainers as pt

home_folder = '/Users/glareprotector/Documents/Dropbox/prostate_figs'
def get_outfile_path(file):
    return '%s/%s' % (home_folder, file)


"""
real data
"""
filtered_data = importlib.import_module('recovery_curve.hard_coded_objects.real_data_no_flats').data


"""
simulated data 
"""
sim_times = [1,2,4,8,12,18,24,30,36,42,48]

sim_pops = set_hard_coded_key_dec(ps.pops, 'dfltpops')(0.3,0.6,5.0)
sim_pops_f = ps.returns_whats_given_f(sim_pops)


sim_N_1 = 200
sim_N_2 = 500

sim_phi_m = 0.1

sim_param_1 = param = set_hard_coded_key_dec(ps.keyed_dict, 'ABC_n2p1p2_phis_05')({\
            'B_a':pandas.Series([-2.0]),\
                'B_b':pandas.Series([1.0]),\
                'B_c':pandas.Series([2.0]),\
                'phi_a':0.05,\
                'phi_b':0.05,\
                'phi_c':0.05,\
                'phi_m':sim_phi_m,\
                })




sim_param_2 = param = set_hard_coded_key_dec(ps.keyed_dict, 'ABC_n2p1p2_phis_1')({\
            'B_a':pandas.Series([-2.0]),\
                'B_b':pandas.Series([1.0]),\
                'B_c':pandas.Series([2.0]),\
                'phi_a':0.1,\
                'phi_b':0.1,\
                'phi_c':0.1,\
                'phi_m':sim_phi_m,\
                })

sim_noise_f = gd.normal_noise(sim_phi_m)

sim_hypers = ps.hypers(1.0, 1.0, 1.0, 15.0, 15.0, 15.0, 'doesnt matter')

#sim_get_posterior_f_partial = ps.keyed_partial(gp.get_pystan_diffcovs_posterior_f, sim_pops_f, sim_hypers)
sim_get_posterior_f_partial = ps.keyed_partial(gp.get_pystan_diffcovs_posterior_phi_m_fixed_f, 0.1, sim_pops_f, sim_hypers)


"""
stratifying real patients
"""

def get_init_bin(_datum):
    if _datum.xa.iloc[2] > 0:
        return 0
    elif _datum.xa.iloc[1] > 0:
        return 1
    elif _datum.xa.iloc[3] > 0:
        return 2
    else:
        return 3

def get_age_bin(_datum):
    if _datum.xa.iloc[4] > 0:
        return 0
    elif _datum.xa.iloc[5] > 0:
        return 1
    else:
        return 2

num_age_bins = 3
num_init_bins = 4

age_titles = ['age 0 to 55','age 55 to 65','age 65+']
init_titles = ['init 41-', 'init 41 to 60', 'init 60 to 80', 'init 80 to 100']


"""
real_data_inference
"""
real_fixed_phi_m = 0.1
real_times = sim_times
real_get_pops_f = ps.train_shape_pops_f()
real_hypers = sim_hypers
real_get_posterior_f_partial = ps.keyed_partial(gp.get_pystan_diffcovs_posterior_phi_m_fixed_f, real_fixed_phi_m, real_get_pops_f, real_hypers)
real_num_chains = 4
real_iters = 5000
real_get_posterior_f = gp.parallel_merged_get_posterior_f(real_get_posterior_f_partial, real_iters, real_num_chains, global_stuff.num_processes)
our_data_point_trainer = pt.t_point_trainer_from_t_distribution_trainer(pt.t_distribution_trainer_from_abc_distribution_trainer(pt.builtin_auto_abc_distribution_trainer(get_posterior_f)), ps.median_f())
logreg_trainer = pt.logreg_trainer(real_times)
avg_shape_trainer = pt.prior_trainer(real_get_pops_f)
avg_curve_trainer = pt.avg_value_trainer()
loss_fs = [ps.abs_loss_f(), ps.scaled_logistic_loss_f(10)]
loss_f_names = ['abs', 'convex f']
real_performance_cv_f = ps.cv_fold_f(4)
real_percentiles_to_plot = [0.25, 0.50, 0.75]
real_analyze_fold = ps.self_test_cv_f(filtered_data, filtered_data)
patients_to_plot_posterior_curves = [] # this i will hard code in after figuring it out elsewhere
