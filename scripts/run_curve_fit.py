import pdb
import recovery_curve.global_stuff
import recovery_curve.global_stuff as global_stuff
import recovery_curve.prostate_specifics as ps
from recovery_curve.prostate_specifics import *
import recovery_curve.get_posterior_fs as gp
import recovery_curve.predictors_and_trainers as pt
import recovery_curve.plotters as plotters
import importlib
import sys
import recovery_curve.getting_data as gd
import matplotlib.pyplot as plt

"""                                                                                                                                                                                                        
goal of this script is to get posteriors, not even to see results                                                                                                                                          
"""


data = importlib.import_module(sys.argv[1]).data
data_module = importlib.import_module(sys.argv[1])



"""
plot aggregate curve and shape
"""
fig = plt.figure()
ax = fig.add_subplot(1,1,1)
agg_shape = ps.aggregate_shape_f()(data)
agg_curve = ps.aggregate_curve_f()(data)
ax.plot(agg_shape.index, agg_shape, label = 'aggregate shape')
ax.plot(agg_curve.index, agg_curve, label = 'aggregate curve')
ax.legend(loc=4)
ax.set_xlabel('time')
fig.suptitle('aggregates of filtered data')
ps.figure_to_pdf(fig,'%s/%s' % (global_stuff.for_dropbox, 'aggregates_filtered'))





#data = ps.data([data.d['30011']])

l_m = .1

phi_m = .1

cs_l = 1.0

_hypers = hypers(1.0, 1.0, 1.0, 15.0, 15.0, 15.0, 'doesnt matter')

_pops_f = returns_whats_given_f(pops(0.5, 0.5, 5.0))


def truncated_normal(true, obs):
    return truncated_normal_pdf(obs, true, phi_m, 0.0, 1.0)

def normal(true, obs):
    return normal_pdf(obs, true, phi_m)

def dist(true, obs):
    return abs(true - obs)

#get_posterior_f_cons_partial = ps.keyed_partial(gp.get_pystan_curve_fit_beta_noise_posterior_f, l_m, cs_l)


#get_posterior_f_cons_partial = ps.keyed_partial(gp.get_pystan_curve_fit_normal_noise_posterior_f, l_m, cs_l)

#get_posterior_f_cons_partial = ps.keyed_partial(gp.get_pystan_curve_fit_normal_noise_fixed_posterior_f, phi_m, cs_l)



#get_posterior_f_cons_partial = ps.keyed_partial(gp.get_pystan_curve_fit_normal_noise_fixed_as_fixed_posterior_f, phi_m, cs_l)

#get_posterior_f_cons_partial = ps.keyed_partial(gp.get_pystan_curve_fit_normal_nonshared_noise_posterior_f, l_m, cs_l)

#get_posterior_f_cons_partial = ps.keyed_partial(gp.get_pystan_curve_fit_beta_noise_nonshared_posterior_f, l_m, cs_l)

#get_posterior_f_cons_partial = ps.keyed_partial(gp.get_pystan_diffcovs_posterior_phi_m_fixed_f, phi_m, _pops_f, _hypers)

#get_posterior_f_cons_partial = ps.keyed_partial(gp.get_pystan_diffcovs_posterior_phi_m_fixed_has_test_f, phi_m, _pops_f, _hypers)
get_posterior_f_cons_partial = ps.keyed_partial(gp.get_pystan_diffcovs_posterior_truncated_phi_m_fixed_has_test_f, phi_m, _pops_f, _hypers)
#get_posterior_f_cons_partial = ps.keyed_partial(gp.get_pystan_curve_fit_truncated_normal_noise_fixed_posterior_f, phi_m, cs_l)


#get_posterior_f_cons_partial = ps.keyed_partial(gp.get_pystan_curve_fit_truncated_normal_noise_posterior_f, l_m, cs_l)

num_iter, num_chains, num_processes = 2000, 4, 4

get_posterior_f = gp.parallel_merged_get_posterior_f(get_posterior_f_cons_partial, num_iter, num_chains, num_processes)

#get_posterior_f = gp.merged_get_posterior_f(get_posterior_f_cons_partial, num_iter, num_chains)

cv_f = ps.self_test_cv_f()
#cv_f = ps.cv_fold_f(3)


times = [1,2,4,8,12,18,24,30,36,42,48]

print get_posterior_f.get_key()



color_list = ['b','g','r','c','m','y','k']


all_pids = [_datum.pid for _datum in data]



bad_folder = '%s/%s' % (global_stuff.for_dropbox, 'good_bad/bad')
good_folder = '%s/%s' % (global_stuff.for_dropbox, 'good_bad/good')

def try_it(pid):
    single_data = data_module.get_single_pid_data(pid)
    try:
        posterior = get_posterior_f(single_data)
    except Exception, e:
        f = open(bad_folder + '/' + pid, 'w')
        f.write(str(e))
        print e
        f.close()
    else:
        f = open(good_folder + '/' + pid, 'w')
        f.write(str(posterior['As'].mean()))
        f.write('\n')
        f.write(str(posterior['Bs'].mean()))
        f.write('\n')
        f.write(str(posterior['Cs'].mean()))
        f.write('\n')
        f.close()

#map(try_it, all_pids)
#ps.parallel_map(try_it, all_pids, 4)
        

"""
for train_data, test_data in cv_f(data):                                                                                                                                                                  
    try:
        posterior = get_posterior_f(train_data, test_data)
        #ps.abc_trace_plots()(posterior)
        #ps.summary_traces()(posterior)
        #gelman = ps.gelman_statistic_f()(posterior)
    except TypeError:
        posterior = get_posterior_f(train_data)
        #ps.abc_trace_plots()(posterior)
        #ps.summary_traces()(posterior)
        #gelman = ps.gelman_statistic_f()(posterior)
"""



trainer_plotter_cons_tuples = [\
#        (pt.t_point_trainer_from_t_distribution_trainer(pt.t_distribution_trainer_from_abc_distribution_trainer(pt.builtin_regular_abc_distribution_trainer(get_posterior_f)), ps.mean_f()), ps.keyed_partial(plotters.t_point_predictor_curve_plotter, 0,50,50,'orange','beta_pointwise_mean')), \
#            (pt.t_point_trainer_from_t_distribution_trainer(pt.t_distribution_trainer_from_abc_distribution_trainer(pt.builtin_regular_abc_distribution_trainer(get_posterior_f)), ps.median_f()), ps.keyed_partial(plotters.t_point_predictor_curve_plotter, 0,50,50,'cyan','pointwise_median')), \
#            (pt.builtin_auto_abc_distribution_trainer(get_posterior_f), ps.keyed_partial(plotters.abc_distribution_predictor_curve_plotter, 0.01,150,0,50,50,'orange','beta_regular')), \
#        (pt.t_point_trainer_from_abc_point_trainer(pt.abc_point_trainer_from_abc_distribution_trainer(pt.builtin_regular_abc_distribution_trainer(get_posterior_f), ps.mean_f())), ps.keyed_partial(plotters.t_point_predictor_curve_plotter, 0,50,50,'green','beta_mean_abc')), \
#                (pt.builtin_auto_abc_distribution_trainer(get_posterior_f), ps.keyed_partial(plotters.abc_distribution_predictor_curve_plotter, 0.01,150,0,50,50,'orange',r'$f(t)$')), \
#                (pt.builtin_auto_f_star_distribution_trainer(get_posterior_f,500), ps.keyed_partial(plotters.t_distribution_predictor_curve_plotter_perturbed_ts, 1,150,0,50,50,'red',r'$f^*(t)$')),\
#                (pt.t_point_trainer_from_t_distribution_trainer(pt.builtin_auto_f_star_distribution_trainer(get_posterior_f,500), ps.median_f()), ps.keyed_partial(plotters.t_point_predictor_curve_plotter, 0,50,50, 'magenta',r'$f^*(t)$ median')),\
#            (pt.t_point_trainer_from_t_distribution_trainer(pt.t_distribution_trainer_from_abc_distribution_trainer(pt.builtin_auto_abc_distribution_trainer(get_posterior_f)), ps.median_f()), ps.keyed_partial(plotters.t_point_predictor_curve_plotter, 0,50,25,'blue',r'$f(t)$ median')), \
#    (pt.t_point_trainer_from_t_distribution_trainer(pt.generic_auto_f_star_distribution_trainer(get_posterior_f,500), ps.median_f()), ps.keyed_partial(plotters.t_point_predictor_curve_plotter, 0,50,50, 'magenta',r'$f^*(t)$ median')),\
##    (pt.generic_auto_abc_distribution_trainer(get_posterior_f, 150), ps.keyed_partial(plotters.abc_distribution_predictor_curve_plotter, 0.5,150,0,50,50,'red',r'$f(t)$')), \
#    (pt.generic_auto_abc_distribution_top_k_trainer(get_posterior_f, 10), ps.keyed_partial(plotters.abc_distribution_predictor_curve_plotter, 1.0,150,0,50,50,'magenta',r'$f(t)$')), \
###    (pt.generic_auto_abc_phi_m_distribution_trainer(get_posterior_f, 150), ps.keyed_partial(plotters.abc_phi_m_distribution_predictor_curve_plotter, 0.5,150,0,50,50,plt.cm.coolwarm,r'$f(t)$ colored')), \
    (pt.generic_auto_abc_distribution_trainer(get_posterior_f, 150), ps.keyed_partial(plotters.abc_distribution_predictor_chainwise_curve_plotter, 0.3,150,0,50,50,color_list,r'$f(t)$ colored')), \
            (pt.t_point_trainer_from_t_distribution_trainer(pt.t_distribution_trainer_from_abc_distribution_trainer(pt.generic_auto_abc_distribution_trainer(get_posterior_f, 500)), ps.median_f()), ps.keyed_partial(plotters.t_point_predictor_curve_plotter, 0,50,25,'green',r'$f(t)$ median')), \
            (pt.t_point_trainer_from_t_distribution_trainer(pt.t_distribution_trainer_from_abc_distribution_trainer(pt.generic_auto_abc_distribution_trainer(get_posterior_f, 500)), ps.median_f()), ps.keyed_partial(plotters.point_t_discrete_print_prob_plotter, normal, 'green',r'$f(t)$ median')), \
#            (pt.logregshape_trainer(times), ps.keyed_partial(plotters.point_t_discrete_plotter, 'blue', 'logreg_shape')),\
#            (pt.logreg_trainer(times), ps.keyed_partial(plotters.point_t_discrete_plotter, 'cyan', 'logreg')),\
                (pt.t_point_trainer_from_abc_point_trainer(pt.curve_fit_cheating_abc_point_trainer()),ps.keyed_partial(plotters.t_point_predictor_curve_plotter, 0,50,50,'orange','cheating')),\
#                (pt.t_point_trainer_from_abc_point_trainer(pt.curve_fit_cheating_abc_point_trainer()),ps.keyed_partial(plotters.point_t_discrete_print_prob_plotter, normal, 'orange','cheating')),\
#                (pt.builtin_auto_f_star_nonshared_noise_distribution_trainer(get_posterior_f,500), ps.keyed_partial(plotters.t_distribution_predictor_curve_plotter_perturbed_ts, 1,150,0,50,50,'red',r'$f^*(t)$')),\
    ]


#ps.plot_all_predictions_fig_f(trainer_plotter_cons_tuples, cv_f)(data)


"""
full_trainer = pt.t_point_trainer_from_t_distribution_trainer(pt.t_distribution_trainer_from_abc_distribution_trainer(pt.generic_builtin_abc_distribution_trainer(get_posterior_f, 500)), ps.median_f())
#full_trainer = pt.t_point_trainer_from_t_distribution_trainer(pt.t_distribution_trainer_from_abc_distribution_trainer(pt.generic_auto_abc_distribution_trainer(get_posterior_f, 500)), ps.median_f())
#full_predictor = full_trainer(data, data)
full_predictor = full_trainer(data)
full_plotter = ps.shape_plotter(full_predictor, 0, 100, 200)

logreg_trainer = pt.logreg_trainer(times)
logreg_predictor = logreg_trainer(data)
logreg_plotter = ps.discrete_shape_plotter(logreg_predictor)

plotters = [full_plotter, logreg_plotter]


find the patients in each bin

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

datum_by_age = [[None for x in xrange(4)] for y in xrange(3)]
datum_by_init = [[None for x in xrange(3)] for y in xrange(4)]

for _datum in data:
    age = get_age_bin(_datum)
    init = get_init_bin(_datum)
    datum_by_age[age][init] = _datum
    datum_by_init[init][age] = _datum

_cm = plt.cm.coolwarm

age_colors = [_cm(0.0), _cm(0.5), _cm(1.0)]
init_colors = [_cm(0.0), _cm(0.33), _cm(.66), _cm(1.0)]

age_titles = ['age 0 to 55','age 55 to 65','age 65+']
init_titles = ['init 41-', 'init 41 to 60', 'init 60 to 80', 'init 80 to 100']

#ps.multiple_figures_to_pdf(plot_stuff(plotters, datum_by_age, init_colors, init_titles, age_titles), '%s/%s/%s' % (global_stuff.for_dropbox, 'stratified', 'by_age'))
#ps.multiple_figures_to_pdf(plot_stuff(plotters, datum_by_init, age_colors, age_titles, init_titles), '%s/%s/%s' % (global_stuff.for_dropbox, 'stratified', 'by_init'))

"""





#pdb.set_trace()

for train_data, test_data in cv_f(data):                                                                                                                                                                  
    try:
        posterior = get_posterior_f(train_data, test_data)
        #ps.abc_trace_plots()(posterior)
        #ps.summary_traces()(posterior)
        #gelman = ps.gelman_statistic_f()(posterior)
    except TypeError:

        posterior = get_posterior_f(train_data)
        #ps.abc_trace_plots()(posterior)
        #ps.summary_traces()(posterior)
        #gelman = ps.gelman_statistic_f()(posterior)


#gp.plot_ind_phi_m_posterior(500)(posterior)



try:
    generic_plot_all_predictions_fig_f(trainer_plotter_cons_tuples, cv_f, ps.keyed_partial(plot_predictions_fig_f))(data)
except:
    print_traceback()
    pdb.set_trace()

pdb.set_trace()

"""


ps.plot_diffcovs_posterior_f(3,2,cv_f, get_posterior_f)(data)



"""
                                                                                                                                                                                                        

    





log_loss_f = ps.loss_f(ps.get_true_val(), ps.scaled_logistic_loss_f(10))
signed_loss_f = ps.loss_f(ps.get_true_val(), ps.signed_loss_f())

trainers = [\
#        pt.t_point_trainer_from_t_distribution_trainer(pt.t_distribution_trainer_from_abc_distribution_trainer(pt.builtin_regular_abc_distribution_trainer(get_posterior_f)), ps.median_f()), \
             #    pt.t_point_trainer_from_t_distribution_trainer(pt.t_distribution_trainer_from_abc_distribution_trainer(pt.builtin_regular_abc_distribution_trainer(get_posterior_f)), ps.mean_f()),\
    pt.logregshape_trainer(times),\
             pt.logreg_trainer(times),\
        pt.t_point_trainer_from_t_distribution_trainer(pt.t_distribution_trainer_from_abc_distribution_trainer(pt.generic_builtin_abc_distribution_trainer(get_posterior_f, 500)), ps.median_f()), \
#        pt.t_point_trainer_from_t_distribution_trainer(pt.t_distribution_trainer_from_abc_distribution_trainer(pt.builtin_auto_abc_distribution_trainer(get_posterior_f)), ps.median_f()),\
#        pt.t_point_trainer_from_t_distribution_trainer(pt.builtin_auto_f_star_distribution_trainer(get_posterior_f,500), ps.median_f()),\
        pt.t_point_trainer_from_abc_point_trainer(pt.curve_fit_cheating_abc_point_trainer()),\
             ]

percentiles = [.25,.5,.75]
#percentiles = []

trainer_colors = [\
    'orange',\
        'blue',\
        'red',\
#        'black',\
        'green',\
        ]
trainer_names = [\
    'logregshape'\
        ,'logreg',\
        r'$f(t)$ median',\
#        r'$f^*(t)$ median',\
        'cheating curve fit',\
        ]


ps.model_comparer_f(trainers, cv_f, signed_loss_f, percentiles, times, trainer_colors, trainer_names)(data)
ps.model_comparer_f(trainers, cv_f, log_loss_f, percentiles, times, trainer_colors, trainer_names)(data)

pdb.set_trace()




class stratified_model_comparer_f(possibly_cached):
    """
    hard code the loss functions used
    """

    def __init__(self, trainers, cv_f, loss_f, percentiles, times, display_colors, display_names, stratify_names, stratify_fs):
        self.trainers, self.cv_f, self.loss_f, self.percentiles, self.times, self.stratify_names, self.stratify_fs = trainers, cv_f, loss_f, percentiles, times, stratify_names, stratify_fs
        self.display_colors, self.display_names = display_colors, display_names

    @save_to_file
    @memoize
    def __call__(self, data):
        fig = plt.figure()
        axes = []
        for stratify_name,i in itertools.izip(self.stratify_names,xrange(4)):
            axes.append(fig.add_subplot(2,2,i+1))
        fig.subplots_adjust(hspace=.3,wspace=.3)
        fig.suptitle('stratified losses under %s' % self.loss_f.get_key())
        for trainer, display_color, display_name in itertools.izip(self.trainers, self.display_colors, self.display_names):
            scores_getter = cross_validated_scores_f(trainer, self.cv_f, self.times)
            scores = scores_getter(data)
            for ax, stratify_f, stratify_name in itertools.izip(axes, self.stratify_fs, self.stratify_names):
                #this_scores =  keyed_DataFrame(scores.loc[:,[pid for pid in scores.columns if stratify_f(pid)]])
                this_scores = gd.filtered_by_column_label_DataFrame(stratify_f)(scores)
                _performance_series_f = performance_series_f(self.loss_f, self.percentiles, data, self.times)
                this_perfs = _performance_series_f(this_scores)
                add_performances_to_ax(ax, this_perfs, display_color, display_name)
                ax.set_xlim(-1.0,50)
                ax.set_ylim(-0.5,1.1)
                ax.set_title('%s n=%d' % (stratify_name, this_scores.shape[1]))
                ax.set_xlabel('time')
                ax.set_ylabel('loss')
        ax.legend(prop={'size':6})
        #fig.show()
        return fig

    def get_introspection_key(self):
        return '%s_%s_%s_%s_%s' % ('strat',self.loss_f.get_key(), self.cv_f.get_key(), self.trainers[0].get_key(), self.stratify_names[0])
        return '%s_%s_%s_%s' % ('bs', self.trainers.get_key(), self.cv_f.get_key(), self.loss_f.get_key())

    def key_f(self, data):
        return '%s_%s' % (self.get_key(), data.get_key())

    def location_f(self, data):
        return '%s/%s' % (global_stuff.for_dropbox,'betweenmodel_perfs')

    print_handler_f = staticmethod(figure_to_pdf)

    read_f = staticmethod(not_implemented_f)



from recovery_curve.hard_coded_objects.feature_sets import default_age_categorical_f_all_levels, default_initial_categorical_f_all_levels

age_stratify_fs = default_age_categorical_f_all_levels
age_stratify_names = ['low_age', 'med_age', 'high_age']

initial_stratify_fs = default_initial_categorical_f_all_levels
initial_stratify_names = ['init_level_0_41','init_level_41_60','init_level_60_80','init_level_80_100']


def time_1_f(pid):
    return data_module.ys_f(pid)[1]

def scaled_time_1_f(pid):
    return time_1_f(pid) / data_module.init_f(pid)

time_1_level_categorical_f_all_levels = set_hard_coded_key_dec(gd.get_categorical_fs_all_levels,'time_1_f')(time_1_f, [bin(None,0.25), bin(.25,.50), bin(.50,.75), bin(.75,1.0)])
time_1_names = ['time_1_val_0_0.25', 'time_1_val_0.25_0.50', 'time_1_val_0.50_0.75', 'time_1_val_0.75_1.0']

scaled_time_1_level_categorical_f_all_levels = set_hard_coded_key_dec(gd.get_categorical_fs_all_levels,'time_1_f')(scaled_time_1_f, [bin(None,0.25), bin(.25,.50), bin(.50,.75), bin(.75,1.0)])
scaled_time_1_names = ['scaled_time_1_val_0_0.25', 'scaled_time_1_val_0.25_0.50', 'scaled_time_1_val_0.50_0.75', 'scaled_time_1_val_0.75_1.0']

"""

stratified_model_comparer_f(trainers, cv_f, log_loss_f, percentiles, times, trainer_colors, trainer_names, time_1_names, time_1_level_categorical_f_all_levels)(data)
stratified_model_comparer_f(trainers, cv_f, signed_loss_f, percentiles, times, trainer_colors, trainer_names, time_1_names, time_1_level_categorical_f_all_levels)(data)


stratified_model_comparer_f(trainers, cv_f, log_loss_f, percentiles, times, trainer_colors, trainer_names, initial_stratify_names, initial_stratify_fs)(data)
stratified_model_comparer_f(trainers, cv_f, log_loss_f, percentiles, times, trainer_colors, trainer_names, age_stratify_names, age_stratify_fs)(data)

stratified_model_comparer_f(trainers, cv_f, signed_loss_f, percentiles, times, trainer_colors, trainer_names, initial_stratify_names, initial_stratify_fs)(data)
stratified_model_comparer_f(trainers, cv_f, signed_loss_f, percentiles, times, trainer_colors, trainer_names, age_stratify_names, age_stratify_fs)(data)


print_diffcovs_posterior_means_effect(cv_f, get_posterior_f)(data)
"""

loss_fs = [\
#    log_loss_f, \
#        signed_loss_f, \
#        scaled_loss_f(log_loss_f), \
        scaled_loss_f(signed_loss_f),\
        ]


stratify_tuples = [\
    [scaled_time_1_level_categorical_f_all_levels, scaled_time_1_names],\
    [time_1_level_categorical_f_all_levels, time_1_names],\
        [age_stratify_fs, age_stratify_names],\
        [initial_stratify_fs, initial_stratify_names],\
        ]

for _loss_f in loss_fs:
    for stratify_fs, stratify_f_names in stratify_tuples:
        stratified_model_comparer_f(trainers, cv_f, _loss_f, percentiles, times, trainer_colors, trainer_names, stratify_f_names, stratify_fs)(data)
