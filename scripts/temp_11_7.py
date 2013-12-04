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


"""                                                                                                                                                                                                        
goal of this script is to get posteriors, not even to see results                                                                                                                                          
"""


data = importlib.import_module(sys.argv[1]).data
data_module = importlib.import_module(sys.argv[1])
get_posterior_f_module = importlib.import_module(sys.argv[2])
get_posterior_f = get_posterior_f_module.get_posterior_f
#cv_f = ps.cv_fold_f(4)
cv_f = ps.self_test_cv_f()

#data = gd.reduce_data_f(0.5)(data)

times = [1,2,4,8,12,18,24,30,36,42,48]

print get_posterior_f.get_key()






trainer_plotter_cons_tuples = [\
#        (pt.t_point_trainer_from_t_distribution_trainer(pt.t_distribution_trainer_from_abc_distribution_trainer(pt.builtin_regular_abc_distribution_trainer(get_posterior_f)), ps.mean_f()), ps.keyed_partial(plotters.t_point_predictor_curve_plotter, 0,50,50,'orange','beta_pointwise_mean')), \
#            (pt.t_point_trainer_from_t_distribution_trainer(pt.t_distribution_trainer_from_abc_distribution_trainer(pt.builtin_regular_abc_distribution_trainer(get_posterior_f)), ps.median_f()), ps.keyed_partial(plotters.t_point_predictor_curve_plotter, 0,50,50,'cyan','pointwise_median')), \
#            (pt.builtin_auto_abc_distribution_trainer(get_posterior_f), ps.keyed_partial(plotters.abc_distribution_predictor_curve_plotter, 0.01,150,0,50,50,'orange','beta_regular')), \
#        (pt.t_point_trainer_from_abc_point_trainer(pt.abc_point_trainer_from_abc_distribution_trainer(pt.builtin_regular_abc_distribution_trainer(get_posterior_f), ps.mean_f())), ps.keyed_partial(plotters.t_point_predictor_curve_plotter, 0,50,50,'green','beta_mean_abc')), \
#                (pt.builtin_auto_abc_distribution_trainer(get_posterior_f), ps.keyed_partial(plotters.abc_distribution_predictor_curve_plotter, 0.01,150,0,50,50,'orange',r'$f(t)$')), \
#                (pt.builtin_auto_f_star_distribution_trainer(get_posterior_f,500), ps.keyed_partial(plotters.t_distribution_predictor_curve_plotter_perturbed_ts, 1,150,0,50,50,'red',r'$f^*(t)$')),\
#                (pt.t_point_trainer_from_t_distribution_trainer(pt.builtin_auto_f_star_distribution_trainer(get_posterior_f,500), ps.median_f()), ps.keyed_partial(plotters.t_point_predictor_curve_plotter, 0,50,50, 'magenta',r'$f^*(t)$ median')),\
#            (pt.t_point_trainer_from_t_distribution_trainer(pt.t_distribution_trainer_from_abc_distribution_trainer(pt.builtin_auto_abc_distribution_trainer(get_posterior_f)), ps.median_f()), ps.keyed_partial(plotters.t_point_predictor_curve_plotter, 0,50,50,'green',r'$f(t)$ median')), \
            (pt.logregshape_trainer(times), ps.keyed_partial(plotters.point_t_discrete_plotter, 'blue', 'logreg_shape')),\
            (pt.logreg_trainer(times), ps.keyed_partial(plotters.point_t_discrete_plotter, 'cyan', 'logreg')),\
    ]


ps.plot_all_predictions_fig_f(trainer_plotter_cons_tuples, cv_f)(data)

pdb.set_trace()

for train_data, test_data in cv_f(data):                                                                                                                                                                  
    try:                                                                                                                                                                                                  
        posterior = get_posterior_f(train_data, test_data)
        ps.abc_trace_plots()(posterior)
        ps.summary_traces()(posterior)
        gelman = ps.gelman_statistic_f()(posterior)
    except TypeError:
        posterior = get_posterior_f(train_data)
        ps.abc_trace_plots()(posterior)
        ps.summary_traces()(posterior)
        gelman = ps.gelman_statistic_f()(posterior)


#pdb.set_trace()


"""


ps.plot_diffcovs_posterior_f(3,2,cv_f, get_posterior_f)(data)



"""
                                                                                                                                                                                                        
for train_data, test_data in cv_f(data):                                                                                                                                                                   
    try:                                                                                                                                                                                                   
        posterior = get_posterior_f(train_data, test_data)                                                                                                                                                 
    except TypeError:                                                                                                                                                                                      
        posterior = get_posterior_f(train_data)
    





log_loss_f = ps.loss_f(ps.get_true_val(), ps.scaled_logistic_loss_f(10))
signed_loss_f = ps.loss_f(ps.get_true_val(), ps.signed_loss_f())

trainers = [\
#        pt.t_point_trainer_from_t_distribution_trainer(pt.t_distribution_trainer_from_abc_distribution_trainer(pt.builtin_regular_abc_distribution_trainer(get_posterior_f)), ps.median_f()), \
             #    pt.t_point_trainer_from_t_distribution_trainer(pt.t_distribution_trainer_from_abc_distribution_trainer(pt.builtin_regular_abc_distribution_trainer(get_posterior_f)), ps.mean_f()),\
    pt.logregshape_trainer(times),\
             pt.logreg_trainer(times),\
        pt.t_point_trainer_from_t_distribution_trainer(pt.t_distribution_trainer_from_abc_distribution_trainer(pt.builtin_auto_abc_distribution_trainer(get_posterior_f)), ps.median_f()),\
        pt.t_point_trainer_from_t_distribution_trainer(pt.builtin_auto_f_star_distribution_trainer(get_posterior_f,500), ps.median_f()),\
             ]

#percentiles = [.25,.5,.75]
percentiles = []

trainer_colors = ['orange','blue','red','black']
trainer_names = ['logregshape','logreg',r'$f(t)$ median',r'$f^*(t)$ median']

"""
ps.model_comparer_f(trainers, cv_f, log_loss_f, percentiles, times, trainer_colors, trainer_names)(data)
ps.model_comparer_f(trainers, cv_f, signed_loss_f, percentiles, times, trainer_colors, trainer_names)(data)
"""





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
