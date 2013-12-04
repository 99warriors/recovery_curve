import pdb
import recovery_curve.global_stuff
import recovery_curve.global_stuff
import recovery_curve.prostate_specifics as ps
import recovery_curve.get_posterior_fs as gp
import recovery_curve.predictors_and_trainers as pt
import recovery_curve.plotters as plotters
import importlib
import sys



"""                                                                                                                                                                                                        
goal of this script is to get posteriors, not even to see results                                                                                                                                          
"""


data = importlib.import_module(sys.argv[1]).data
get_posterior_f_module = importlib.import_module(sys.argv[2])
get_posterior_f = get_posterior_f_module.get_posterior_f
cv_f = ps.cv_fold_f(4)

times = [1,2,4,8,12,18,24,30,36,42,48]

print get_posterior_f.get_key()


trainer_plotter_cons_tuples = [\
        (pt.t_point_trainer_from_t_distribution_trainer(pt.t_distribution_trainer_from_abc_distribution_trainer(pt.builtin_regular_abc_distribution_trainer(get_posterior_f)), ps.mean_f()), ps.keyed_partial(plotters.t_point_predictor_curve_plotter, 0,50,50,'orange','beta_pointwise_mean')), \
            (pt.t_point_trainer_from_t_distribution_trainer(pt.t_distribution_trainer_from_abc_distribution_trainer(pt.builtin_regular_abc_distribution_trainer(get_posterior_f)), ps.median_f()), ps.keyed_partial(plotters.t_point_predictor_curve_plotter, 0,50,50,'cyan','beta_pointwise_median')), \
#        (pt.t_point_trainer_from_abc_point_trainer(pt.abc_point_trainer_from_abc_distribution_trainer(pt.builtin_regular_abc_distribution_trainer(get_posterior_f), ps.mean_f())), ps.keyed_partial(plotters.t_point_predictor_curve_plotter, 0,50,50,'green','beta_mean_abc')), \
            (pt.logreg_trainer(times), ps.keyed_partial(plotters.point_t_discrete_plotter, 'red', 'logreg')),\
            (pt.logregshape_trainer(times), ps.keyed_partial(plotters.point_t_discrete_plotter, 'blue', 'logreg_shape')),\
    ]

ps.plot_all_predictions_fig_f(trainer_plotter_cons_tuples, cv_f)(data)

pdb.set_trace()




"""

#ps.plot_diffcovs_posterior_f(3,2,cv_f, get_posterior_f)(data)

#pdb.set_trace()


                                                                                                                                                                                                        
for train_data, test_data in cv_f(data):                                                                                                                                                                   
    try:                                                                                                                                                                                                   
        posterior = get_posterior_f(train_data, test_data)                                                                                                                                                 
    except TypeError:                                                                                                                                                                                      
        posterior = get_posterior_f(train_data)
    gelman = ps.gelman_statistic_f()(posterior)
    pdb.set_trace()
"""



#loss_f = ps.loss_f(ps.get_true_val(), ps.scaled_logistic_loss_f(10))
loss_f = ps.loss_f(ps.get_true_val(), ps.signed_loss_f())

trainers = [\
    pt.logregshape_trainer(times),\
        pt.t_point_trainer_from_t_distribution_trainer(pt.t_distribution_trainer_from_abc_distribution_trainer(pt.builtin_regular_abc_distribution_trainer(get_posterior_f)), ps.median_f()), \
             #    pt.t_point_trainer_from_t_distribution_trainer(pt.t_distribution_trainer_from_abc_distribution_trainer(pt.builtin_regular_abc_distribution_trainer(get_posterior_f)), ps.mean_f()),\
             pt.logreg_trainer(times),\
             ]

percentiles = [.25,.5,.75]

ps.model_comparer_f(trainers, cv_f, loss_f, percentiles, times)(data)



