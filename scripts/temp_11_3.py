import pdb
import recovery_curve.global_stuff
import recovery_curve.prostate_specifics as ps
import recovery_curve.get_posterior_fs as gp
import recovery_curve.predictors_and_trainers as pt
import recovery_curve.plotters as plotters
import importlib
import sys



"""
goal of this script is to see self_cv results
"""


data = importlib.import_module(sys.argv[1]).data
get_posterior_f = importlib.import_module(sys.argv[2]).get_posterior_f

times = [1,2,4,8,12,18,24,30,36,42,48]
cv_f = ps.self_test_cv_f()

pointwise_mean_trainer = pt.t_point_trainer_from_t_distribution_trainer(pt.t_distribution_trainer_from_abc_distribution_trainer(pt.builtin_auto_abc_distribution_trainer(get_posterior_f)), ps.mean_f())
abc_mean_trainer = pt.t_point_trainer_from_abc_point_trainer(pt.abc_point_trainer_from_abc_distribution_trainer(pt.builtin_auto_abc_distribution_trainer(get_posterior_f), ps.mean_f()))
logreg_trainer = pt.logreg_trainer(times)
logreg_shape_trainer = pt.logregshape_trainer(times)

trainer_plotter_cons_tuples = [\
    (pt.builtin_auto_abc_distribution_trainer(get_posterior_f), ps.keyed_partial(plotters.abc_distribution_predictor_curve_plotter, 0.5, 150, 0, 50, 100, 'orange', 'beta_noise')), \
        (pointwise_mean_trainer, ps.keyed_partial(plotters.t_point_predictor_curve_plotter, 0,50,150,'red','pointwise_mean')), \
        (abc_mean_trainer, ps.keyed_partial(plotters.t_point_predictor_curve_plotter, 0,50,150,'cyan','mean_abc')), \
        (logreg_trainer, ps.keyed_partial(plotters.point_t_discrete_plotter, 'blue','logreg')), \
        (logreg_shape_trainer, ps.keyed_partial(plotters.point_t_discrete_plotter, 'magenta','logreg_shape')), \
    ]

posteriors = get_posterior_f(data)
stats = ps.gelman_statistic_f()(posteriors)
ps.plot_single_posterior_f(3,2)(posteriors)

ps.plot_all_predictions_fig_f(trainer_plotter_cons_tuples, cv_f)(data)

loss_f = ps.loss_f(ps.get_true_val(), ps.scaled_logistic_loss(10))
percentiles = [.25,.50,.75]
trainers = [pointwise_mean_trainer, abc_mean_trainer, logreg_trainer, logreg_shape_trainer]

ps.model_comparer_f(trainers, cv_f, loss_f, percentiles, times)


