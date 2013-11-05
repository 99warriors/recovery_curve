import pdb
import recovery_curve.global_stuff
import recovery_curve.prostate_specifics as ps
import recovery_curve.get_posterior_fs as gp
import recovery_curve.predictors_and_trainers as pt
import recovery_curve.plotters as plotters
import importlib
import sys

from recovery_curve.hard_coded_objects.truncated_normal_regular_get_posterior_f import get_posterior_f as truncated_normal_get_posterior_f
from recovery_curve.hard_coded_objects.beta_regular_get_posterior_f import get_posterior_f as beta_get_posterior_f

trainer_plotter_cons_tuples = [\
    (pt.builtin_auto_abc_distribution_trainer(truncated_normal_get_posterior_f), ps.keyed_partial(plotters.abc_distribution_predictor_curve_plotter, 0.1, 150, 0, 50, 500, 'cyan', 'normal')), \
    (pt.builtin_auto_abc_distribution_trainer(beta_get_posterior_f), ps.keyed_partial(plotters.abc_distribution_predictor_curve_plotter, 0.1, 150, 0, 50, 500, 'red', 'beta')), \
        (pt.t_point_trainer_from_abc_point_trainer(pt.abc_point_trainer_from_abc_distribution_trainer(pt.builtin_auto_abc_distribution_trainer(truncated_normal_get_posterior_f), ps.mean_f())), ps.keyed_partial(plotters.t_point_predictor_curve_plotter, 0,50,150,'orange','normal_pointwise_mean')), \
        (pt.t_point_trainer_from_abc_point_trainer(pt.abc_point_trainer_from_abc_distribution_trainer(pt.builtin_auto_abc_distribution_trainer(beta_get_posterior_f), ps.mean_f())), ps.keyed_partial(plotters.t_point_predictor_curve_plotter, 0,50,150,'green','beta_pointwise_mean')), \
    ]
try:
    posteriors = get_posterior_f(data)
    #stats = ps.gelman_statistic_f()(posteriors)
    #ps.plot_single_posterior_f(3,2)(posteriors)

    ps.plot_all_predictions_fig_f(trainer_plotter_cons_tuples, cv_f)(data)

    loss_f = ps.loss_f(ps.get_true_val(), ps.scaled_logistic_loss_f(10))
    percentiles = [.25,.50,.75]
    trainers = [pointwise_mean_trainer, abc_mean_trainer, logreg_trainer, logreg_shape_trainer]
    
    ps.model_comparer_f(trainers, cv_f, loss_f, percentiles, times)
except Exception, e:
    print e
    ps.print_traceback()
