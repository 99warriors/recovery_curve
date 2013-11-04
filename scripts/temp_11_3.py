import pdb
import recovery_curve.global_stuff
import recovery_curve.hard_coded_objects.go_two_sim_data as sim_data
import recovery_curve.hard_coded_objects.go_two_real_data as real_data
import recovery_curve.hard_coded_objects.hypers as hard_coded_hypers
import recovery_curve.prostate_specifics as ps
import recovery_curve.get_posterior_fs as gp
import recovery_curve.predictors_and_trainers as pt
import recovery_curve.plotters as plotters

data = sim_data.data

num_iter = 5000
num_chains = 5
get_posterior_f_cons = gp.get_pystan_diffcovs_beta_noise_posterior_f

times = [1,2,4,8,12,18,24,30,36,42,48]

cv_f = ps.self_test_cv_f()

get_pops_f_to_use = ps.returns_whats_given_f(sim_data.pops)

hypers = hard_coded_hypers.default_hyper

get_posterior_f_cons_partial = ps.keyed_partial(get_posterior_f_cons, get_pops_f_to_use, hypers)

get_posterior_f = gp.merged_get_posterior_f(get_posterior_f_cons_partial, num_iter, num_chains)

pointwise_mean_trainer = t_point_predictor_from_t_distribution_predictor(t_distribution_predictor_from_abc_distribution_predictor(pt.builtin_auto_abc_distribution_predictor_trainer(get_posterior_f)), ps.mean_f())
abc_mean_trainer = t_point_predictor_from_abc_point_predictor(abc_point_trainer_from_abc_distribution_trainer(pt.builtin_auto_abc_distribution_predictor_trainer(get_posterior_f), ps.mean_f()))
logreg_trainer = pt.logreg_trainer(times)
logreg_shape_trainer = pt.logregshape_trainer(times)


trainer_plotter_cons_tuples = [\
    (pt.builtin_auto_abc_distribution_predictor_trainer(get_posterior_f), ps.keyed_partial(plotters.abc_distribution_predictor_curve_plotter, 0.5, 150, 0, 50, 100, 'orange', 'beta_noise')), \
        (pointwise_mean_trainer, t_point_predictor_curve_plotter(0,50,150,'red','pointwise_mean')), \
        (abc_mean_trainer, t_point_predictor_curve_plotter(0,50,150,'cyan','mean_abc')), \
        (logreg_trainer,point_t_discrete_plotter('blue','logreg')), \
        (logreg_shape_trainer,point_t_discrete_plotter('magenta','logreg_shape')), \
    ]
posteriors = get_posterior_f(data)
stats = ps.gelman_statistic_f()(posteriors)
ps.plot_all_predictions_fig_f(trainer_plotter_cons_tuples, cv_f)(data)

loss_f = ps.loss_f(ps.get_true_val(), ps.scaled_logistic_loss(10))

percentiles = [.25,.50,.75]

trainers = [pointwise_mean_trainer, abc_mean_trainer, logreg_trainer, logreg_shape_trainer]

ps.model_comparer_f(trainers, cv_f, loss_f, percentiles, times)


