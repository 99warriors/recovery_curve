import recovery_curve.hard_coded_objects.go_two_sim_data as sim_data
import recovery_curve.hard_coded_objects.go_two_real_data as real_data
import recovery_curve.hard_coded_objects.params as hypers
import recovery_curve.prostate_specifics as ps
import recovery_curve.get_posterior_fs as gp
import recovery_curve.predictors_and_trainers as pt
import recovery_curve.plotters as plotters

num_iter = 5000
num_chains = 5

cv_f = ps.self_test_cv_f()

get_pops_f_to_use = ps.returns_whats_given_f(sim_data.pops)

hypers = hypers.default_hyper

get_posterior_f_cons_partial = ps.keyed_partial(gp.get_pystan_diffcovs_beta_noise_posterior_f, get_pops_f_to_use, hypers)
get_posterior_f = ps.merged_get_posterior_f(get_posterior_f_cons_partial, num_iter, num_chains)

trainer_plotter_cons_tuples = [\
    (pt.builtin_auto_abc_distribution_predictor_trainer(get_posterior_f), ps.keyed_partial(plotters.abc_distribution_predictor_curve_plotter, 0.5, 150, 0, 50, 100, 'orange', 'beta_noise')), \
    ]

ps.plot_all_predictions_fig_f(trainer_plotter_cons_tuples, cv_f)

get_posterior_f(sim_data.data)
