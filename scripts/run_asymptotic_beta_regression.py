"""
reads in data module, which could be simulated or real
hard code get_posterior_f
calculate loss, plot model fit
"""

import recovery_curve.global_stuff as global_stuff
import recovery_curve.prostate_specifics as ps
from recovery_curve.prostate_specifics import *
import recovery_curve.get_posterior_fs as gp
import recovery_curve.predictors_and_trainers as pt
import recovery_curve.plotters as plotters
import importlib
import sys
import recovery_curve.getting_data as gd
import numpy as np
import pandas


class loss_from_scores(keyed_object):
    """
    returns some sort of loss object given scores
    this one assumes that scores is a scalar, and that truth is a set of numbers
    will hardcoded everything - for a patient, calculate average signed loss
    """
    def __init__(self, data):
        self.data = data

    def __call__(self, scores):
        losses = {}
        for pid, score in scores.iteritems():
            losses[pid] = np.mean([y - score for y in data.d[pid].ys])
        return keyed_Series(losses)

class cross_validated_scores(keyed_object):
    """
    returns some sort of scores object.  convention will be to have columns be patients if object is df.
    this one assumes that the thing trainer returns is just a number
    so cross_validated_score objects are differentiated by what they expect predictor to return
    """
    def __init__(self, cv_f, trainer):
        self.cv_f, self.trainer = cv_f, trainer

    def __call__(self, data):
        scores = {}
        for train_data, test_data in self.cv_f(data):
            if self.trainer.normal():
                predictor = self.trainer(train_data)
            else:
                predictor = self.trainer(train_data, test_data)
            for datum in test_data:
                scores[datum.pid] = predictor(datum)
        return keyed_Series(scores)

data = importlib.import_module(sys.argv[1]).data

num_iter = 2500

num_chains = 4

num_processes = 4

pop_val = 0.5

l = 15

l_m = 15

c = 1



predictor_max_num = 1000

cv_f = ps.self_test_cv_f()

#get_posterior_f_cons_partial = ps.keyed_partial(gp.get_pystan_simple_hierarchical_beta_regression_f, pop_val, l, l_m, c)

get_posterior_f_cons_partial = ps.keyed_partial(gp.get_pystan_curve_fit_beta_noise_posterior_f, l_m)

get_posterior_f = gp.parallel_merged_get_posterior_f(get_posterior_f_cons_partial, num_iter, num_chains, num_processes)

#get_posterior_f = gp.merged_get_posterior_f(get_posterior_f_cons_partial, num_iter, num_chains)

obs_trainer = pt.auto_beta_hierarchical_obs_val_distribution_trainer(get_posterior_f, predictor_max_num)



obs_scores = cross_validated_scores(cv_f, obs_trainer)(data)

obs_losses = loss_from_scores(data)(obs_scores)

latent_trainer = pt.auto_beta_hierarchical_latent_val_distribution_trainer(get_posterior_f, predictor_max_num)

latent_scores = cross_validated_scores(cv_f, latent_trainer)(data)

latent_losses = loss_from_scores(data)(latent_scores)

pdb.set_trace()

trainer_plotter_cons_tuples = [\
    [obs_trainer, ps.keyed_partial(plotters.scalar_distribution_predictor_plotter, 0.2, 50, 0.0, 1.0, 'yellow', 'obs')],\
    [latent_trainer, ps.keyed_partial(plotters.scalar_distribution_predictor_plotter, 0.2, 50, 0.0, 1.0, 'orange', 'latent')],\
        ]

ps.generic_plot_all_predictions_fig_f(trainer_plotter_cons_tuples, cv_f, ps.keyed_partial(ps.plot_scalar_prediction_fig))(data)


