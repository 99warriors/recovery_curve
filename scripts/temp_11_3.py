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
get_posterior_f_module = importlib.import_module(sys.argv[2])
get_posterior_f = get_posterior_f_module.get_posterior_f
cv_f = ps.cv_fold_f(4)


for train_data, test_data in cv_f(data):
    try:
        posterior = get_posterior_f(train_data, test_data)
    except TypeError:
        posterior = get_posterior_f(train_data)



