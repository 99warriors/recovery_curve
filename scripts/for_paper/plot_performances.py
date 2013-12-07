import options
import recovery_curve.prostate_specifics as ps
import matplotlib.pyplot as plt
import recovery_curve.global_stuff as global_stuff
import random
import recovery_curve.get_posterior_fs as gp
import recovery_curve.getting_data as gd
import numpy as np

"""
under the 2 loss functions, plot performance for:
- our model
- logistic regression
- prior shape
- prior curve
"""

trainers = [\
    options.our_data_point_trainer,\
        options.logreg_trainer,\
        options.avg_shape_trainer,\
        options.avg_curve_trainer,
    ]

trainer_names = [\
    'our model',\
        'logistic regression',\
        'avg shape',\
        'avg curve',\
]

trainer_colors = [\
    'blue',\
        'red',\
        'magenta',\
        'cyan',
    ]

cv_f = options.real_performance_cv_f
loss_fs = options.loss_fs
loss_f_names = options.loss_f_names

for loss_f, loss_f_name in zip(loss_fs, loss_f_names):
    fig = ps.model_comparer(trainers, cv_f, loss_f, options.percentiles_to_plot, options.real_times, trainer_names, trainer_colors)
    fig.suptitle(loss_f_name)
    ps.figure_to_pdf(fig, options.get_outfile_path('%s_%s' % ('performance', loss_f_name)))

