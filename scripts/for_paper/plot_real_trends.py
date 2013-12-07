import options
import recovery_curve.prostate_specifics as ps
import matplotlib.pyplot as plt
import recovery_curve.global_stuff as global_stuff
import random
import recovery_curve.get_posterior_fs as gp
import recovery_curve.getting_data as gd
import numpy as np

"""
makes 2 plots, showing trend by age, also controlled for init, and vice versa for age
"""

outfile_age = options.get_outfile_path('age_trend')
outfile_init = options.get_outfile_path('init_trend')
data = options.filtered_data
trainer = options.our_data_point_trainer
curve_plot_times = np.linspace(0,50,100)


options.real_get_posterior_f(data, data)

pdb.set_trace()

"""
meat
"""

predictor = trainer(data, data)

repr_d = {}
for datum in data:
    repr_d[(options.get_init_bin(datum),options.get_age_bin(age))] = datum

pred_d = {key:pandas.Series([predictor(datum,t) for t in curve_plot_times], index=curve_plot_times) for key,datum in repr_d.iteritems()}

pred_by_init = [[pred_d[(j,i)] for i in xrange(options.num_age_bins)] for j in xrange(options.num_init_bins)]
agg_by_init = map(ps.aggregate_curve_f(), pred_by_init)

pred_by_age = [[pred_d[(i,j)] for j in xrange(options.num_init_bins)] for i in xrange(options.num_age_bins)]
agg_by_age = map(ps.aggregate_curve_f(), pred_by_age)

"""
show trend in init
"""

init_fig = plt.figure()
ax = init_fig.add_subplot(1,2,1)
for curve, label in zip(agg_by_init, options.init_titles):
    ax.plot(curve.index, curve, label = label)
ax.set_xlabel('months')
ax.set_ylabel('scaled f')
ax.set_title('aggregate curve by init')

for i, age_label in zip(xrange(options.num_age_bin), options.age_titles):
    ax = init_fig.add_subplot(3,2,i+1)
    ax.set_title(age_label)
    for j in xrange(options.num_init_bins):
        curve = pred_d[(j,i)]
        ax.plot(curve.index, curve)

ps.figure_to_pdf(init_fig, outfile_init)

"""
show trend in age
"""

age_fig = plt.figure()
ax = init_fig.add_subplot(1,2,1)
for curve, label in zip(agg_by_age, options.age_titles):
    ax.plot(curve.index, curve, label = label)
ax.set_xlabel('months')
ax.set_ylabel('scaled f')
ax.set_title('aggregate curve by age')

for i, init_label in zip(xrange(options.num_init_bins), options.init_titles):
    ax = age_fig.add_subplot(3,2,i+1)
    ax.set_title(init_label)
    for j in xrange(options.num_age_bins):
        curve = pred_d[(i,j)]
        ax.plot(curve.index, curve)
        
ps.figure_to_pdf(age_fig, outfile_age)
