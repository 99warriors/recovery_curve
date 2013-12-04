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
import matplotlib.pyplot as plt

"""
get the 3 data lists
"""
all_data = importlib.import_module('recovery_curve.hard_coded_objects.real_data_all').data
filtered_data = importlib.import_module('recovery_curve.hard_coded_objects.go_two_real_data').data
filtered_data_no_flats = importlib.import_module('recovery_curve.hard_coded_objects.real_data_no_flats').data

def subtract_data(d1, d2):
    """
    returns d1-d2
    """
    return ps.data(list(set(d1)-set(d2)))

filtered_flat_data = subtract_data(filtered_data, filtered_data_no_flats)
all_filtered = subtract_data(all_data, filtered_data_no_flats)

out_file = '%s/%s' % (global_stuff.for_dropbox, 'aggregate_omitted_data')

"""
for each filtered dataset, for each feature, plot aggregate curves, stratified by feature
"""

filtered_flat_data_by_initial = [[] for x in xrange(4)]
filtered_flat_data_by_age = [[] for x in xrange(3)]

all_filtered_data_by_initial = [[] for x in xrange(4)]
all_filtered_data_by_age = [[] for x in xrange(3)]

filtered_no_flat_data_by_initial = [[] for x in xrange(4)]
filtered_no_flat_data_by_age = [[] for x in xrange(3)]

legend_size = 4
legend_loc = 4

def get_init_bin(_datum):
    if _datum.xa.iloc[2] > 0:
        return 0
    elif _datum.xa.iloc[1] > 0:
        return 1
    elif _datum.xa.iloc[3] > 0:
        return 2
    else:
        return 3

def get_age_bin(_datum):
    if _datum.xa.iloc[4] > 0:
        return 0
    elif _datum.xa.iloc[5] > 0:
        return 1
    else:
        return 2



for _datum in filtered_flat_data:
    filtered_flat_data_by_initial[get_init_bin(_datum)].append(_datum)
    filtered_flat_data_by_age[get_age_bin(_datum)].append(_datum)

for _datum in filtered_data_no_flats:
    filtered_no_flat_data_by_initial[get_init_bin(_datum)].append(_datum)
    filtered_no_flat_data_by_age[get_age_bin(_datum)].append(_datum)

for _datum in all_filtered:
    all_filtered_data_by_initial[get_init_bin(_datum)].append(_datum)
    all_filtered_data_by_age[get_age_bin(_datum)].append(_datum)

initial_names = ['init_0_41', 'init_41_60', 'init_60_80', 'init_80_100']
age_names = ['age_55-', 'age_55_65', 'age_65+']

filtered_flat_data_by_initial = [ps.data(_data) for _data in filtered_flat_data_by_initial]
filtered_flat_data_by_age = [ps.data(_data) for _data in filtered_flat_data_by_age]

filtered_no_flat_data_by_initial = [ps.data(_data) for _data in filtered_no_flat_data_by_initial]
filtered_no_flat_data_by_age = [ps.data(_data) for _data in filtered_no_flat_data_by_age]

all_filtered_data_by_initial = [ps.data(_data) for _data in all_filtered_data_by_initial]
all_filtered_data_by_age = [ps.data(_data) for _data in all_filtered_data_by_age]

figs = []

fig = plt.figure()
fig.suptitle('flat data')
fig.subplots_adjust(hspace=0.6, wspace=0.3)
ax = fig.add_subplot(2,1,1)
ax.set_title('stratified by init')
for _data, name in zip(filtered_flat_data_by_initial, initial_names):
    agg = ps.aggregate_shape_f()(_data)
    ax.plot(agg.index, agg, label=name)
ax.legend(prop={'size':legend_size},loc=legend_loc)

ax = fig.add_subplot(2,1,2)
ax.set_title('stratified by age')
for _data, name in zip(filtered_flat_data_by_age, age_names):
    agg = ps.aggregate_shape_f()(_data)
    ax.plot(agg.index, agg, label=name)
ax.legend(prop={'size':legend_size},loc=legend_loc)

figs.append(fig)

fig = plt.figure()
fig.subplots_adjust(hspace=0.6, wspace=0.3)
fig.suptitle('all filtered data')
ax = fig.add_subplot(2,1,1)
ax.set_title('stratified by init')
for _data, name in zip(all_filtered_data_by_initial, initial_names):
    agg = ps.aggregate_shape_f()(_data)
    ax.plot(agg.index, agg, label=name)
ax.legend(prop={'size':legend_size},loc=legend_loc)

ax = fig.add_subplot(2,1,2)
ax.set_title('stratified by age')
for _data, name in zip(all_filtered_data_by_age, age_names):
    agg = ps.aggregate_shape_f()(_data)
    ax.plot(agg.index, agg, label=name)
ax.legend(prop={'size':legend_size},loc=legend_loc)

figs.append(fig)

fig = plt.figure()
fig.subplots_adjust(hspace=0.6, wspace=0.3)
fig.suptitle('data actually used')
ax = fig.add_subplot(2,1,1)
ax.set_title('stratified by init')
for _data, name in zip(filtered_no_flat_data_by_initial, initial_names):
    agg = ps.aggregate_shape_f()(_data)
    ax.plot(agg.index, agg, label=name)
ax.legend(prop={'size':legend_size},loc=legend_loc)

ax = fig.add_subplot(2,1,2)
ax.set_title('stratified by age')
for _data, name in zip(filtered_no_flat_data_by_age, age_names):
    agg = ps.aggregate_shape_f()(_data)
    ax.plot(agg.index, agg, label=name)
ax.legend(prop={'size':legend_size},loc=legend_loc)

figs.append(fig)

ps.multiple_figures_to_pdf(figs, out_file)
