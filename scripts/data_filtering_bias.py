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
get the 2 data lists
"""
filtered_data = importlib.import_module('recovery_curve.hard_coded_objects.go_two_real_data').data
filtered_data_no_flats = importlib.import_module('recovery_curve.hard_coded_objects.real_data_no_flats').data

"""
need to get data into lists of sets
"""

filtered_data_by_initial = [set() for x in xrange(4)]
filtered_data_by_age = [set() for x in xrange(3)]

no_flats_data_by_initial = [set() for x in xrange(4)]
no_flats_data_by_age = [set() for x in xrange(3)]

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

for _datum in filtered_data:
    filtered_data_by_initial[get_init_bin(_datum)].add(_datum)
    filtered_data_by_age[get_age_bin(_datum)].add(_datum)

for _datum in filtered_data_no_flats:
    no_flats_data_by_initial[get_init_bin(_datum)].add(_datum)
    no_flats_data_by_age[get_age_bin(_datum)].add(_datum)

"""
for each bin, print out counts in each dataset
"""

initial_names = ['init_0_41', 'init_41_60', 'init_60_80', 'init_80_100']
age_names = ['age_55-', 'age_55_65', 'age_65+']

for filtered_bin, no_flats_bin, name in zip(filtered_data_by_initial, no_flats_data_by_initial, initial_names):
    print name, len(filtered_bin), len(no_flats_bin), float(len(no_flats_bin))/len(filtered_bin)

for filtered_bin, no_flats_bin, name in zip(filtered_data_by_age, no_flats_data_by_age, age_names):
    print name, len(filtered_bin), len(no_flats_bin), float(len(no_flats_bin))/len(filtered_bin)
