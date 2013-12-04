from recovery_curve.prostate_specifics import *
from recovery_curve.management_stuff import *
from recovery_curve.getting_data import *


default_simple_indicators = set_hard_coded_key_dec(keyed_list, 'default_simple_indicators')(get_categorical_fs(ucla_cov_f(ucla_cov_f.age), [bin(None,55), bin(55,65), bin(65,None)]) + get_categorical_fs(s_f(ys_f(ys_f.sexual_function)), [bin(0,0.41), bin(0.41,0.6), bin(0.6,0.8), bin(0.8,None)]) + [ones_f()])

default_age_categorical_f = set_hard_coded_key_dec(get_categorical_fs,'dfltage')(ucla_cov_f(ucla_cov_f.age), [bin(None,55), bin(55,65), bin(65,None)])
medium_age_categorical_f = set_hard_coded_key_dec(get_categorical_fs,'medage')(ucla_cov_f(ucla_cov_f.age), [bin(None,55), bin(55,60), bin(60,65), bin(65,None)])
high_age_categorical_f = set_hard_coded_key_dec(get_categorical_fs,'hiage')(ucla_cov_f(ucla_cov_f.age), [bin(None,50), bin(50,55), bin(55,60), bin(60,65), bin(65,70), bin(70,None)])

default_age_categorical_f_all_levels = set_hard_coded_key_dec(get_categorical_fs_all_levels,'dfltage')(ucla_cov_f(ucla_cov_f.age), [bin(None,55), bin(55,65), bin(65,None)])
medium_age_categorical_f_all_levels = set_hard_coded_key_dec(get_categorical_fs_all_levels,'medage')(ucla_cov_f(ucla_cov_f.age), [bin(None,55), bin(55,60), bin(60,65), bin(65,None)])
high_age_categorical_f_all_levels = set_hard_coded_key_dec(get_categorical_fs_all_levels,'hiage')(ucla_cov_f(ucla_cov_f.age), [bin(None,50), bin(50,55), bin(55,60), bin(60,65), bin(65,70), bin(70,None)])

default_initial_categorical_f = set_hard_coded_key_dec(get_categorical_fs,'dfltinit')(s_f(ys_f(ys_f.sexual_function)), [bin(0,0.41), bin(0.41,0.6), bin(0.6,0.8), bin(0.8,None)])
high_initial_categorical_f = set_hard_coded_key_dec(get_categorical_fs,'hiinit')(s_f(ys_f(ys_f.sexual_function)), [bin(0,0.41), bin(0.41,0.6), bin(0.6,0.7), bin(0.7,0.8), bin(0.8,0.9), bin(0.9,1.0)])
higher_initial_categorical_f = set_hard_coded_key_dec(get_categorical_fs,'hierinit')(s_f(ys_f(ys_f.sexual_function)), [bin(0,0.30), bin(0.30,0.5), bin(0.5,0.6), bin(0.6,0.7), bin(0.7,0.8), bin(0.8,0.9), bin(0.9,1.0)])
medium_initial_categorical_f = set_hard_coded_key_dec(get_categorical_fs,'medinit')(s_f(ys_f(ys_f.sexual_function)), [bin(0,0.30), bin(0.30,0.5), bin(0.5,0.7), bin(0.7,0.8), bin(0.8,0.9), bin(0.9,1.0)])
highminus_initial_categorical_f = set_hard_coded_key_dec(get_categorical_fs,'himnit')(s_f(ys_f(ys_f.sexual_function)), [bin(0,0.41), bin(0.41,0.6), bin(0.6,0.8), bin(0.8,0.9), bin(0.9,1.0)])

default_initial_categorical_f_all_levels = set_hard_coded_key_dec(get_categorical_fs_all_levels,'dfltinit')(s_f(ys_f(ys_f.sexual_function)), [bin(0,0.41), bin(0.41,0.6), bin(0.6,0.8), bin(0.8,None)])
high_initial_categorical_f_all_levels = set_hard_coded_key_dec(get_categorical_fs_all_levels,'hiinit')(s_f(ys_f(ys_f.sexual_function)), [bin(0,0.41), bin(0.41,0.6), bin(0.6,0.7), bin(0.7,0.8), bin(0.8,0.9), bin(0.9,1.0)])
higher_initial_categorical_f_all_levels = set_hard_coded_key_dec(get_categorical_fs_all_levels,'hierinit')(s_f(ys_f(ys_f.sexual_function)), [bin(0,0.30), bin(0.30,0.5), bin(0.5,0.6), bin(0.6,0.7), bin(0.7,0.8), bin(0.8,0.9), bin(0.9,1.0)])
medium_initial_categorical_f_all_levels = set_hard_coded_key_dec(get_categorical_fs_all_levels,'medinit')(s_f(ys_f(ys_f.sexual_function)), [bin(0,0.30), bin(0.30,0.5), bin(0.5,0.7), bin(0.7,0.8), bin(0.8,0.9), bin(0.9,1.0)])
highminus_initial_categorical_f_all_levels = set_hard_coded_key_dec(get_categorical_fs_all_levels,'himnit')(s_f(ys_f(ys_f.sexual_function)), [bin(0,0.41), bin(0.41,0.6), bin(0.6,0.8), bin(0.8,0.9), bin(0.9,1.0)])
