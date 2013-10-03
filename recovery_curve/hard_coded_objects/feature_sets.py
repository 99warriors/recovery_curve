from recovery_curve.prostate_specifics import *
from recovery_curve.management_stuff import *



default_simple_indicators = set_hard_coded_key_dec(keyed_list, 'default_simple_indicators')(get_categorical_fs(ucla_cov_f(ucla_cov_f.age), [bin(None,55), bin(55,65), bin(65,None)]) + get_categorical_fs(s_f(ys_f(ys_f.sexual_function)), [bin(0,0.41), bin(0.41,0.6), bin(0.6,0.8), bin(0.8,None)]) + [ones_f()])

default_age_categorical_f = set_hard_coded_key_dec(get_categorical_fs,'dfltage')(ucla_cov_f(ucla_cov_f.age), [bin(None,55), bin(55,65), bin(65,None)])
medium_age_categorical_f = set_hard_coded_key_dec(get_categorical_fs,'medtage')(ucla_cov_f(ucla_cov_f.age), [bin(None,55), bin(55,60), bin(60,65), bin(65,None)])

default_initial_categorical_f = set_hard_coded_key_dec(get_categorical_fs,'dfltinit')(s_f(ys_f(ys_f.sexual_function)), [bin(0,0.41), bin(0.41,0.6), bin(0.6,0.8), bin(0.8,None)])
high_initial_categorical_f = set_hard_coded_key_dec(get_categorical_fs,'hiinit')(s_f(ys_f(ys_f.sexual_function)), [bin(0,0.41), bin(0.41,0.6), bin(0.6,0.7), bin(0.7,0.8), bin(0.8,0.9), bin(0.9,1.0)])
