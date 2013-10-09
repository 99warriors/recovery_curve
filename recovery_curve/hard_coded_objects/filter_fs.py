from recovery_curve.prostate_specifics import *
from recovery_curve.management_stuff import *

"""
define and hardcode key of several f's that filter based on s and ys only, which can be used in filtered_get_data
"""

old_filter_f = set_hard_coded_key_dec(ys_bool_input_curve_f, 'oldfilt')(0, 0.08, 0.05, 2, 0.0)
med_filter_f = set_hard_coded_key_dec(ys_bool_input_curve_f, 'medfilt')(6, 0.072, 0.05, 2, 0.1)
high_filter_f = set_hard_coded_key_dec(ys_bool_input_curve_f, 'highfilt')(8, 0.065, 0.05, 2, 0.1)
