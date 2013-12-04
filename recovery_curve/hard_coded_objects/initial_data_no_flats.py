import importlib
import recovery_curve.getting_data as gd


real_data = importlib.import_module('recovery_curve.hard_coded_objects.real_data_no_flats').data

data = gd.initial_value_data(4,.01)(real_data)
