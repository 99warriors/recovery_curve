import importlib
import recovery_curve.getting_data as gd


real_data = importlib.import_module('recovery_curve.hard_coded_objects.real_data_no_flats').data

data = gd.asymptotic_data_from_data(2.0,3)(real_data)
