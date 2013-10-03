import recovery_curve.prostate_specifics as ps
from recovery_curve.management_stuff import *

default_hyper = set_hard_coded_key_dec(ps.hypers, 'dflthyper')(1.0,1.0,1.0,15.0,15.0,15.0,10.0)  
medium_hyper = set_hard_coded_key_dec(ps.hypers, 'medhyper')(0.5,0.5,0.5,15.0,15.0,15.0,10.0)  
relaxed_hyper = set_hard_coded_key_dec(ps.hypers, 'relhyper')(1.0,1.0,1.0,5.0,5.0,5.0,10.0)  
