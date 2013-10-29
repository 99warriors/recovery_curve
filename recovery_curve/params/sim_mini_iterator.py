import recovery_curve.prostate_specifics as ps
from recovery_curve.management_stuff import *
import recovery_curve.hard_coded_objects.hypers as hard_coded_hypers
import pdb
import pandas
import itertools

"""
iterator used for comparing different models with fixed feature set
10-1 - using to search for config where full_model beats logistic regression
"""

class the_iterable_cls(object):

    def __iter__(self):

        dim = 1 #

        #sim_get_data_f_constructors = [ps.simulated_get_data_truncate_0_f]
        sim_get_data_f_constructors = [ps.simulated_get_data_f]


        #get_posterior_f_constructors = [ps.get_pystan_diffcovs_truncated_posterior_f, ps.get_pystan_diffcovs_posterior_f]
        get_posterior_f_constructors = [ps.get_pystan_diffcovs_posterior_f]


        num_pids = [1000]
        diffcovs_iters = [10000]
        diffcovs_numchains = [10]
        seeds = [1]

        
        sim_times = [[1,2,4,8,12,18,24,30,36,42,48]]
        #get_pops_fs = [ps.train_shape_pops_f()]
        get_pops_fs = [ps.train_better_pops_f()]

        hypers = [hard_coded_hypers.default_hyper]

        B_a_gap, B_b_gap, B_c_gap = 0.5, 0.5, 0.5 #
        B_a_start, B_b_start, B_c_start = -2, 1, 2 #

        params = [set_hard_coded_key_dec(ps.keyed_dict, 'n_1000_ABC_n2p1p2_phis_05_m_05')({\
                    'B_a':pandas.Series(ps.get_seq(B_a_start, B_a_gap, dim)),\
                        'B_b':pandas.Series(ps.get_seq(B_b_start, B_b_gap, dim)),\
                        'B_c':pandas.Series(ps.get_seq(B_c_start, B_c_gap, dim)),\
                        'phi_a':0.05,\
                        'phi_b':0.05,\
                        'phi_c':0.05,\
                        'phi_m':0.05,\
                    })]

        pops = [set_hard_coded_key_dec(ps.pops, 'dfltpops')(0.3,0.6,5.0)]

        id_to_x_s_fs = [ps.an_id_to_x_s_f(dim)]
        
        return itertools.product(sim_get_data_f_constructors, get_posterior_f_constructors, num_pids, diffcovs_iters, diffcovs_numchains, seeds, sim_times, get_pops_fs, hypers, params, pops, id_to_x_s_fs)

the_iterable = the_iterable_cls()
