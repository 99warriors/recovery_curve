from management_stuff import *
import prostate_specifics as ps
import getting_data as gd

dim = 1
pops = [set_hard_coded_key_dec(ps.pops, 'dfltpops')(0.3,0.6,5.0)]
id_to_x_s_fs = [ps.an_id_to_x_s_f(dim)]

N = 200

seed = 0

B_a_gap, B_b_gap, B_c_gap = 0.5, 0.5, 0.5 #
B_a_start, B_b_start, B_c_start = -2, 1, 2 #

params = [set_hard_coded_key_dec(ps.keyed_dict, 'ABC_n2p1p2_phis_05')({\
            'B_a':pandas.Series(ps.get_seq(B_a_start, B_a_gap, dim)),\
                'B_b':pandas.Series(ps.get_seq(B_b_start, B_b_gap, dim)),\
                'B_c':pandas.Series(ps.get_seq(B_c_start, B_c_gap, dim)),\
                'phi_a':0.05,\
                'phi_b':0.05,\
                'phi_c':0.05,\
                })]

phi_m = 0.05

noise_f = gd.beta_noise(phi_m)

sim_times = [1,2,4,8,12,18,24,30,36,42,48]

pid_iterator = gd.fake_pid_iterator(N)
data = sim_get_data_f_constructor(params, pops, id_to_x_s_f, sim_times, noise_f, seed)(pid_iterator)
