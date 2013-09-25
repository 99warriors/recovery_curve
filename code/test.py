from prostate_specifics import *
from management_stuff import *

import pdb

pids = all_ucla_pid_iterator()

ucla_treatment_f()

surgery_pids = filtered_pid_iterator(pids, bin_f(ucla_treatment_f(),equals_bin([ucla_treatment_f.surgery])))

xa_fs = keyed_list([ucla_cov_f(ucla_cov_f.age), bin_f(ucla_cov_f(ucla_cov_f.psa), bin(0,20)), s_f(ys_f(ys_f.sexual_function))])
xb_fs = xa_fs
xc_fs = xa_fs

init = s_f(ys_f(ys_f.sexual_function))

a_ys = modified_ys_f(ys_f(ys_f.sexual_function), score_modifier_f(0))

gg=set_hard_coded_key_dec(x_abc_fs, 'feat')(xa_fs, xb_fs, xc_fs)


#d = get_dataframe_f(xa_fs)(pids)

#pdb.set_trace()

#pdb.set_trace()

#d = get_dataframe_f(xa_fs)(pids)

#pdb.set_trace()

data = get_data_f(gg, init, a_ys)(surgery_pids)

pdb.set_trace()

filtered_data = filtered_get_data_f()(data)

pdb.set_trace()

pops = train_better_pops_f()(filtered_data)

pdb.set_trace()



pdb.set_trace()

pids = [x for x in all_ucla_pid_iterator()]

print ys_f(ys_f.sexual_function)(pids[0])

class f(object):

    def __repr__(self):
        return 'z'

    def __hash__(self):
        import pdb
        #print 'z'
        #pdb.set_trace()
        return 1

    def __cmp__(self, other):
        print 'ggg'
        return self.__hash__() == other.__hash__()

class g(object):

    def __repr__(self):
        return 'e'

    def __hash__(self):
        import pdb
        #print 'e'
        #pdb.set_trace()
        return 2

a = f()
aa = f()
b = g()
import pandas
d=pandas.DataFrame({a:[3,4,5], b:[7,8,9]})
import pdb
print d[aa]
print d[b]
pdb.set_trace()
