

data_home = './'
train_diffcovs_r_script = ''
xs_file = '../raw_data/xs.csv'
ys_folder = '../raw_data'

from management_stuff import *


"""
read_fs
"""


def read_posterior_traces(folder):
    B_a_trace = pandas.read_csv(folder+'out_B_a.csv', header=None)
    B_b_trace = pandas.read_csv(folder+'out_B_b.csv', header=None)
    B_c_trace = pandas.read_csv(folder+'out_B_c.csv', header=None)
    phi_a_trace = pandas.read_csv(folder+'out_phi_a.csv', header=None)
    phi_b_trace = pandas.read_csv(folder+'out_phi_b.csv', header=None)
    phi_c_trace = pandas.read_csv(folder+'out_phi_c.csv', header=None)
    phi_m_trace = pandas.read_csv(folder+'out_phi_m.csv', header=None)
    return keyed_dict({'B_a_trace':B_a_trace, 'B_b_trace':B_b_trace, 'B_c_trace':B_c_trace, 'phi_a_trace':phi_a_trace, 'phi_b_trace':phi_b_trace, 'phi_c_trace':phi_c_trace,'phi_m_trace':phi_m_trace})

def read_diffcovs_data(folder):
    import pandas as pd
    pids_file = '%s/%s' % (folder, 'pids.csv')
    xas_file = '%s/%s' % (folder, 'xas.csv')
    xbs_file = '%s/%s' % (folder, 'xbs.csv')
    xcs_file = '%s/%s' % (folder, 'xcs.csv')
    ss_file = '%s/%s' % (folder, 'ss.csv')
    pids = pd.read_csv(pids_file, header=False, squeeze=True)
    xasd = pd.read_csv(xbs_file, header=True, index_col=0)
    xbs = pd.read_csv(xbs_file, header=True, index_col=0)
    xcs = pd.read_csv(xcs_file, header=True, index_col=0)
    ss = pd.read_csv(ss_file, header=False, index_col=0, squeeze=True)
    ys_folder = '%s/%s' % (folder, 'datapoints')
    l = []
    for pid, xa, xb, xc, s in zip(pids, xas, xbs, xcs, ss):
        p_ys_file = '%s/%s' % (ys_folder, pid)
        p_ys = pd.read_csv(p_ys_file,header=False, index_col = 1)
        l.append(datum(pid, xa, xb, xc, s, p_ys))
    return data(l)

def read_dataframe(full_path):
    import pandas
    return keyed_DataFrame(pandas.from_csv(full_path, index_col=0, header=0))

"""
print_fs
"""
def write_diffcovs_data(d, folder):
    """
    files: xas, xbs, xcs, s, folder with 1 file for every series
    """
    import pandas as pd
    pids = [p.pid for p in d]
    pids_file = '%s/%s' % (folder, 'pids.csv')
    pids.to_csv(pids_file, header=False, index=False)
    xas = pd.DataFrame([p.xa for p in d], index = pids)
    xbs = pd.DataFrame([p.xb for p in d], index = pids)
    xcs = pd.DataFrame([p.xc for p in d], index = pids)
    ss = pd.Series([p.s for p in d], index = pids)
    xas_file = '%s/%s' % (folder, 'xas.csv')
    xas.to_csv(xas_file, header=True, index=True)
    xbs_file = '%s/%s' % (folder, 'xbs.csv')
    xbs.to_csv(xbs_file, header=True, index=True)
    xcs_file = '%s/%s' % (folder, 'xcs.csv')
    xcs.to_csv(xcs_file, header=True, index=True)
    ss_file = '%s/%s' % (folder, 'ss.csv')
    ss.to_csv(ss_file, header=False, index=True)
    ys_folder = '%s/%s' % (folder, 'datapoints')
    make_folder(ys_folder)
    for p in d:
        p_ys_file = '%s/%s' % (ys_folder, p.pid)
        p.ys.to_csv(p_ys_file, header=False, index=True)

def hypers_print_f(h):
    return '%.2f,%.2f,%.2f,%.2f,%.2f,%.2f,%.2f,' % (h.c_a,h.c_b,h.c_c,h.l_a,h.l_b,h.l_c,h.l_m)

def write_dataframe(df, full_path):
    df.to_csv(full_path, header=True, index=True)


"""
keyed_objects
"""

class keyed_dict(keyed_object):
    pass

class keyed_DataFrame(keyed_object):
    pass



"""
id_iterators
"""


"""
features
"""

class ucla_cov_f(keyed_object):

    age, race, gleason, stage, psa, comor = range(6)
    cov_names = {age:'age', race:'race', gleason:'gleason', stage:'stage', psa:'psa', comor:'comor'}

    def __init__(self, which_cov):
        import pandas
        self.which_cov = which_cov
        self.all_covs = pandas.read_csv(xs_file, index_col=0)

    def __call__(self, pid):
        return self.all_covs[pid][ucla_cov_f.cov_names]

"""
model_trainers
"""

class pops(keyed_object):

    def __init__(self, pop_a, pop_b, pop_c):
        self.pop_a, self.pop_b, self.pop_c = pop_a, pop_b, pop_c

    @staticmethod
    def print_f(x):
        return '%.4f, %.4f, %.f4' % (self.pop_a, self.pop_b, self.pop_c)

    @staticmethod
    def read_f(full_path):
        f = open(full_path,'r')
        raw = f.readline().strip().split(',')
        return pops(raw[0], raw[1], raw[2])

class train_better_pops_f(possibly_cached):
    
    def get_introspection_key(self):
        return 'train_better_pops'

    def key_f(self, data):
        return '%s_%s' % (self.get_key(), data.get_key())

    def location_f(self, data):
        return '%s/%s' % (data.get_location(), 'better_pops')

    print_handler_f = staticmethod(string_adapter(pops.print_f))

    read_f = staticmethod(pops.read_f)

    to_recalculate = False

    def __call__(self, data):
        """
        averages curves, and then fits a curve to it
        """
        def obj_f(x):
            a,b,c = x[0],x[1],x[2]
            error = 0.0
            for k,data in d.iteritems():                                                        
                for t,v in data.data_points.iteritems():
                    fitted_val = the_f(t,data.cov.s,a,b,c)
                    error = error + pow(fitted_val - v, 2)
            return error

        import scipy.optimize
        x, f, d = scipy.optimize.fmin_l_bfgs_b(obj_f, np.array([0.5, 0.5, 2.0]), approx_grad = True, bounds = [(0.01,.99),[0.01,.99],[0.01,None]])
        return x

class train_diffcovs_model(possibly_cached):
    """
    returns posterior traces
    """
    
    def get_introspection_key(self):
        return '%s_%s_%s_%d_%d_%d', ('diffcovs', self.get_pops_f.get_key(), self.hypers.get_key(), self.iters, self.chains, self.seed)
        
    def key_f(self, data):
        return '%s_%s' % (self.get_key(), data.get_key())

    def location_f(self, data):
        return '%s/%s/%' % (data.get_location(), 'trained_diffcovs', self.get_pops_f.get_key())

    print_handler_f = staticmethod(do_nothing_adapter)

    read_f = staticmethod(read_posterior_traces)

    to_recalculate = False

    def __init__(self, get_pops_f, hypers, iters, chains, seed):
        self.r_script = train_diffcovs_r_script
        self.get_pops_f, self.hypers, self.iters, self.chains, self.seed = get_pops_f, hypers, iters, chains, seed

    def __call__(self, data):
        pops = call_and_save(self.get_pops_f)(data)
        pops_path = pops.get_full_path()
        data_path = data.get_full_path()
        save_object()(self.hypers)
        hypers_path = self.hypers.get_full_path()
        save_path = self.full_path_f(data, hypers, iters, chains)
        make_folder(save_path)
        import subprocess
        cmd = '%s %s %s %s %d %d %d %s' % (self.r_script, pops_path, data_path, hypers_path, self.iters, self.chains, self.seed, save_path)
        subprocess.call(cmd, shell=True)
        return train_diffcovs_model.read_f(save_path)


"""
predictors
"""

class get_param_mean_f(possibly_cached):
    """
    returns dictionary of mean params
    """

    def get_introspection_key(self):
        return '%s_%s' % ('mean_post_params', self.train_model_f.get_key())

    def key_f(self, data):
        return '%s_%s' % (self.get_key(), data.get_key())

    def __init__(self, train_model_f):
        self.train_model_f = train_model_f

    def __call__(self, data):
        post_params = call_and_save(self.train_model_f(data))
        import numpy as np
        return {p:np.mean(p) for p in post_params}


class tester_f(possibly_cached):
    """
    given a datum and a time, gives point prediction for that time
    requires the output of training a model
    """
    def __init__(self, param_points, pops):
        self.param_points, self.pop, = param_points, pops

    def __call__(self, datum, time):
        a = g_a(pops.pop_a, datum.xa)
        b = g_b(pops.pop_b, datum.xb)
        c = g_c(pops.pop_c, datum.xc)
        return the_f(time, datum.s, a, b, c)


class performance_series_f(possibly_cached):
    """
    returns a dataframe.  mean score will be the first column
    """

    def __init__(self, times, loss_f, tester_f, percentiles):
        self.times, self.loss_f, self.tester_f, self.percentiles = times, loss_f, tester_f, percentiles

    def __call__(self, data):
        pass

    def get_introspection_key(self):
        return '%s_%s_%s' % ('perf', self.get_key(), self.loss_f.get_key(), self.tester_f.get_key())

    def key_f(self, data):
        return '%s_%s' % self.get_key(), data.get_key()

    def location_f(self, data):
        '%s/%s' % (data.get_location(), tester_f.__class__.__name__)

    print_handler_f = staticmethod(write_dataframe)

    read_f = staticmethod(read_dataframe)

    to_recalculate = False

"""
what 'data' means for this project
"""

class datum(keyed_object):

    def __init__(self, pid, xa, xb, xc, s, ys):
        self.pid, self.xa, self.xb, self.xc, self.s, self.ys = pid, xa, xb, xc, s, ys


class data(keyed_object):

    def __init__(self, l):
        self.l = l

    def __iter__(self):
        return self.l.__iter__()

    def __len__(self):
        return len(self.l)

class get_data_f(possibly_cached):

    def __init__(self, x_abc_fs, s_f, ys_f):
        self.x_abc_fs, self.s_f, self.ys_f = x_abc_fs, s_f, ys_f

    def get_introspection_key(self):
        return '%s_%s_%s_%s' % ('data', self.x_abc_fs.get_key(), self.s_f.get_key(), self.ys_f.get_key())

    def key_f(self, pid_iterator):
        return '%s_%s' % (self.get_key(), pid_iterator.get_key())

    def location_f(self, pid_iterator):
        return '%s/%s/%s' % (data_home, 'data', self.get_key())

    print_handler_f = staticmethod(write_diffcovs_data)

    read_f = staticmethod(read_diffcovs_data)

    to_recalculate = False

    def __call__(self, pid_iterator):

        l = []

        for pid in pid_iterator:
            try:
                xa = x_abc_fs.xa_fs(pid)
                xb = x_abc_fs.xb_fs(pid)
                xc = x_abc_fs.xc_fs(pid)
                s = sf(pid)
                ys = ys_f(pid)
            except:
                pass
            else:
                l.append(datum(pid, xa, xb, xc, s, ys))
        l = sorted(l, key = lambda x: x.pid)
        return data(l)
                
class filtered_get_data_f(keyed_object):

    def get_introspection_key(self):
        return '%s_%s_%s' % 'filtered_data', self.filtering_f.get_key(), self.get_data_f.get_key()

    def key_f(self):
        return '%s_%s' % (self.get_key(), id_iterator.get_key())

    def location_f(self, id_iterator):
        return '%s/%s/%s' % (data_home, 'data', self.get_key())

    print_handler_f = staticmethod(write_diffcovs_data)

    read_f = staticmethod(read_diffcovs_data)

    to_recalculate = False

    def __init__(self, filtering_f, get_data_f):
        self.filtering_f, self.get_data_f = filtering_f, get_data_f

    def __call__(self, id_iterator):
        l = self.get_data_f(id_iterator)
        filtered_l = []
        for d in l:
            if filtering_f(d):
                filtered_l.append(d)
        return filtered_l
            
class get_data_fold_training(possibly_cached):

    def get_introspection_key(self):
        '%s_%d_%d_%s' % 'train_fold', self.fold_i, self.fold_k, self.get_data_f.get_key()

    def key_f(self, pid_iterator):
        '%s_%s' % (self.get_key(), pid_iterator.get_key())

    def location_f(self, pid_iterator):
        return '%s/%s_%d_%d' % (self.get_data_f.location_f(pid_iterator), 'training', self.fold_i, self.fold_k)

    print_handler = staticmethod(write_diffcovs_data)

    read_f = staticmethod(read_diffcovs_data)

    to_recalculate = False

    def __init__(self, get_data_f, fold_i, fold_k):
        self.get_data_f, self.fold_i, self.fold_k = get_data_f, fold_i, fold_k

    def __call__(self, pid_iterator):
        data = self.get_data_f(pid_iterator)
        return data([datum for datum,i in zip(data, range(len(data))) if i%fold_k != fold_i])


class get_data_fold_testing(possibly_cached):

    def get_introspection_key(self):
        '%s_%d_%d_%s' % 'testing_fold', self.fold_i, self.fold_k, self.get_data_f.get_key()

    def key_f(self, pid_iterator):
        '%s_%s' % (self.get_key(), pid_iterator.get_key())

    def location_f(self, pid_iterator):
        return '%s/%s_%d_%d' % (self.get_data_f.location_f(pid_iterator), 'testing', self.fold_i, self.fold_k)

    print_handler = staticmethod(write_diffcovs_data)

    read_f = staticmethod(read_diffcovs_data)

    to_recalculate = False

    def __init__(self, get_data_f, fold_i, fold_k):
        self.get_data_f, self.fold_i, self.fold_k = get_data_f, fold_i, fold_k

    def __call__(self, pid_iterator):
        data = self.get_data_f(pid_iterator)
        return data([datum for datum,i in zip(data, range(len(data))) if i%fold_k == fold_i])
            

"""
related to x_abc_fs
"""

class x_abc_fs(keyed_object):

    def __init__(self, xa_fs, xb_fs, xc_fs):
        self.xa_fs, self.xb_fs, self.xc_fs = xa_fs, xb_fs, xc_fs




"""
related to ys_f, as in getting the series
"""

class score_modifier_f(keyed_object):

    def get_instrospection_key(self):
        return '%s_%.2f' % ('upscaled', c)

    def __init__(self, c):
        self.c = c

    def __call__(self, s):
        return (s+c) / (1+c)

class ys_f(keyed_object):

    physical_condition, mental_condition, urinary_function, urinary_bother, bowel_function, bowel_bother, sexual_function, sexual_bother = 2,3,4,5,6,7,8,9

    function_names = {physical_condition:'physical_condition', mental_condition:'mental_condition', urinary_function:'urinary_function', urinary_bother:'urinary_bother', bowel_function:'bowel_function', bowel_bother:'bowel_bother', sexual_function:'sexual_function', sexual_bother:'sexual_bother'}

    def __init__(self, which_function):
        import pandas
        function_name = ys_f.function_names[which_function]
        raw_file = '%s/%s.csv %' (ys_folder, function_name)
        self.which_function = which_function
        self.dump = pandas.read_csv(raw_file, index_col=0)

    def __call__(self, pid):
        return self.dump[pid]


"""
related to hypers
"""
class hypers(keyed_object):
    """
    key is hard coded
    """
    def __init__(self, c_a, c_b, c_c, l_a, l_b, l_c, l_m, key):
        self.c_a, self.c_b, self.c_c, self.l_a, self.l_b, self.l_c, self.l_m = c_a, c_b, c_c, l_a, l_b, l_c, l_m
        self.key = key
        
    def get_location(self):
        return '%s/%s' % (data_home, 'hypers')

    print_handler_f = staticmethod(string_adapter(hypers_print_f))

    to_recalculate = False

"""
helpers
"""

def the_f(t, s, a, b, c):
    return s * ( (1.0-a) - (1.0-a)*(b) * math.exp(-1.0*t/c))

def g_a(pop_a, B_a, xa):
    pass
                 
def g_c(pop_b, B_b, xb):
    pass

def g_c(pop_c, B_c, xc):
    pass

def get_curve_abc(s, curve):
    import math
    import scipy.optimize
    def obj_f(x):
        error = 0.0
        for time, value in curve.iteritems():
            if not np.isnan(value):
                error += pow(_f(time, s, x[0], x[1], x[2]) - value, 2)
        return error
    x, f, d = scipy.optimize.fmin_l_bfgs_b(obj_f, np.array([0.5, 0.5, 2.0]), approx_grad = True, bounds = [(0.00,1.0),[0.00,1.0],[0.01,None]])
    return x

class hard_coded_f(object):

    def __init__(self, val):
        self.val = val

    def __call__(self, *args, **kwargs):
        return self.val

def make_folder(folder):
    if not os.path.isdir(folder):
        os.makedirs(folder)

class bin(object):

    def __init__(self, low, high):
        self.low, self.high = low, high

    def __contains__(self, obj):
        """
        returns true if obj is in [low, high)
        """
        if (not self.low ==  None) and obj < self.low:
            return False
        if (not self.high == None) and obj >= self.high:
            return False
        return True

    def __repr__(self):
        first = 'none' if self.low == None else '%.2f' % self.low
        second = 'none' if self.high == None else '%.2f' % self.high
        return first + '_' + second


class equals_bin(object):

    def __init__(self, in_vals):
        self.in_vals = in_vals

    def contains(self, obj):
        if obj in self.in_vals:
            return True
        else:
            return False

    def __repr__(self):
        import string
        return string.join([str(v) for v in self.in_vals], sep='_')


