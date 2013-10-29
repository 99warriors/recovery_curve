from management_stuff import *
import pandas
import numpy as np
import functools
import math
import matplotlib.pyplot as plt
import matplotlib
from matplotlib.backends.backend_pdf import PdfPages
import itertools
import global_stuff
import multiprocessing
import random

"""

"""
"""
read_fs
"""


def read_unheadered_posterior_traces(folder):
    B_a_trace = pandas.read_csv(folder+'/out_B_a.csv', header=None)
    B_b_trace = pandas.read_csv(folder+'/out_B_b.csv', header=None)
    B_c_trace = pandas.read_csv(folder+'/out_B_c.csv', header=None)
    phi_a_trace = pandas.read_csv(folder+'/out_phi_a.csv', header=None)
    phi_b_trace = pandas.read_csv(folder+'/out_phi_b.csv', header=None)
    phi_c_trace = pandas.read_csv(folder+'/out_phi_c.csv', header=None)
    phi_m_trace = pandas.read_csv(folder+'/out_phi_m.csv', header=None)
    return keyed_dict({'B_a':B_a_trace, 'B_b':B_b_trace, 'B_c':B_c_trace, 'phi_a':phi_a_trace, 'phi_b':phi_b_trace, 'phi_c':phi_c_trace,'phi_m':phi_m_trace})

def read_posterior_traces(folder):
    B_a_trace = pandas.read_csv(folder+'/out_B_a.csv', header=0)
    B_b_trace = pandas.read_csv(folder+'/out_B_b.csv', header=0)
    B_c_trace = pandas.read_csv(folder+'/out_B_c.csv', header=0)
    phi_a_trace = pandas.read_csv(folder+'/out_phi_a.csv', header=0)
    phi_b_trace = pandas.read_csv(folder+'/out_phi_b.csv', header=0)
    phi_c_trace = pandas.read_csv(folder+'/out_phi_c.csv', header=0)
    phi_m_trace = pandas.read_csv(folder+'/out_phi_m.csv', header=0)
    return keyed_dict({'B_a':B_a_trace, 'B_b':B_b_trace, 'B_c':B_c_trace, 'phi_a':phi_a_trace, 'phi_b':phi_b_trace, 'phi_c':phi_c_trace,'phi_m':phi_m_trace})

def read_diffcovs_data(folder):
    import pandas as pd
    pids_file = '%s/%s' % (folder, 'pids.csv')
    xas_file = '%s/%s' % (folder, 'xas.csv')
    xbs_file = '%s/%s' % (folder, 'xbs.csv')
    xcs_file = '%s/%s' % (folder, 'xcs.csv')
    ss_file = '%s/%s' % (folder, 'ss.csv')
    pids = pd.read_csv(pids_file, header=None, squeeze=True, converters={0:str}, index_col=None)
    xas = pd.read_csv(xbs_file, header=0, index_col=0)
    xbs = pd.read_csv(xbs_file, header=0, index_col=0)
    xcs = pd.read_csv(xcs_file, header=0, index_col=0)
    ss = pd.read_csv(ss_file, header=None, index_col=0, squeeze=True)
    ys_folder = '%s/%s' % (folder, 'datapoints')
    l = []
    for pid, xa, xb, xc, s in zip(pids, xas.iteritems(), xbs.iteritems(), xcs.iteritems(), ss):
        p_ys_file = '%s/%s' % (ys_folder, pid)
        p_ys = pd.read_csv(p_ys_file,header=None, index_col = 0, squeeze=True).dropna()
        l.append(datum(pid, xa[1], xb[1], xc[1], s, p_ys))
    return data(l)

def hypers_read_f(full_path):
    f = open(full_path, 'r')
    a = [x for x in f.readline().strip().split(',')]
    f = open(full_path, 'r')
    c_a,c_b,c_c,l_a,l_b,l_c,l_m = [float(x) for x in f.readline().strip().split(',')]
    return hypers(c_a,c_b,c_c,l_a,l_b,l_c,l_m)

def read_DataFrame(full_path):
    return keyed_DataFrame(pandas.read_csv(full_path, index_col=0, header=0))

def read_Series(full_path):
    return keyed_Series(pandas.read_csv(full_path, index_col=0, header=0, squeeze=True))

"""
print_fs
"""
def write_diffcovs_data(d, folder):
    """
    files: xas, xbs, xcs, s, folder with 1 file for every series
    """
    import pandas as pd
    pids = pandas.Series([p.pid for p in d])
    pids_file = '%s/%s' % (folder, 'pids.csv')
    pids.to_csv(pids_file, header=False, index=False)
    xas = pd.DataFrame({p.pid:p.xa for p in d})
    xbs = pd.DataFrame({p.pid:p.xa for p in d})
    xcs = pd.DataFrame({p.pid:p.xa for p in d})
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

def write_posterior_traces(traces, folder):
    traces['B_a'].to_csv(folder+'/out_B_a.csv', header=True, index=False)
    traces['B_b'].to_csv(folder+'/out_B_b.csv', header=True, index=False)
    traces['B_c'].to_csv(folder+'/out_B_c.csv', header=True, index=False)
    traces['phi_a'].to_csv(folder+'/out_phi_a.csv', header=True, index=False)
    traces['phi_b'].to_csv(folder+'/out_phi_b.csv', header=True, index=False)
    traces['phi_c'].to_csv(folder+'/out_phi_c.csv', header=True, index=False)
    traces['phi_m'].to_csv(folder+'/out_phi_m.csv', header=True, index=False)

def hypers_print_f(h):
    return '%.2f,%.2f,%.2f,%.2f,%.2f,%.2f,%.2f' % (h.c_a,h.c_b,h.c_c,h.l_a,h.l_b,h.l_c,h.l_m)

def write_DataFrame(df, full_path):
    df.to_csv(full_path, header=True, index=True)

def write_Series(s, full_path):
    s.to_csv(full_path, header=True, index=True)

def figure_to_pdf(fig, full_path):
    fig.savefig('%s.pdf' % full_path, format='PDF')

def multiple_figures_to_pdf(fig_list, full_path):
    pp = PdfPages('%s.pdf' % full_path)
    for fig in fig_list:
        pp.savefig(fig)
    pp.close()

"""
keyed_objects
"""

class keyed_dict(dict, keyed_object):
    pass

class keyed_DataFrame(pandas.DataFrame, keyed_object):
    pass

class keyed_list(list, keyed_object):
   
    def get_introspection_key(self):
        return '_'.join([x.get_key() for x in self])

class keyed_Series(pandas.Series, keyed_object):
    pass


class keyed_iterable(keyed_object):

    def __init__(self, iterable):
        self.iterable = iterable

    def __iter__(self):
        return self.iterable.__iter__()

"""
id_iterators
"""

class all_ucla_pid_iterator(keyed_object):

    def get_introspection_key(self):
        return 'all_ucla'

    def __init__(self):
        self.pids = pandas.read_csv(global_stuff.all_pid_file, header=False, index_col=None, squeeze=True, converters={0:str}).tolist()

    def __iter__(self):
        return iter(self.pids)

class filtered_pid_iterator(keyed_object):

    def get_introspection_key(self):
        return '%s_%s_%s' % ('ftit', self.backing_iterator.get_key(), self.bool_f.get_key())

    def __init__(self, backing_iterator, bool_f):
        self.backing_iterator, self.bool_f = backing_iterator, bool_f

    def __iter__(self):
        for pid in self.backing_iterator:
            try:
                if not self.bool_f(pid):
                    raise my_Exception
            except my_Exception:
                pass
            else:
                yield pid


class is_good_pid(keyed_object):
    
    def get_introspection_key(self):
        return 'isgood'

    def __init__(self):
        self.pids = pandas.read_csv(global_stuff.good_file, header=None,index_col=None, squeeze=True, converters={0:str}).tolist()

    def __call__(self, pid):
        return pid in self.pids
        
class is_medium_pid(keyed_object):
    
    def __init__(self):
        self.pids = pandas.read_csv(global_stuff.medium_file, header=None,index_col=None, squeeze=True,converters={0:str}).tolist()

    def __call__(self, pid):
        return pid in self.pids


class ys_bool_input_curve_f(keyed_object):
    """
    given curve = (s, ys), returns T/F
    """
    def get_introspection_key(self):
        return '%s_%d_%.2f_%.2f_%d_%.2f' % ('sysboolf', self.min_data_points, self.max_avg_error, self.above_s_tol, self.above_s_max, self.min_s)

    def __init__(self, min_data_points, max_avg_error, above_s_tol, above_s_max, min_s):
        self.min_data_points, self.max_avg_error, self.above_s_tol, self.above_s_max, self.min_s = min_data_points, max_avg_error, above_s_tol, above_s_max, min_s

    def __call__(self, s, ys):
        if sum([x > (s + self.above_s_tol) for x in ys]) >= self.above_s_max:
            return False
        a,b,c = get_curve_abc(s, ys)
        fit_f = functools.partial(the_f, s=s, a=a, b=b, c=c)
        fitted = pandas.Series(ys.index,index=ys.index).apply(fit_f)
        error = (fitted - ys).abs().sum()
        if error / len(ys) > self.max_avg_error:
            return False
        if len(ys) < self.min_data_points:
            return False
        if s < self.min_s:
            return False
        return True






class ys_bool_input_pid_f(keyed_object):
    """
    given parameters like tolerable error, number of points greater than s
    gives function that returns bool given pid
    filter_f operates on s, ys
    """
    def get_introspection_key(self):
        return '%s_%s_%s' % (self.s_f.get_key(), self.actual_ys_f.get_key(), self.filter_f.get_key())

    def __init__(self, s_f, actual_ys_f, filter_f):
        self.s_f, self.actual_ys_f, self.filter_f = s_f, actual_ys_f, filter_f

    def __call__(self, pid):
        s = self.s_f(pid)
        ys = self.actual_ys_f(pid)
        return self.filter_f(s, ys)
        


"""
features
"""

class feat(keyed_object):

    def to_normalize(self):
        return True

    def __repr__(self):
        return self.get_key()

    
    def __eq__(self, other):
        try:
            return self.get_key() == other.get_key()
        except:
            return self.get_key() == str(other)
    
    def __hash__(self):
        return hash(self.get_key())
    
"""
    def __cmp__(self, other):
        print 'cmp1', self.get_key()
        try:
            other_key = other.get_key()
        except AttributeError:
            pdb.set_trace()
            other_key = str(other)
        print 'cmp2', self.get_key(), other_key
        return self.get_key() == other_key
"""
class ones_f(feat):

    def to_normalize(self):
        return False

    def get_introspection_key(self):
        return 'ones_f'

    def __call__(self, pid):
        return 1

    def get_introspection_key(self):
        return 'ones_f'

class ucla_treatment_f(feat):

    surgery, radiation, brachy = 1,2,3

    treatment_names = {surgery:'surg', radiation:'rad', brachy:'brachy'}

    def get_introspection_key(self):
        return 'treat_f'
        
    @raise_if_na
    def __call__(self, pid):
        first = pid[0]
        if first == '1':
            return ucla_treatment_f.brachy
        elif first == '2':
            return ucla_treatment_f.radiation
        elif first == '3':
            return ucla_treatment_f.surgery
        assert False

class ucla_cov_f(feat):

    age, race, gleason, stage, psa, comor = range(6)
    cov_names = {age:'age', race:'race', gleason:'gleason', stage:'stage', psa:'psa', comor:'comor'}

    def get_introspection_key(self):
        return ucla_cov_f.cov_names[self.which_cov]

    def __init__(self, which_cov):
        import pandas
        self.which_cov = which_cov
        self.all_covs = pandas.read_csv(global_stuff.xs_file, index_col=0)

    @raise_if_na
    def __call__(self, pid):
        return self.all_covs[pid][ucla_cov_f.cov_names[self.which_cov]]

class bin_f(feat):

    def get_introspection_key(self):
        return '%s_%s_%s' % ('bin_f', self.backing_f, self.bin)

    def __init__(self, backing_f, bin):
        self.backing_f, self.bin = backing_f, bin

    @raise_if_na
    def __call__(self, pid):
        raw = self.backing_f(pid)
        return int(raw in self.bin)



"""
predictor factories(Aka trainers) and predictor class definitions
"""

class point_predictor(object):

    def plot_prediction(self, ax, prediction, color=None, name=None):
        if color == None:
            color=self.display_color
        if name == None:
            name=self.display_name
        add_series_to_ax(prediction, ax, color, name, linestyle='-')

    def get_prediction_series(self, datum, ts):
        d = {}
        for t in ts:
            d[t] = self.__call__(datum, t)
        return pandas.Series(d)

class nonpoint_predictor_base(object):
    """
    __call__ will return the entire unsorted distribution of f(t)
    """
    def get_prediction_series(self, datum, ts):
        d = {}
        for t in ts:
            d[t] = self.__call__(datum, t)
        return pandas.DataFrame(d)

    def plot_prediction(self, ax, prediction, percentiles=[.25,.50,.75], color=None, name=None):
        if color == None:
            color=self.display_color
        if name == None:
            name=self.display_name
        d = {}
        for t, samples in prediction.iteritems():
            cp = samples.copy()
            cp.sort()
            s_d = {}
            s_d['mean'] = cp.mean()
            num = len(cp)
            for percentile in percentiles:
                s_d[percentile] = cp[percentile * num]
            d[t] = pandas.Series(s_d)
        d = pandas.DataFrame(d)
        for l, s in d.iterrows():
            if l == 'mean':
                ax.plot(s.index, s, linestyle='--', marker='x', label=name, color=color)
            else:
                ax.plot(s.index, s, linestyle='None', marker='_', color=color)

class pops(keyed_object):

    def __repr__(self):
        return '%.2f, %.2f, %.2f' % (self.pop_a, self.pop_b, self.pop_c)

    def __init__(self, pop_a, pop_b, pop_c):
        self.pop_a, self.pop_b, self.pop_c = pop_a, pop_b, pop_c

    @staticmethod
    def print_f(x):
        return '%.4f, %.4f, %.4f' % (x.pop_a, x.pop_b, x.pop_c)

    @staticmethod
    def read_f(full_path):
        f = open(full_path,'r')
        raw = f.readline().strip().split(',')
        return pops(float(raw[0]), float(raw[1]), float(raw[2]))


class train_shape_pops_f(possibly_cached):
    """
    used as init input for full and prior model
    """
    
    def get_introspection_key(self):
        return 'shapepops'

    def key_f(self, data):
        return '%s_%s' % (self.get_key(), data.get_key())

    def location_f(self, data):
        return '%s/%s' % (data.get_location(), 'shape_pops')

    print_handler_f = staticmethod(string_adapter(pops.print_f))

    read_f = staticmethod(pops.read_f)

    @key
    @save_and_memoize
    #@read_from_file
    def __call__(self, data):
        avg_shape = aggregate_shape_f()(data)
        a,b,c = get_curve_abc(1.0, avg_shape)
        return pops(a,b,c)

class train_better_pops_f(possibly_cached):
    """
    used as init input for full and prior model
    """
    
    def get_introspection_key(self):
        return 'betpops'

    def key_f(self, data):
        return '%s_%s' % (self.get_key(), data.get_key())

    def location_f(self, data):
        return '%s/%s' % (data.get_location(), 'better_pops')

    print_handler_f = staticmethod(string_adapter(pops.print_f))

    read_f = staticmethod(pops.read_f)

    @key
    @save_and_memoize
    #@read_from_file
    def __call__(self, data):
        """
        averages curves, and then fits a curve to it
        """
        
        def obj_f(x):
            a,b,c = x[0],x[1],x[2]
            error = 0.0
            for datum in data:
                fit_f = functools.partial(the_f, s=datum.s, a=a, b=b, c=c)
                ys = datum.ys
                fitted = pandas.Series(ys.index,index=ys.index).apply(fit_f)
                diff_vect = (fitted - ys)
                this = diff_vect.dot(diff_vect)
                error += this
                
                """
                for t,v in datum.ys.iteritems():
                    fitted_val = the_f(t,datum.s,a,b,c)
                    if not np.isnan(v):
                        error = error + pow(fitted_val - v, 2)
                """
            #print error
            return error

        import scipy.optimize
        x, f, d = scipy.optimize.fmin_l_bfgs_b(obj_f, np.array([0.5, 0.5, 2.0]), approx_grad = True, bounds = [(0.01,.99),[0.01,.99],[0.01,None]])
        return pops(x[0],x[1],x[2])


class prior_predictor(keyed_object, point_predictor):

    display_color = 'blue'

    display_name = 'prior'

    def get_introspection_key(self):
        return '%s_%s' % 'priorpred', self.pops.get_key()

    def __init__(self, pops):
        self.pops = pops

    def __call__(self, datum, time):
        return the_f(time, datum.s, self.pops.pop_a, self.pops.pop_b, self.pops.pop_c)


class get_prior_predictor_f(possibly_cached):

    display_color = prior_predictor.display_color

    display_name = prior_predictor.display_name

    def get_introspection_key(self):
        return '%s_%s' % ('priorpred-f', self.get_pops_f.get_key())

    def key_f(self, data):
        return '%s_%s' % (self.get_key(), data.get_key())

    def __init__(self, get_pops_f):
        self.get_pops_f = get_pops_f

    @key
    def __call__(self, data):
        pops = self.get_pops_f(data)
        return prior_predictor(pops)



class get_diffcovs_posterior_f(possibly_cached_folder):
    """
    returns posteriors for diffcovs model
    """
    
    def get_introspection_key(self):
        return '%s_%s_%s_%d_%d_%d' % ('diffcovs', self.get_pops_f.get_key(), self.hypers.get_key(), self.iters, self.chains, self.seed)
        
    def key_f(self, data):
        return '%s_%s' % (self.get_key(), data.get_key())

    def location_f(self, data):
        return '%s/%s/%s' % (data.get_location(), 'trained_diffcovs', self.get_pops_f.get_key())

    print_handler_f = staticmethod(folder_adapter(write_posterior_traces))

    read_f = staticmethod(read_posterior_traces)

    def has_file_content(self, data):
        return os.path.exists('%s/%s' % (self.full_file_path_f(data), 'out_B_a.csv'))

    def __init__(self, get_pops_f, hypers, iters, chains, seed):
        self.r_script = global_stuff.train_diffcovs_r_script
        self.get_pops_f, self.hypers, self.iters, self.chains, self.seed = get_pops_f, hypers, iters, chains, seed

#    @save_and_memoize
    @key
    @read_from_pickle
    @save_to_file
    @save_to_pickle
    def __call__(self, data):
        pops = self.get_pops_f(data)
        pops_path = self.get_pops_f.full_file_path_f(data)
        #data.get_creator().save(data)
        data_path = data.get_full_path()
        hypers_save_f()(self.hypers)
        hypers_path = hypers_save_f().full_file_path_f(self.hypers)
        save_path = self.full_file_path_f(data)
        make_folder(save_path)
        train_helper_file = '%s/%s/%s' % (global_stuff.home, 'recovery_curve', 'train_helper.r')
        diffcovs_model_file = '%s/%s/%s' % (global_stuff.home, 'recovery_curve', 'full_model_diffcovs.stan')
        import subprocess, os
        cmd = '%s \"%s\" \"%s\" \"%s\" \"%s\" %d %d %d \"%s\" \"%s\" \"%s\" %d' % ('Rscript', self.r_script, pops_path, data_path, hypers_path, self.iters, self.chains, self.seed, save_path, train_helper_file, diffcovs_model_file, os.getpid())
        print cmd
        subprocess.call(cmd, shell=True)
        posteriors = read_unheadered_posterior_traces(save_path)
        posteriors = read_posterior_traces(save_path)
        # set the column names of posterior traces
        a_datum = iter(data).next()
        posteriors['B_a'].columns = a_datum.xa.index
        posteriors['B_b'].columns = a_datum.xb.index
        posteriors['B_c'].columns = a_datum.xc.index
        posteriors['phi_a'].columns = ['phi_a']
        posteriors['phi_b'].columns = ['phi_b']
        posteriors['phi_c'].columns = ['phi_c']
        posteriors['phi_m'].columns = ['phi_m']
        return posteriors


class combined_posterior_getter_f(possibly_cached):
    """
    takes a list of functions that get posteriors, calls them with data, and combines the posteriors
    does keying using the last one in the list for now, just to save space
    """
    def key_f(self, data):
        return '%s_%s_%s' % ('combined_posterior', self.posterior_f_list.get_key(), data.get_key())

    def __init__(self, posterior_f_list):
        self.posterior_f_list = posterior_f_list

    def __call__(self, data):
        def combine_posterior(p1, p2):
            p = {}
            for key in p1:
                p[key] = concat([p1[key], p2[key]])
            return p
        posteriors = []
        for posterior_f in self.posterior_f_list:
            posteriors.append(posterior_f(data))
        return reduce(combine_posterior, posteriors)
    

class get_seed_seq_get_posterior_f_list_f(possibly_cached):
    """
    takes in partial of 
    """
    pass


class get_pystan_diffcovs_posterior_f(possibly_cached_folder):
    """
    returns posteriors for diffcovs model, using pystan instead of rstan
    """
    
    def get_introspection_key(self):
        return '%s_%s_%s_%d_%d_%d' % ('pydiffcovs', self.get_pops_f.get_key(), self.hypers.get_key(), self.iters, self.chains, self.seed)
        
    def key_f(self, data):
        return '%s_%s' % (self.get_key(), data.get_key())

    def location_f(self, data):
        return '%s/%s/%s' % (data.get_location(), 'pytrained_diffcovs', self.get_pops_f.get_key())

    print_handler_f = staticmethod(folder_adapter(write_posterior_traces))

    read_f = staticmethod(read_posterior_traces)

    def has_file_content(self, data):
        return os.path.exists('%s/%s' % (self.full_file_path_f(data), 'out_B_a.csv'))

    def __init__(self, get_pops_f, hypers, iters, chains, seed):
        self.get_pops_f, self.hypers, self.iters, self.chains, self.seed = get_pops_f, hypers, iters, chains, seed
        self.diffcovs_model_file = '%s/%s/%s' % (global_stuff.home, 'recovery_curve/stan_files', 'full_model_diffcovs.stan')

    #@save_and_memoize
    @key
    @read_from_pickle
    #@save_to_file
    @save_to_pickle
    def __call__(self, data):
        """
        need to convert data to proper form.  also need to convert to my form, which is a dictionary of dataframes, where keys are the original function objects
        """
        import pystan

        d = {}

        pops = self.get_pops_f(data)
        d['pop_a'] = pops.pop_a
        d['pop_b'] = pops.pop_b
        d['pop_c'] = pops.pop_c

        d['c_a'] = self.hypers.c_a
        d['c_b'] = self.hypers.c_b
        d['c_c'] = self.hypers.c_c
        d['l_a'] = self.hypers.l_a
        d['l_b'] = self.hypers.l_b
        d['l_c'] = self.hypers.l_c
        d['l_m'] = self.hypers.l_m

        d['N'] = len(data)
        _a_datum = iter(data).next()
        d['K_A'] = len(_a_datum.xa)
        d['K_B'] = len(_a_datum.xb)
        d['K_C'] = len(_a_datum.xc)

        xas = pandas.DataFrame({a_datum.pid:a_datum.xa for a_datum in data})
        xbs = pandas.DataFrame({a_datum.pid:a_datum.xb for a_datum in data})
        xcs = pandas.DataFrame({a_datum.pid:a_datum.xc for a_datum in data})
        d['xas'] = xas.T.as_matrix()
        d['xbs'] = xbs.T.as_matrix()
        d['xcs'] = xcs.T.as_matrix()

        d['ss'] = [a_datum.s for a_datum in data]

        ls = reduce(lambda x, a_datum: x + [len(a_datum.ys)], data, [])
        ts = reduce(lambda x, a_datum: x + a_datum.ys.index.tolist(), data, [])
        vs = reduce(lambda x, a_datum: x + a_datum.ys.tolist(), data, [])
        d['ls'] = ls
        d['ts'] = ts
        d['vs'] = vs
        d['L'] = len(ts)
        assert len(ts) == sum(ls)



        fit = pystan.stan(file=self.diffcovs_model_file, data=d, iter=self.iters, seed=self.seed, chains=self.chains, verbose=True)

        traces = fit.extract(permuted=True)
        
        # need to convert arrays to dataframes, and give them the same indicies as in data
        
        posteriors = keyed_dict({})
        posteriors['B_a'] = pandas.DataFrame(traces['B_a'])
        posteriors['B_a'].columns = _a_datum.xa.index
        posteriors['B_b'] = pandas.DataFrame(traces['B_b'])
        posteriors['B_b'].columns = _a_datum.xb.index
        posteriors['B_c'] = pandas.DataFrame(traces['B_c'])
        posteriors['B_c'].columns = _a_datum.xc.index
        posteriors['phi_a'] = pandas.DataFrame(traces['phi_a'])
        posteriors['phi_a'].columns = ['phi_a']
        posteriors['phi_b'] = pandas.DataFrame(traces['phi_b'])
        posteriors['phi_b'].columns = ['phi_b']
        posteriors['phi_c'] = pandas.DataFrame(traces['phi_c'])
        posteriors['phi_c'].columns = ['phi_c']
        posteriors['phi_m'] = pandas.DataFrame(traces['phi_m'])
        posteriors['phi_m'].columns = ['phi_m']

        # also extract the A_i,B_i,C_i parameters
        pids = [_datum.pid for _datum in data]
        pid_to_i = {pid:i for pid,i in zip(pids,xrange(N))}
        N = len(data)
        d_A, d_B, d_C = {}. {}. {}
        for i in xrange(N):
            d_A[pid_to_i] = traces[A][i]
            d_B[pid_to_i] = traces[B][i]
            d_C[pid_to_i] = traces[C][i]
        posteriors['As'] = pandas.DataFrame(d_A)
        posteriors['Bs'] = pandas.DataFrame(d_B)
        posteriors['Cs'] = pandas.DataFrame(d_C)

        return posteriors




class get_pystan_diffcovs_posterior_phi_m_fixed_f(possibly_cached_folder):
    """
    returns posteriors for diffcovs model, using pystan instead of rstan
    """
    
    def get_introspection_key(self):
        return '%s_%s_%s_%d_%d_%d' % ('pydiffcovs_phimfixed', self.get_pops_f.get_key(), self.hypers.get_key(), self.iters, self.chains, self.seed)
        
    def key_f(self, data):
        return '%s_%s' % (self.get_key(), data.get_key())

    def location_f(self, data):
        return '%s/%s/%s' % (data.get_location(), 'pytrained_diffcovs', self.get_pops_f.get_key())

    print_handler_f = staticmethod(folder_adapter(write_posterior_traces))

    read_f = staticmethod(read_posterior_traces)

    def has_file_content(self, data):
        return os.path.exists('%s/%s' % (self.full_file_path_f(data), 'out_B_a.csv'))

    def __init__(self, phi_m, get_pops_f, hypers, iters, chains, seed):
        self.get_pops_f, self.hypers, self.iters, self.chains, self.seed = get_pops_f, hypers, iters, chains, seed
        self.phi_m = phi_m
        self.diffcovs_model_file = '%s/%s/%s' % (global_stuff.home, 'recovery_curve/stan_files', 'full_model_diffcovs_phi_m_fixed.stan')

    #@save_and_memoize
    @key
    @read_from_pickle
    @save_to_file
    @save_to_pickle
    def __call__(self, data):
        """
        need to convert data to proper form.  also need to convert to my form, which is a dictionary of dataframes, where keys are the original function objects
        """
        import pystan

        d = {}

        pops = self.get_pops_f(data)
        d['pop_a'] = pops.pop_a
        d['pop_b'] = pops.pop_b
        d['pop_c'] = pops.pop_c

        d['c_a'] = self.hypers.c_a
        d['c_b'] = self.hypers.c_b
        d['c_c'] = self.hypers.c_c
        d['l_a'] = self.hypers.l_a
        d['l_b'] = self.hypers.l_b
        d['l_c'] = self.hypers.l_c
        #d['l_m'] = self.hypers.l_m

        d['N'] = len(data)
        _a_datum = iter(data).next()
        d['K_A'] = len(_a_datum.xa)
        d['K_B'] = len(_a_datum.xb)
        d['K_C'] = len(_a_datum.xc)

        xas = pandas.DataFrame({a_datum.pid:a_datum.xa for a_datum in data})
        xbs = pandas.DataFrame({a_datum.pid:a_datum.xb for a_datum in data})
        xcs = pandas.DataFrame({a_datum.pid:a_datum.xc for a_datum in data})
        d['xas'] = xas.T.as_matrix()
        d['xbs'] = xbs.T.as_matrix()
        d['xcs'] = xcs.T.as_matrix()

        d['ss'] = [a_datum.s for a_datum in data]

        ls = reduce(lambda x, a_datum: x + [len(a_datum.ys)], data, [])
        ts = reduce(lambda x, a_datum: x + a_datum.ys.index.tolist(), data, [])
        vs = reduce(lambda x, a_datum: x + a_datum.ys.tolist(), data, [])
        d['ls'] = ls
        d['ts'] = ts
        d['vs'] = vs
        d['L'] = len(ts)
        d['phi_m'] = self.phi_m
        assert len(ts) == sum(ls)



        fit = pystan.stan(file=self.diffcovs_model_file, data=d, iter=self.iters, seed=self.seed, chains=self.chains, verbose=True)

        traces = fit.extract(permuted=True)
        
        # need to convert arrays to dataframes, and give them the same indicies as in data
        posteriors = keyed_dict({})
        posteriors['B_a'] = pandas.DataFrame(traces['B_a'])
        posteriors['B_a'].columns = _a_datum.xa.index
        posteriors['B_b'] = pandas.DataFrame(traces['B_b'])
        posteriors['B_b'].columns = _a_datum.xb.index
        posteriors['B_c'] = pandas.DataFrame(traces['B_c'])
        posteriors['B_c'].columns = _a_datum.xc.index
        posteriors['phi_a'] = pandas.DataFrame(traces['phi_a'])
        posteriors['phi_a'].columns = ['phi_a']
        posteriors['phi_b'] = pandas.DataFrame(traces['phi_b'])
        posteriors['phi_b'].columns = ['phi_b']
        posteriors['phi_c'] = pandas.DataFrame(traces['phi_c'])
        posteriors['phi_c'].columns = ['phi_c']
        #posteriors['phi_m'] = pandas.DataFrame(traces['phi_m'])
        #posteriors['phi_m'].columns = ['phi_m']

        return posteriors



class get_pystan_diffcovs_truncated_posterior_f(possibly_cached_folder):
    """
    returns posteriors for diffcovs truncated normal_model, using pystan instead of rstan
    """
    
    def get_introspection_key(self):
        return '%s_%s_%s_%d_%d_%d' % ('pydiffcovs_truncated', self.get_pops_f.get_key(), self.hypers.get_key(), self.iters, self.chains, self.seed)
        
    def key_f(self, data):
        return '%s_%s' % (self.get_key(), data.get_key())

    def location_f(self, data):
        return '%s/%s/%s' % (data.get_location(), 'pytrained_diffcovs', self.get_pops_f.get_key())

    print_handler_f = staticmethod(folder_adapter(write_posterior_traces))

    read_f = staticmethod(read_posterior_traces)

    def has_file_content(self, data):
        return os.path.exists('%s/%s' % (self.full_file_path_f(data), 'out_B_a.csv'))

    def __init__(self, get_pops_f, hypers, iters, chains, seed):
        self.get_pops_f, self.hypers, self.iters, self.chains, self.seed = get_pops_f, hypers, iters, chains, seed
        self.diffcovs_model_file = '%s/%s/%s' % (global_stuff.home, 'recovery_curve/stan_files', 'full_model_diffcovs.stan')

    #@save_and_memoize
    @key
    @read_from_pickle
    @save_to_file
    @save_to_pickle
    def __call__(self, data):
        """
        need to convert data to proper form.  also need to convert to my form, which is a dictionary of dataframes, where keys are the original function objects
        """
        import pystan

        d = {}

        pops = self.get_pops_f(data)
        d['pop_a'] = pops.pop_a
        d['pop_b'] = pops.pop_b
        d['pop_c'] = pops.pop_c

        d['c_a'] = self.hypers.c_a
        d['c_b'] = self.hypers.c_b
        d['c_c'] = self.hypers.c_c
        d['l_a'] = self.hypers.l_a
        d['l_b'] = self.hypers.l_b
        d['l_c'] = self.hypers.l_c
        d['l_m'] = self.hypers.l_m

        d['N'] = len(data)
        _a_datum = iter(data).next()
        d['K_A'] = len(_a_datum.xa)
        d['K_B'] = len(_a_datum.xb)
        d['K_C'] = len(_a_datum.xc)

        xas = pandas.DataFrame({a_datum.pid:a_datum.xa for a_datum in data})
        xbs = pandas.DataFrame({a_datum.pid:a_datum.xb for a_datum in data})
        xcs = pandas.DataFrame({a_datum.pid:a_datum.xc for a_datum in data})
        d['xas'] = xas.T.as_matrix()
        d['xbs'] = xbs.T.as_matrix()
        d['xcs'] = xcs.T.as_matrix()

        d['ss'] = [a_datum.s for a_datum in data]

        ls = reduce(lambda x, a_datum: x + [len(a_datum.ys)], data, [])
        ts = reduce(lambda x, a_datum: x + a_datum.ys.index.tolist(), data, [])
        vs = reduce(lambda x, a_datum: x + a_datum.ys.tolist(), data, [])
        d['ls'] = ls
        d['ts'] = ts
        d['vs'] = vs
        d['L'] = len(ts)
        assert len(ts) == sum(ls)



        fit = pystan.stan(file=self.diffcovs_model_file, data=d, iter=self.iters, seed=self.seed, chains=self.chains, verbose=True)

        traces = fit.extract(permuted=True)
        
        # need to convert arrays to dataframes, and give them the same indicies as in data
        posteriors = keyed_dict({})
        posteriors['B_a'] = pandas.DataFrame(traces['B_a'])
        posteriors['B_a'].columns = _a_datum.xa.index
        posteriors['B_b'] = pandas.DataFrame(traces['B_b'])
        posteriors['B_b'].columns = _a_datum.xb.index
        posteriors['B_c'] = pandas.DataFrame(traces['B_c'])
        posteriors['B_c'].columns = _a_datum.xc.index
        posteriors['phi_a'] = pandas.DataFrame(traces['phi_a'])
        posteriors['phi_a'].columns = ['phi_a']
        posteriors['phi_b'] = pandas.DataFrame(traces['phi_b'])
        posteriors['phi_b'].columns = ['phi_b']
        posteriors['phi_c'] = pandas.DataFrame(traces['phi_c'])
        posteriors['phi_c'].columns = ['phi_c']
        posteriors['phi_m'] = pandas.DataFrame(traces['phi_m'])
        posteriors['phi_m'].columns = ['phi_m']

        return posteriors



class print_diffcovs_posterior_means(possibly_cached):
    """
    for each coefficient, make a plot.  x-axis is for different features, different line
    this should probably take in a list of posteriors rather than data
    """

    def get_introspection_key(self):
        return '%s_%s_%s' % ('meanplt', self.cv_f.get_key(), self.get_posterior_f.get_key())

    def key_f(self, data):
        return '%s_%s' % (self.get_key(), data.get_key())

    def location_f(self, data):
        return '%s/%s' % (global_stuff.data_home, 'posterior_means')

    print_handler_f = staticmethod(multiple_figures_to_pdf)

    def __init__(self, cv_f, get_posterior_f):
        self.cv_f, self.get_posterior_f = cv_f, get_posterior_f

    print_handler_f = staticmethod(multiple_figures_to_pdf)


    @save_to_file
    def __call__(self, data):
        posteriors = []
        for train_data, test_data in self.cv_f(data):
            posteriors.append(self.get_posterior_f(train_data))

        param_names = ['B_a', 'B_b', 'B_c']
        figs = []

        for param_name in param_names:

            fig = plt.figure()
            fig.suptitle(data.get_key(),fontsize=8)
            fig.subplots_adjust(bottom=0.4)
            ax = fig.add_subplot(1,1,1)
            ax.set_title(param_name)
            fold_posteriors = [posterior[param_name] for posterior in posteriors]
            for fold_posterior in fold_posteriors:
                fold_posterior_mean = fold_posterior.mean()
                num_feat = len(fold_posterior_mean)
                ax.set_xlim(-1,num_feat)
                ax.set_ylim(-5,5)
                ax.plot(range(num_feat), fold_posterior_mean, linestyle='None', marker='o')
                ax.set_xticks(range(num_feat))
                ax.set_xticklabels(fold_posterior_mean.index, rotation='vertical')
                xticks = ax.get_xticklabels()
                for xtick,i in zip(xticks,range(len(xticks))):
                    xtick.set_fontsize(10)
            figs.append(fig)

        return figs

class print_diffcovs_posterior_means_effect(possibly_cached):
    """
    for each coefficient, make a plot.  x-axis is for different features, different line
    this should probably take in a list of posteriors rather than data
    """

    def get_introspection_key(self):
        return '%s_%s_%s' % ('meaneffectplt', self.cv_f.get_key(), self.get_posterior_f.get_key())

    def key_f(self, data):
        return '%s_%s' % (self.get_key(), data.get_key())

    def location_f(self, data):
        return '%s/%s' % (global_stuff.data_home, 'posterior_means')

    print_handler_f = staticmethod(multiple_figures_to_pdf)

    def __init__(self, cv_f, get_posterior_f):
        self.cv_f, self.get_posterior_f = cv_f, get_posterior_f

    print_handler_f = staticmethod(multiple_figures_to_pdf)


    @save_to_file
    def __call__(self, data):
        posteriors = []
        for train_data, test_data in self.cv_f(data):
            posteriors.append(self.get_posterior_f(train_data))

        param_names = ['B_a', 'B_b', 'B_c']
        figs = []

        # figure out max/min of covariates
        covs = pandas.DataFrame({_datum.pid:_datum.xa for _datum in data})
        feat_range = covs.max(axis=1) - covs.min(axis=1)
        D = np.diag(feat_range)
        for param_name in param_names:

            fig = plt.figure()
            fig.suptitle(data.get_key(),fontsize=8)
            fig.subplots_adjust(bottom=0.4)
            ax = fig.add_subplot(1,1,1)
            ax.set_title(param_name)
            fold_posteriors = [posterior[param_name] for posterior in posteriors]
            for fold_posterior in fold_posteriors:
                fold_posterior_mean = fold_posterior.mean()
                num_feat = len(fold_posterior_mean)
                ax.set_xlim(-1,num_feat)
                ax.set_ylim(-5,5)
                ax.plot(range(num_feat), fold_posterior_mean, linestyle='None', marker='o', color='black')

                ax.plot(range(num_feat), D.dot(fold_posterior_mean), linestyle='None', marker='o', color='blue')                
                ax.set_xticks(range(num_feat))
                ax.set_xticklabels(fold_posterior_mean.index, rotation='vertical')
                xticks = ax.get_xticklabels()
                for xtick,i in zip(xticks,range(len(xticks))):
                    xtick.set_fontsize(10)
            figs.append(fig)

        return figs




class plot_diffcovs_posterior_f(possibly_cached):
    """
    takes in traces, plots histogram of training folds from cv on data.  takes in data, cv_f, get_posterior_f
    """
    def get_introspection_key(self):
        return '%s_%s_%s' % ('postplt', self.cv_f.get_key(), self.get_posterior_f.get_key())

    def key_f(self, data):
        return '%s_%s' % (self.get_key(), data.get_key())

    def location_f(self, data):
        return '%s/%s' % (global_stuff.data_home, 'posterior_plots')

    print_handler_f = staticmethod(multiple_figures_to_pdf)

    def __init__(self, num_row, num_col, cv_f, get_posterior_f):
        self.num_row, self.num_col = num_row, num_col
        self.cv_f, self.get_posterior_f = cv_f, get_posterior_f

    @key
    @save_to_file
    def __call__(self, data):
        """
        output is a list of figures
        """

        roll = pp_roll(2,3)
        xlim_left, xlim_right = -4, 4
        B_bins = 20
        phi_bins = 40
        title_size = 5.5
        posteriors = []
        alpha=0.25
        for train_data, test_data in self.cv_f(data):
            posteriors.append(self.get_posterior_f(train_data))

        B_a_iteritems = [posterior['B_a'].iteritems() for posterior in posteriors]
        for feat_cols in itertools.izip(*B_a_iteritems):
            ax = roll.get_axes()
            for feat, col in feat_cols:
                ax.hist(col, alpha=alpha, bins=B_bins)
            ax.set_title('%s %s' % (str(feat), 'B_a'), fontdict={'fontsize':title_size})
            ax.set_xlim(xlim_left, xlim_right)

        roll.start_new_page()

        B_b_iteritems = [posterior['B_b'].iteritems() for posterior in posteriors]
        for feat_cols in itertools.izip(*B_b_iteritems):
            ax = roll.get_axes()
            for feat, col in feat_cols:
                ax.hist(col, alpha=alpha, bins=B_bins)
            ax.set_title('%s %s' % (str(feat), 'B_b'), fontdict={'fontsize':title_size})
            ax.set_xlim(xlim_left, xlim_right)

        roll.start_new_page()

        B_c_iteritems = [posterior['B_c'].iteritems() for posterior in posteriors]
        for feat_cols in itertools.izip(*B_c_iteritems):
            ax = roll.get_axes()
            for feat, col in feat_cols:
                ax.hist(col, alpha=alpha, bins=B_bins)
            ax.set_title('%s %s' % (str(feat), 'B_c'), fontdict={'fontsize':title_size})
            ax.set_xlim(xlim_left, xlim_right)

        roll.start_new_page()

        phi_a_cols = [posterior['phi_a']['phi_a'] for posterior in posteriors]
        ax = roll.get_axes()
        for phi_a_col in phi_a_cols:
            ax.hist(phi_a_col, alpha=alpha, bins=phi_bins)
        ax.set_title('phi_a')

        phi_b_cols = [posterior['phi_b']['phi_b'] for posterior in posteriors]

        ax = roll.get_axes()
        for phi_b_col in phi_b_cols:
            ax.hist(phi_b_col, alpha=alpha, bins=phi_bins)
        ax.set_title('phi_b')

        phi_c_cols = [posterior['phi_c']['phi_c'] for posterior in posteriors]
        ax = roll.get_axes()
        for phi_c_col in phi_c_cols:
            ax.hist(phi_c_col, alpha=alpha, bins=phi_bins)
        ax.set_title('phi_c')

        phi_m_cols = [posterior['phi_m']['phi_m'] for posterior in posteriors]
        ax = roll.get_axes()
        for phi_m_col in phi_m_cols:
            ax.hist(phi_m_col, alpha=alpha, bins=phi_bins)
        ax.set_title('phi_m')

        return roll.figs


class plot_single_posterior_f(possibly_cached):
    """
    takes in traces, plots histogram of training folds from cv on data.  takes in data, cv_f, get_posterior_f
    """
    def get_introspection_key(self):
        return 'singlepostplt'

    def key_f(self, posteriors):
        return '%s_%s' % (self.get_key(), posteriors.get_key())

    def location_f(self, posteriors):
        return '%s/%s' % (global_stuff.data_home, 'single_posterior_plots')

    print_handler_f = staticmethod(multiple_figures_to_pdf)

    def __init__(self, num_row, num_col):
        self.num_row, self.num_col = num_row, num_col

    @key
    @save_to_file
    def __call__(self, posteriors):
        """
        output is a list of figures
        for now, 1 plot per element, labelled with mean, std
        """

        roll = pp_roll(self.num_col, self.num_row)

        xlim_left, xlim_right = -4, 4
        B_bins = 20
        phi_bins = 40

        param_names = ['B_a','B_b','B_c','phi_a','phi_b','phi_c','phi_m']
        true_vals = [-2,1,2, 0.05, 0.05, 0.05, 0.05]

        for param_name, true_val in zip(param_names, true_vals):
            for col_name, col in posteriors[param_name].iteritems():
                print param_name, col_name, col.shape
                ax = roll.get_axes()
                ax.hist(col, bins=40, alpha=0.5)
                for tick in ax.xaxis.get_major_ticks():
                    tick.label.set_fontsize(6) 
                ax.set_title('%s t:%.2f m:%.2f s:%.2f' % (param_name, true_val, col.mean(), col.std()))
                ax.axvline(x=true_val, linewidth=4)
            roll.start_new_page()


        return roll.figs

class full_model_point_predictor(keyed_object, point_predictor):
    """
    given point estimates of posterior params, does prediction
    """
    display_color = 'red'

    display_name = 'full'

    def get_introspection_key(self):
        return '%s_%s_%s' % ('full_pred', self.params.get_key(), self.pops.get_key())

    def __init__(self, params, pops):
        self.params, self.pops = params, pops

    def __call__(self, datum, time):
        a = g_a(self.pops.pop_a, datum.xa, self.params['B_a'])
        b = g_b(self.pops.pop_b, datum.xb, self.params['B_b'],)
        c = g_c(self.pops.pop_c, datum.xc, self.params['B_c'],)
        return the_f(time, datum.s, a, b, c)


class get_param_mean_f(possibly_cached):
    """
    returns dictionary of mean params
    """
    def get_introspection_key(self):
        return 'param_mean'

    def key_f(self, post_param):
        return '%s_%s' % (self.get_key(), post_param.get_key())

    @key
    @memoize
    def __call__(self, post_params):
        return keyed_dict({p:v.apply(pandas.Series.mean, axis=0) for p,v in post_params.iteritems()})
        

class get_diffcovs_point_predictor_f(possibly_cached):
    """
    return trained object that makes point predictions
    have to make sure the point_predictor_f (which is hard-coded right now) is compatible with posteriors.  
    """
    display_color = full_model_point_predictor.display_color

    display_name = full_model_point_predictor.display_name

    def get_introspection_key(self):
        return '%s_%s_%s' % ('fullpred_f', self.get_diffcovs_posterior_f.get_key(), self.summarize_posterior_f.get_key())

    def key_f(self, data):
        return '%s_%s' % (self.get_key(), data.get_key())

    def __init__(self, get_diffcovs_posterior_f, summarize_posterior_f):
        self.get_diffcovs_posterior_f, self.summarize_posterior_f = get_diffcovs_posterior_f, summarize_posterior_f

    @key
    def __call__(self, data):
        posteriors = self.get_diffcovs_posterior_f(data)
        params = self.summarize_posterior_f(posteriors)
        # assuming that get_diffcovs_posterior_f has a get_pops_f attribute i can call
        pops = self.get_diffcovs_posterior_f.get_pops_f(data)
        return full_model_point_predictor(params, pops)


class mean_f(keyed_object):

    def get_introspection_key(self):
        return 'meanf'

    def __call__(self, l):
        total = 0.0
        count = 0.0
        for x in l:
            total += x
            count += 1
        return total/count


        

class nonpoint_predictor_f(keyed_object, nonpoint_predictor_base):

    display_color = 'orange'

    display_name = 'f(t) post'

    def get_introspection_key(self):
        return 'nonpoint'
        return self.point_f.get_key()
        return '%s_%s_%s' % (self.posteriors.get_key(), self.point_f.get_key(), self.pops.get_key())
        return '%s_%s_%s' % (self.posteriors.get_key(), self.point_f.get_key())

    def __init__(self, posteriors, pops):
        self.posteriors, self.pops = posteriors, pops
    
    def __call__(self, datum, time):
        print datum.pid, time
        if time == 1:
            import sys
            sys.stdout.flush()
            print datum.pid
        if datum.pid in posteriors['As'].columns:
            f = functools.partial(the_f, time, datum.s)
            return pandas.Series(list(itertools.imap(f, posteriors['As'][datum.pid], posteriors['Bs'][datum.pid], posteriors['Cs'][datum.pid])))
        assert False
        f = functools.partial(the_f, time, datum.s)
        return pandas.Series(list(itertools.imap(f, itertools.imap(functools.partial(g_a, self.pops.pop_a, datum.xa), itertools.imap(lambda x:x[1], self.posteriors['B_a'].iterrows())), itertools.imap(functools.partial(g_b, self.pops.pop_b, datum.xb), itertools.imap(lambda x:x[1], self.posteriors['B_b'].iterrows())), itertools.imap(functools.partial(g_c, self.pops.pop_c, datum.xc), itertools.imap(lambda x:x[1], self.posteriors['B_c'].iterrows())))))

        #return self.point_f(itertools.imap(f, itertools.imap(functools.partial(g_a, self.pops.pop_a, datum.xa), itertools.imap(lambda x:x[1], self.posteriors['B_a'].iterrows())), itertools.imap(functools.partial(g_b, self.pops.pop_b, datum.xb), itertools.imap(lambda x:x[1], self.posteriors['B_b'].iterrows())), itertools.imap(functools.partial(g_c, self.pops.pop_c, datum.xc), itertools.imap(lambda x:x[1], self.posteriors['B_c'].iterrows()))))

class get_diffcovs_nonpoints_predictor_f(possibly_cached):
    """
    return trained object that makes point predictions
    """
    display_color = full_model_point_predictor.display_color

    display_name = full_model_point_predictor.display_name

    def get_introspection_key(self):
        return '%s_%s' % ('nonpt_f', self.get_diffcovs_posterior_f.get_key())

    def key_f(self, data):
        return '%s_%s' % (self.get_key(), data.get_key())

    def __init__(self, get_diffcovs_posterior_f):
        self.get_diffcovs_posterior_f = get_diffcovs_posterior_f

    @key
    def __call__(self, data):
        posteriors = self.get_diffcovs_posterior_f(data)
        
        # assuming that get_diffcovs_posterior_f has a get_pops_f attribute i can call
        pops = self.get_diffcovs_posterior_f.get_pops_f(data)
        return nonpoint_predictor_f(posteriors, pops)




class logreg_predictor(keyed_object, point_predictor):
    """
    
    """

    display_color = 'orange'

    display_name = 'logreg'

    def __repr__(self):
        return 'logreg'

    def get_introspection_key(self):
        return 'log_pred'

    def __init__(self, params):
        self.params = params

    def __call__(self, datum, time):
        return logistic(self.params[time].dot(datum.xa))

class get_logreg_predictor_f(possibly_cached):
    """
    returns trained logreg predictor
    """
    display_color = logreg_predictor.display_color

    display_name = logreg_predictor.display_name

    def get_introspection_key(self):
        return 'logpred_f'

    def key_f(self, data):
        return '%s_%s' % (self.get_key(), data.get_key())

    def __init__(self, times):
        self.times = times

    @key
    def __call__(self, _data):
        """
        for now, use only xa for prediction
        """
        Bs = {}
        xas = pandas.DataFrame({datum.pid:datum.xa for datum in _data})
        for time in self.times:
            y_d = {}
            for datum in _data:
                try:
                    y_d[datum.pid] = datum.ys[time]
                except KeyError:
                    pass
            y = pandas.Series(y_d)
            this_xas = xas[y.index]
            Bs[time] = train_logistic_model(this_xas, y)
        return logreg_predictor(pandas.DataFrame(Bs))


class logregshape_predictor(keyed_object, point_predictor):
    """
    
    """

    display_color = 'cyan'

    display_name = 'logregshape'

    def __repr__(self):
        return 'logregshape'

    def get_introspection_key(self):
        return 'logshape_pred'

    def __init__(self, params):
        self.params = params

    def __call__(self, datum, time):
        return datum.s*logistic(self.params[time].dot(datum.xa))

class get_logregshape_predictor_f(possibly_cached):
    """
    returns trained logreg predictor, where logistic regression predicts the value *proportional* to s
    generative error term assumed to have variance proportional to s
    """
    display_color = logregshape_predictor.display_color

    display_name = logregshape_predictor.display_name

    def get_introspection_key(self):
        return 'logshape_pred_f'

    def key_f(self, data):
        return '%s_%s' % (self.get_key(), data.get_key())

    def __init__(self, times):
        self.times = times

    @key
    def __call__(self, _data):
        """
        for now, use only xa for prediction
        """
        Bs = {}
        xas = pandas.DataFrame({datum.pid:datum.xa for datum in _data})
        for time in self.times:
            y_d = {}
            for datum in _data:
                try:
                    if datum.s > 0.1:
                        y_d[datum.pid] = datum.ys[time] / datum.s
                except KeyError:
                    pass
            y = pandas.Series(y_d)
            this_xas = xas[y.index]
            Bs[time] = train_logistic_model(this_xas, y)
        return logreg_predictor(pandas.DataFrame(Bs))


class logregshapeunif_predictor(keyed_object, point_predictor):
    """
    
    """

    display_color = 'brown'

    display_name = 'logregshapeunif'

    def __repr__(self):
        return 'logregshapeunif'

    def get_introspection_key(self):
        return 'logshapeunif_pred'

    def __init__(self, params):
        self.params = params

    def __call__(self, datum, time):
        return datum.s*logistic(self.params[time].dot(datum.xa))

class get_logregshapeunif_predictor_f(possibly_cached):
    """
    returns trained logreg predictor, where logistic regression predicts the value *proportional* to s
    generative error term assumed to have independent of initial value
    """
    display_color = logregshapeunif_predictor.display_color

    display_name = logregshapeunif_predictor.display_name

    def get_introspection_key(self):
        return 'logshapeunif_pred_f'

    def key_f(self, data):
        return '%s_%s' % (self.get_key(), data.get_key())

    def __init__(self, times):
        self.times = times

    @key
    def __call__(self, _data):
        """
        for now, use only xa for prediction
        """
        Bs = {}
        xas = pandas.DataFrame({datum.pid:datum.xa for datum in _data})
        S = pandas.Series({datum.pid:datum.s for datum in _data})
        for time in self.times:
            y_d = {}
            for datum in _data:
                try:
                    y_d[datum.pid] = datum.ys[time]
                except KeyError:
                    pass
            y = pandas.Series(y_d)
            this_xas = xas[y.index]
            this_S = S[y.index]
            Bs[time] = train_logistic_shapeunif_model(this_xas, y, this_S)
        return logreg_predictor(pandas.DataFrame(Bs))


class plot_logreg_coefficients_fig_f(possibly_cached):
    """
    takes in data, uses a get_logreg_predictor_f to get a trainer containing the coefficients
    then, makes a plot where for each time point, plot 1 dot for each component of logreg at that time
    """

    @save_to_file
    def __call__(self, logreg_predictor):
        fig = plt.figure()
        ax = fig.add_subplot(1,1,1)
        ax.set_title(logreg_predictor.get_key(), fontdict={'fontsize':7})
        Bs = logreg_predictor.params
        times = sorted(Bs.keys())
        a_param = Bs[iter(Bs).next()]
        for f in a_param.index:
            ax.plot(times, [Bs[time][f] for time in times], label=f, linestyle='--', marker='o')
        ax.legend(prop={'size':5})
        return fig

    def get_introspection_key(self):
        return 'logregcoefplt'

    def key_f(self, data):
        return '%s_%s' % (self.get_key(), data.get_key())

    def location_f(self, data):
        return '%s/%s' % (global_stuff.data_home, 'logregcoeffplt')

    print_handler_f = staticmethod(figure_to_pdf)

class cross_validated_scores_f(possibly_cached):
    """
    get_predictor_f is a factory for trained predictor objects, as in it does training
    return dataframe where columns are patients and rows are times
    """
    def get_introspection_key(self):
        return '%s_%s_%s' % ('cvscore', self.get_predictor_f.get_key(), self.cv_f.get_key())

    def key_f(self, data):
        return '%s_%s' % (self.get_key(), data.get_key())

    def location_f(self, data):
        return '%s/%s' % (global_stuff.data_home, 'scores')

    print_handler_f = staticmethod(write_DataFrame)

    read_f = staticmethod(read_DataFrame)

    def __init__(self, get_predictor_f, cv_f, times):
        self.get_predictor_f, self.cv_f, self.times = get_predictor_f, cv_f, times

    @key
    @save_and_memoize
    def __call__(self, data):
        fold_scores = []
        for train_data, test_data in self.cv_f(data):
            predictor = self.get_predictor_f(train_data)
            fold_scores.append(pandas.DataFrame({datum.pid:{time:predictor(datum, time) for time in self.times} for datum in test_data}))
        return keyed_DataFrame(pandas.concat(fold_scores, axis=1))


class plot_predictions_fig_f(possibly_cached):

    def __init__(self, predictors):
        self.predictors = predictors

    def __call__(self, datum):
        fig = plt.figure()
        ax = fig.add_subplot(1,1,1)
        ys = datum.ys
        ts = datum.ys.index
        for predictor in self.predictors:
            prediction_series = predictor.get_prediction_series(datum, ts)
            predictor.plot_prediction(ax, prediction_series)
        add_series_to_ax(ys, ax, 'black', 'true', 'solid')
        try:
            abc_ys = pandas.Series({time:the_f(time,datum.s,datum.true_a,datum.true_b,datum.true_c) for time in np.linspace(0,50,200)})
            add_series_to_ax(abc_ys, ax, 'green', 'sim', 'solid')
        except AttributeError:
            pass
        ax.legend()
        ax.plot(-1,datum.s,'bo')
        ax.set_xlim(-2,50)
        ax.set_ylim(-0.1, 1.1)
        ax.set_xlabel('time')
        ax.set_ylabel('fxn val')
        ax.set_title(datum.pid)
        return fig

class plot_all_predictions_fig_f(possibly_cached):

    def get_introspection_key(self):
        return '%s_%s' % (self.get_predictor_fs.get_key(), self.cv_f.get_key())

    def key_f(self, data):
        return '%s_%s' % (self.get_key(), data.get_key())

    def location_f(self, data):
        return '%s/%s' % (global_stuff.data_home, 'prediction_plots')

    print_handler_f = staticmethod(multiple_figures_to_pdf)

    def __init__(self, get_predictor_fs, cv_f, times):
        self.get_predictor_fs, self.cv_f, self.times = get_predictor_fs, cv_f, times

    @save_to_file
    def __call__(self, data):
        fig_structs = []
        figs = []

        for train_data, test_data in self.cv_f(data):
            predictors = keyed_list([get_predictor_f(train_data) for get_predictor_f in self.get_predictor_fs])

            fig_structs += [[datum,plot_predictions_fig_f(predictors)(datum)] for datum in test_data if datum.pid % 100 == 0]

            #figs += [plot_predictions_fig_f(predictors)(datum) for datum in test_data]

        sorted_fig_structs = sorted(fig_structs, key = lambda x: x[0].s)
        return [x[1] for x in sorted_fig_structs]
        return figs
class cv_fold_f(possibly_cached):
    """
    returns (training,testing) folds as a list.  each element of list needs to be keyed
    """
    def get_introspection_key(self):
        return '%s_%s' % ('cv', self.fold_k)

    def key_f(self, data):
        return 'asdf'

    def __init__(self, fold_k):
        self.fold_k = fold_k

    @key
    def __call__(self, data):
        folds = keyed_list()
        for i in range(self.fold_k):
            train_data = get_data_fold_training(i, self.fold_k)(data)
            test_data = get_data_fold_testing(i, self.fold_k)(data)
            folds.append(keyed_list([train_data, test_data]))
        return folds
            

class self_test_cv_f(possibly_cached):

    def get_introspection_key(self):
        return 'selfcv'

    
    def __call__(self, data):
        folds = keyed_list()
        folds.append(keyed_list([data,data]))
        return folds


class get_true_val(keyed_object):

    def __call__(self, _datum, time):
        return _datum.ys[time]

    def get_introspection_key(self):
        return 'actual'

    display_name = 'actual'

    display_color = 'black'

class get_true_val_abc_sim(keyed_object):
    """
    assumes input is a simulated datapoint. returns f(t;a,b,c) where a,b,c are actual values used to simulate
    """
    def __call__(self, _datum, time):
        return the_f(time, _datum.s, _datum.true_a, _datum.true_b, _datum.true_c)

    def get_introspection_key(self):
        return 'truevalsim'

    display_name = 'sim'

    display_color = 'green'

class get_true_val_abc_fit(keyed_object):
    """
    returns interpolated value
    """
    def get_introspection_key(self):
        return 'truevalinterp'

    def __call__(self, _datum, time):
        fit_a, fit_b, fit_c = get_curve_abc(_datum.s, _datum.ys)
        return the_f(time, _datum.s, fit_a, fit_b, fit_c)

    display_name = 'interpolated'

    display_color = 'red'

class loss_f(keyed_object):

    def get_introspection_key(self):
        return '%s_%s' % (self.get_true_val_f.get_key(), self.distance_f.get_key())

    def __init__(self, get_true_val_f, distance_f):
        self.get_true_val_f, self.distance_f = get_true_val_f, distance_f

    def __call__(self, _datum, time, score):
        return self.distance_f(self.get_true_val_f(_datum, time), score)



class scaled_logistic_loss_f(keyed_object):

    def get_introspection_key(self):
        return '%s_%.2f' % ('logloss', self.c)

    def __init__(self, c):
        self.c = c

    def __call__(self, true_val, pred):
        return scaled_logistic_loss(abs(true_val-pred), self.c)

class abs_loss_f(keyed_object):

    def get_introspection_key(self):
        return 'absloss'

    def __call__(self, true_val, pred):
        return abs(true_val - pred)


class signed_loss_f(keyed_object):

    def get_introspection_key(self):
        return 'signedloss'

    def __call__(self, true_val, pred):
        return true_val - pred

class performance_series_f(possibly_cached):
    """
    returns a dataframe.  mean score will be the first column
    """

    def __init__(self, loss_f, percentiles, data, times):
        self.loss_f, self.percentiles, self.data, self.times = loss_f, percentiles, data, times

    @key
    @save_and_memoize
    def __call__(self, scores):

        losses = pandas.DataFrame({pid:{time:self.loss_f(self.data[pid], time, score) for time, score in scores.iteritems()} for pid, scores in scores.iteritems()})


        # have raw scores for each patient
        #true_ys = pandas.DataFrame({datum.pid:datum.ys for datum in self.data if datum.pid in scores.columns})
        #diff = (scores - true_ys).abs()
        #losses = diff.apply(self.loss_f, axis=1)
        #losses = losses.ix[self.times]
        mean_losses = losses.apply(np.mean, axis=1)
        loss_percentiles = losses.apply(functools.partial(get_percentiles, percentiles=self.percentiles), axis=1)
        loss_percentiles['mean'] = mean_losses
        return keyed_DataFrame(loss_percentiles)
        

    def get_introspection_key(self):
        return '%s_%s' % ('perf', self.loss_f.get_key())

    def key_f(self, data):
        return '%s_%s' % (self.get_key(), data.get_key())

    def location_f(self, data):
        return '%s/%s' % (data.get_location(), 'performances')

    print_handler_f = staticmethod(write_DataFrame)

    read_f = staticmethod(read_DataFrame)

class model_comparer_f(possibly_cached):
    """
    hard code the loss functions used
    """

    def __init__(self, trainers, cv_f, loss_f, percentiles, times):
        self.trainers, self.cv_f, self.loss_f, self.percentiles, self.times = trainers, cv_f, loss_f, percentiles, times

    @save_to_file
    @memoize
    def __call__(self, data):
        fig = plt.figure()
        ax = fig.add_subplot(1,1,1)
        ax.set_title(self.loss_f.get_key())
        for trainer in self.trainers:
            scores_getter = cross_validated_scores_f(trainer, self.cv_f, self.times)
            scores = scores_getter(data)
            _performance_series_f = performance_series_f(self.loss_f, self.percentiles, data, self.times)
            perfs = _performance_series_f(scores)
            add_performances_to_ax(ax, perfs, trainer.display_color, trainer.display_name)
        ax.set_xlim(-1.0,50)
        ax.set_ylim(-0.5,1.1)
        ax.legend()
        fig.show()
        return fig

    def get_introspection_key(self):
        return self.trainers[0].get_key()
        return '%s_%s_%s_%s' % ('bs', self.trainers.get_key(), self.cv_f.get_key(), self.loss_f.get_key())

    def key_f(self, data):
        return '%s_%s' % (self.get_key(), data.get_key())

    def location_f(self, data):
        return '%s/%s' % (global_stuff.data_home,'betweenmodel_perfs')

    print_handler_f = staticmethod(figure_to_pdf)

    read_f = staticmethod(not_implemented_f)


class loss_comparer_f(possibly_cached):
    """
    for a single model, plot several loss functions.  each plot has a fixed get_true_val
    """
    print_handler_f = staticmethod(multiple_figures_to_pdf)

    def __init__(self, scores_getter, distance_fs, get_true_val_fs, percentiles, times):
        self.scores_getter, self.distance_fs, self.get_true_val_fs, self.percentiles, self.times = scores_getter, distance_fs, get_true_val_fs, percentiles, times

    @save_to_file
    def __call__(self, data):
        scores = self.scores_getter(data)
        figs = []
        for distance_f in self.distance_fs:
            fig = plt.figure()
            fig.suptitle(distance_f.get_key())
            ax = fig.add_subplot(1,1,1)
            ax.set_xlabel('time')
            ax.set_ylabel('loss')
            for get_true_val_f in self.get_true_val_fs:
                _loss_f = loss_f(get_true_val_f, distance_f)
                _performance_series_f = performance_series_f(_loss_f, self.percentiles, data, self.times)
                perfs = _performance_series_f(scores)
                add_performances_to_ax(ax, perfs, get_true_val_f.display_color, get_true_val_f.display_name)
            ax.legend()
            figs.append(fig)
        return figs

    def get_introspection_key(self):
        return '%s_%s_%s' % (self.scores_getter.get_key(), self.distance_fs.get_key(), self.get_true_val_fs.get_key())

    def key_f(self, data):
        return '%s_%s' % (self.get_key(), data.get_key())

    def location_f(self, data):
        return '%s/%s' % (global_stuff.data_home,'several_losses')

class stratified_model_comparer_f(possibly_cached):
    """
    hard code the loss functions used
    accepts a categorical feature to stratify
    """

    def __init__(self, trainers, cv_f, loss_f, percentiles, times, cat_f):
        self.trainers, self.cv_f, self.loss_f, self.percentiles, self.times = trainers, cv_f, loss_f, percentiles, times

    @save_to_file
    @memoize
    def __call__(self, data):
        fig = plt.figure()
        # have a dict from feature to ax.  create the axes using a pp_roll
        roll = pp_roll(2,2)
        f_to_ax = {}
        for f in self.cat_f:
            ax = roll.get_axes()
            ax.set_title('%s %s' % (f.get_key(), self.loss_f.get_key()))
            ax.set_xlim(-1.0,50)
            f_to_ax[f] = ax
        
        for trainer in self.trainers:
            scores_getter = cross_validated_scores_f(trainer, self.cv_f, self.times)
            scores = scores_getter(data)
            _performance_series_f = performance_series_f(self.loss_f, self.percentiles, data)
            for f in self.cat_f:
                ok_pids = [pid for pid in scores.columns if f(pid)]
                this_scores = scores[ok_pids]
                this_perfs = _performance_series_f(scores)
                add_performances_to_ax(f_to_ax[f], this_perfs, trainer.display_color, trainer.display_name)

        return roll.figs

    def get_introspection_key(self):
        return '%s_%s_%s_%s' % ('btwmodperf', self.trainers.get_key(), self.cv_f.get_key(), self.loss_f.get_key())

    def key_f(self, data):
        return '%s_%s' % (self.get_key(), data.get_key())

    def location_f(self, data):
        return '%s/%s/%s' % (global_stuff.data_home,'betweenmodel_perfs', data.get_key())

    print_handler_f = staticmethod(figure_to_pdf)

    read_f = staticmethod(not_implemented_f)


"""
what 'data' means for this project
"""

class datum(keyed_object):

    def __init__(self, pid, xa, xb, xc, s, ys):
        self.pid, self.xa, self.xb, self.xc, self.s, self.ys = pid, xa, xb, xc, s, ys

    def __repr__(self):
        return self.pid

    def __eq__(self, other):
        return self.pid == other.pid

    def __hash__(self):
        return hash(self.pid)

class simulated_datum(datum):
    def __init__(self, pid, xa, xb, xc, s, ys, true_a, true_b, true_c):
        self.true_a, self.true_b, self.true_c = true_a, true_b, true_c
        datum.__init__(self, pid, xa, xb, xc, s, ys)

        


class data(keyed_list):
    """
    also offer dictionary[pid] capability
    """
    def __init__(self, l):
        self.d = {_datum.pid:_datum for _datum in l}
        keyed_list.__init__(self, l)

    def get_introspection_key(self):
        return 'data'

class s_f(feat):

    def get_introspection_key(self):
        return '%s_%s' % ('s', self.ys_f.get_key())

    def __init__(self, ys_f):
        self.ys_f = ys_f

    @raise_if_na
    def __call__(self, pid):
        try:
            return self.ys_f(pid)[0]
        except:
            raise my_Exception

class get_data_f(possibly_cached):

    def __init__(self, x_abc_fs, s_f, ys_f):
        self.x_abc_fs, self.s_f, self.ys_f = x_abc_fs, s_f, ys_f

    def get_introspection_key(self):
        return '%s_%s_%s' % ('data', self.x_abc_fs.get_key(), self.ys_f.get_key())

    def key_f(self, pid_iterator):
        return '%s_%s' % (self.get_key(), pid_iterator.get_key())

    def location_f(self, pid_iterator):
        return '%s/%s/%s' % (global_stuff.data_home, 'data', self.get_key())

    print_handler_f = staticmethod(folder_adapter(write_diffcovs_data))

    read_f = staticmethod(read_diffcovs_data)

    @key
    @save_and_memoize
    #@read_from_file
    #@read_from_pickle
    @save_to_pickle
    def __call__(self, pid_iterator):
        """
        change so that get rid of any na's that may be in ys
        """

        l = []

        for pid in pid_iterator:
            try:
                xa = get_feature_series(pid, self.x_abc_fs.xa_fs)
                xb = get_feature_series(pid, self.x_abc_fs.xb_fs)
                xc = get_feature_series(pid, self.x_abc_fs.xc_fs)
                s = self.s_f(pid)
                ys = self.ys_f(pid).dropna()
            except Exception, e:
                print e, '2'
                pass
            else:
                l.append(datum(pid, xa, xb, xc, s, ys))
        l = sorted(l, key = lambda x: x.pid)
        return data(keyed_list(l))


class simulated_get_data_f(possibly_cached):
    """
    re-seed random number generator at every call
    all random calls from here are fed the freshly seeded random.Random()
    """

    def __init__(self, params, pops, id_to_x_s_f, times, seed=0):
        self.params, self.pops, self.id_to_x_s_f, self.times, self.seed = params, pops, id_to_x_s_f, times, seed

    @key
    @save_to_pickle
    @save_to_file
    def __call__(self, pid_iterator):
        r = random.Random(self.seed)
        l = []
        for pid in pid_iterator:
            x,s = self.id_to_x_s_f(pid,r)
            assert len(x) == len(self.params['B_a'])
            m_a = g_a(self.pops.pop_a, x, self.params['B_a'])
            m_b = g_b(self.pops.pop_b, x, self.params['B_b'])
            m_c = g_c(self.pops.pop_c, x, self.params['B_c'])
            a = get_rand_beta(m_a, self.params['phi_a'], r)
            b = get_rand_beta(m_b, self.params['phi_b'], r)
            c = get_rand_gamma(m_c, self.params['phi_c'], r)
            def gen(t):
                return r.normalvariate(the_f(t,s,a,b,c), self.params['phi_m'])
            ys = pandas.Series(self.times, index=self.times).apply(gen)
            l.append(simulated_datum(pid, x, x, x, s, ys, a, b, c))
        return data(l)

    def get_introspection_key(self):
        return '%s_%s_%s_%s' % ('sim_data_f', self.params.get_key(), self.pops.get_key(), self.id_to_x_s_f.get_key())

    def key_f(self, pid_iterator):
        return '%s_%s' % (self.get_key(), pid_iterator.get_key())

    def location_f(self, pid_iterator):
        return '%s/%s/%s' % (global_stuff.data_home, 'sim_data', self.get_key())

    print_handler_f = staticmethod(folder_adapter(write_diffcovs_data))

    read_f = staticmethod(read_diffcovs_data)


class simulated_get_data_truncate_0_f(possibly_cached):
    """
    re-seed random number generator at every call
    all random calls from here are fed the freshly seeded random.Random()
    """

    def __init__(self, params, pops, id_to_x_s_f, times, seed=0):
        self.params, self.pops, self.id_to_x_s_f, self.times, self.seed = params, pops, id_to_x_s_f, times, seed

    @key
    #@read_from_file
    @save_to_file
    def __call__(self, pid_iterator):
        r = random.Random(self.seed)
        l = []
        for pid in pid_iterator:
            x,s = self.id_to_x_s_f(pid,r)
            assert len(x) == len(self.params['B_a'])
            m_a = g_a(self.pops.pop_a, x, self.params['B_a'])
            m_b = g_b(self.pops.pop_b, x, self.params['B_b'])
            m_c = g_c(self.pops.pop_c, x, self.params['B_c'])
            a = get_rand_beta(m_a, self.params['phi_a'], r)
            b = get_rand_beta(m_b, self.params['phi_b'], r)
            c = get_rand_gamma(m_c, self.params['phi_c'], r)
            def gen(t):
                ans = None
                while ans == None or ans < 0:
                    ans = r.normalvariate(the_f(t,s,a,b,c), self.params['phi_m'])
                return ans
            ys = pandas.Series(self.times, index=self.times).apply(gen)
            l.append(datum(pid, x, x, x, s, ys))
        return data(l)

    def get_introspection_key(self):
        return '%s_%s_%s_%s' % ('sim_data_truncate_0_f', self.params.get_key(), self.pops.get_key(), self.id_to_x_s_f.get_key())

    def key_f(self, pid_iterator):
        return '%s_%s' % (self.get_key(), pid_iterator.get_key())

    def location_f(self, pid_iterator):
        return '%s/%s/%s' % (global_stuff.data_home, 'sim_data', self.get_key())

    print_handler_f = staticmethod(folder_adapter(write_diffcovs_data))

    read_f = staticmethod(read_diffcovs_data)

class simulated_get_data_above_0_f(possibly_cached):
    """
    re-seed random number generator at every call
    all random calls from here are fed the freshly seeded random.Random()
    """

    def __init__(self, params, pops, id_to_x_s_f, times, seed=0):
        self.params, self.pops, self.id_to_x_s_f, self.times, self.seed = params, pops, id_to_x_s_f, times, seed

    @key
    #@read_from_file
    @save_to_file
    def __call__(self, pid_iterator):
        r = random.Random(self.seed)
        l = []
        for pid in pid_iterator:
            x,s = self.id_to_x_s_f(pid,r)
            assert len(x) == len(self.params['B_a'])
            m_a = g_a(self.pops.pop_a, x, self.params['B_a'])
            m_b = g_b(self.pops.pop_b, x, self.params['B_b'])
            m_c = g_c(self.pops.pop_c, x, self.params['B_c'])
            a = get_rand_beta(m_a, self.params['phi_a'], r)
            b = get_rand_beta(m_b, self.params['phi_b'], r)
            c = get_rand_gamma(m_c, self.params['phi_c'], r)
            def gen(t):
                return max(r.normalvariate(the_f(t,s,a,b,c), self.params['phi_m']),0)
            ys = pandas.Series(self.times, index=self.times).apply(gen)
            l.append(datum(pid, x, x, x, s, ys))
        return data(l)

    def get_introspection_key(self):
        return '%s_%s_%s_%s' % ('sim_data_above_0_f', self.params.get_key(), self.pops.get_key(), self.id_to_x_s_f.get_key())

    def key_f(self, pid_iterator):
        return '%s_%s' % (self.get_key(), pid_iterator.get_key())

    def location_f(self, pid_iterator):
        return '%s/%s/%s' % (global_stuff.data_home, 'sim_data', self.get_key())

    print_handler_f = staticmethod(folder_adapter(write_diffcovs_data))

    read_f = staticmethod(read_diffcovs_data)

class an_id_to_x_s_f(keyed_object):
    """
    generate x data to have 0 mean, unit variance, 
    """

    def get_introspection_key(self):
        return '%s_%s' % ('stdxsgen', self.dim)

    def __init__(self, dim):
        self.dim = dim

    def __call__(self, pid, r=None):
        if r == None:
            x = pandas.Series([random.normalvariate(0,1) for i in xrange(self.dim)])
            s = random.uniform(0,1)
        else:
            x = pandas.Series([r.normalvariate(0,1) for i in xrange(self.dim)])
            s = r.uniform(0,1)
        return x,s

class normalized_data_f(possibly_cached):

    def get_introspection_key(self):
        return 'normf'

    def key_f(self, d):
        return '%s_%s' % (self.get_key(), d.get_key())

    def location_f(self, d):
        return d.get_location()

    @key
    def __call__(self, d):
        def applier(s):
            print s.name, s.name.to_normalize(), 'applier'
            if s.name.to_normalize():
                return (s - pandas.Series.mean(s)) / pandas.Series.std(s)
            else:
                return s
        xas = pandas.DataFrame({p.pid:p.xa for p in d})
        xbs = pandas.DataFrame({p.pid:p.xa for p in d})
        xcs = pandas.DataFrame({p.pid:p.xa for p in d})
        normalized_xas = xas.apply(applier, axis=1)
        normalized_xbs = xbs.apply(applier, axis=1)
        normalized_xcs = xcs.apply(applier, axis=1)
        l = data([datum(_datum.pid, normalized_xas[_datum.pid],normalized_xbs[_datum.pid],normalized_xcs[_datum.pid],_datum.s, _datum.ys) for _datum in d])
        return l
                
class filtered_get_data_f(possibly_cached):

    def get_introspection_key(self):
        return 'filt'

    def key_f(self, data):
        return '%s_%s' % (self.get_key(), data.get_key())

    def location_f(self, data):
        return '%s/%s/%s' % (global_stuff.data_home, 'filtered_data', data.get_key())

    print_handler_f = staticmethod(folder_adapter(write_diffcovs_data))

    read_f = staticmethod(read_diffcovs_data)

    @key
    #@read_from_pickle
    #@save_to_file
    #@save_to_pickle
    def __call__(self, _data):
        def is_ok(datum):
            try:
                if sum([x > (datum.s + 0.05) for x in datum.ys]) >= 2:
                    raise Exception
                a,b,c = get_curve_abc(datum.s, datum.ys)
                error = 0.0
                for t,v in datum.ys.iteritems():
                    fit_val = the_f(t, datum.s, a, b, c)
                    error += abs(fit_val - v)
                if error / sum(datum.ys.notnull()) > 0.065:
                    raise Exception
                if sum(datum.ys.notnull()) < 8:
                    raise Exception
                if datum.s < 0.1:
                    raise Exception
                
            except Exception:
                return False
            else:
                return True
        return data(filter(is_ok, _data))


class medium_filtered_get_data_f(possibly_cached):

    def get_introspection_key(self):
        return 'medfilt'

    def key_f(self, data):
        return '%s_%s' % (self.get_key(), data.get_key())

    def location_f(self, data):
        return '%s/%s/%s' % (global_stuff.data_home, 'filtered_data', data.get_key())

    print_handler_f = staticmethod(folder_adapter(write_diffcovs_data))

    read_f = staticmethod(read_diffcovs_data)

    @key
    @read_from_pickle
    @save_to_file
    @save_to_pickle
    def __call__(self, _data):
        def is_ok(datum):
            try:
                if sum([x > (datum.s + 0.05) for x in datum.ys]) >= 2:
                    raise Exception
                a,b,c = get_curve_abc(datum.s, datum.ys)
                error = 0.0
                for t,v in datum.ys.iteritems():
                    fit_val = the_f(t, datum.s, a, b, c)
                    error += abs(fit_val - v)
                if error / sum(datum.ys.notnull()) > 0.072:
                    raise Exception
                if sum(datum.ys.notnull()) < 6:
                    raise Exception
                if datum.s < 0.1:
                    raise Exception
                
            except Exception:
                return False
            else:
                return True
        return data(filter(is_ok, _data))


            
class old_filtered_get_data_f(possibly_cached):

    def get_introspection_key(self):
        return 'oldfilt'

    def key_f(self, data):
        return '%s_%s' % (self.get_key(), data.get_key())

    def location_f(self, data):
        return '%s/%s/%s' % (global_stuff.data_home, 'filtered_data', data.get_key())

    print_handler_f = staticmethod(folder_adapter(write_diffcovs_data))

    read_f = staticmethod(read_diffcovs_data)

    @key
    #@read_from_pickle
    #@save_to_file
    #@save_to_pickle
    def __call__(self, _data):
        def is_ok(datum):
            try:
                if sum([x > (datum.s + 0.05) for x in datum.ys]) >= 2:
                    raise Exception
                a,b,c = get_curve_abc(datum.s, datum.ys)
                error = 0.0
                for t,v in datum.ys.iteritems():
                    fit_val = the_f(t, datum.s, a, b, c)
                    error += abs(fit_val - v)
                if error / sum(datum.ys.notnull()) > 0.08:
                    #pdb.set_trace()
                    assert sum(datum.ys.notnull()) == len(datum.ys)
                    raise Exception
            except Exception:
                return False
            else:
                return True
        pdb.set_trace()
        ans = data(filter(is_ok, _data))
        return ans




class generic_filtered_get_data_f(possibly_cached):

    def get_introspection_key(self):
        return '%s_%s' % ('filt', self.ys_bool_input_curve_f.get_key())

    def key_f(self, _data):
        return '%s_%s' % (self.get_key(), _data.get_key())

    def location_f(self, data):
        return '%s/%s/%s' % (global_stuff.data_home, 'filtered_data', data.get_key())

    print_handler_f = staticmethod(folder_adapter(write_diffcovs_data))

    read_f = staticmethod(read_diffcovs_data)

    def __init__(self, ys_bool_input_curve_f):
        self.ys_bool_input_curve_f = ys_bool_input_curve_f

    @key
    @read_from_pickle
    @save_to_pickle
    def __call__(self, _data):
        filter_f = compose_expanded_args(self.ys_bool_input_curve_f, lambda _datum: (_datum.s, _datum.ys))
        return data(filter(filter_f, _data))


class get_data_fold_training(possibly_cached):

    def get_introspection_key(self):
        return '%s_%d_%d' % ('trfld', self.fold_i, self.fold_k)

    def key_f(self, data):
        return '%s_%s' % (self.get_key(), data.get_key())

    def location_f(self, data):
        return '%s/%s_%d_%d' % (data.get_location(), 'training', self.fold_i, self.fold_k)

    print_handler_f = staticmethod(folder_adapter(write_diffcovs_data))

    read_f = staticmethod(read_diffcovs_data)

    def __init__(self, fold_i, fold_k):
        self.fold_i, self.fold_k = fold_i, fold_k

    @key
    @save_to_file
    #@read_from_pickle
    @save_to_pickle
    def __call__(self, _data):
        return data([datum for datum,i in zip(_data, range(len(_data))) if i%self.fold_k != self.fold_i])
            


class get_data_fold_testing(possibly_cached):

    def get_introspection_key(self):
        return '%s_%d_%d' % ('tefld', self.fold_i, self.fold_k)

    def key_f(self, data):
        return '%s_%s' % (self.get_key(), data.get_key())

    def location_f(self, data):
        return '%s/%s_%d_%d' % (data.get_location(), 'testing', self.fold_i, self.fold_k)

    print_handler_f = staticmethod(folder_adapter(write_diffcovs_data))

    read_f = staticmethod(read_diffcovs_data)

    def __init__(self, fold_i, fold_k):
        self.fold_i, self.fold_k = fold_i, fold_k
    
    @key
    @save_to_file
    #@read_from_pickle
    @save_to_pickle
    def __call__(self, _data):
        return data([datum for datum,i in zip(_data, range(len(_data))) if i%self.fold_k == self.fold_i])
            

"""
related to x_abc_fs
"""

class get_dataframe_f(possibly_cached):

    def get_introspection_key(self):
        return '%s_%s' % ('df', self.fs.get_key())

    def key_f(self, fs):
        return '%s_%s' % (self.get_key(), fs.get_key())

    def location_f(self, fs):
        return './scratch'

    print_handler_f = staticmethod(write_DataFrame)

    read_f = staticmethod(read_DataFrame)

    def __init__(self, fs):
        self.fs = fs


    @save_and_memoize
    @read_from_file
    def __call__(self, pid_iterator):
        """
        get_dataframe_f __call__
        """
        d = {}
        for pid in pid_iterator:
            try:
                d[pid] = get_feature_series(pid, self.fs)
            except Exception:
                pass
        return keyed_DataFrame(d)


class x_abc_fs(keyed_object):

    def get_introspection_key(self):
        return '%s_%s_%s_%s' % ('abc_f', self.xa_fs.get_key(), self.xb_fs.get_key(), self.xc_fs.get_key())

    def __init__(self, xa_fs, xb_fs, xc_fs):
        self.xa_fs, self.xb_fs, self.xc_fs = xa_fs, xb_fs, xc_fs




"""
related to ys_f, as in getting the series
"""

class score_modifier_f(keyed_object):

    def get_introspection_key(self):
        if self.shift == 0:
            return '%s_%.2f' % ('up', self.c)
        else:
            return '%s_%.2f_s%d' % ('up', self.c, self.shift)

    def __init__(self, c, shift=0):
        self.c, self.shift = c, shift

    def __call__(self, t, v):
        return t-self.shift, ((v+self.c) / (1.0+self.c))

class actual_ys_f(possibly_cached):

    def get_introspection_key(self):
        return '%s_%s_%d' % ('actys', self.ys_f.get_key(), self.shift)

    def __init__(self, ys_f, shift):
        self.ys_f, self.shift = ys_f, shift

    def __call__(self, pid):
        """
        get rid of data at time 0 always, since times in ys_f are not shifted yet
        then shift the remaining times
        """
        raw = self.ys_f(pid)
        d = {}
        for t,v in raw.iteritems():
            if t != 0:
                shifted_time = t - self.shift
                if shifted_time >= 0:
                    d[shifted_time] = v
        return pandas.Series(d)
        

class modified_ys_f(possibly_cached):
    """
    this will return values at all times, not just those > 0
    ERROR: this should actually accept a ys_f so that fxns like this can be composed
    score modifier might also shift the times
    """
    def get_introspection_key(self):
        return '%s_%s_%s' % ('modys', self.ys_f.get_key(), self.score_modifier_f.get_key())

    def __init__(self, ys_f, score_modifier_f):
        self.ys_f, self.score_modifier_f  = ys_f, score_modifier_f

    def __call__(self, pid):
        raw = self.ys_f(pid)
        ans = pandas.Series(dict(self.score_modifier_f(k,v) for k,v in raw.iteritems()))
        return ans


class ys_f(keyed_object):

    physical_condition, mental_condition, urinary_function, urinary_bother, bowel_function, bowel_bother, sexual_function, sexual_bother = 2,3,4,5,6,7,8,9

    function_names = {physical_condition:'physical_condition', mental_condition:'mental_condition', urinary_function:'urinary_function', urinary_bother:'urinary_bother', bowel_function:'bowel_function', bowel_bother:'bowel_bother', sexual_function:'sex', sexual_bother:'sexual_bother'}

    def get_introspection_key(self):
        return '%s_%s' % ('ys', ys_f.function_names[self.which_function])

    def __init__(self, which_function):
        import pandas
        function_name = ys_f.function_names[which_function]
        raw_file = '%s/%s.csv' % (global_stuff.ys_folder, function_name)
        self.which_function = which_function
        self.dump = pandas.read_csv(raw_file, index_col=0, header=0)

    def __call__(self, pid):
        import pdb
        return self.dump[pid].dropna()


class smooth_data_f(possibly_cached):

    def get_introspection_key(self):
        return 'smth'

    def key_f(self, data):
        return '%s_%s' % (self.get_key(), data.get_key())

    def location_f(self, data):
        return '%s/%s' % (global_stuff.data_home, 'smoothed_data')

    @key
    def __call__(self, _data):
        l = []
        import copy
        _get_curve_abc_f = get_curve_abc_f()
        for _datum in _data:
            a,b,c = _get_curve_abc_f(_datum.s, _datum.ys)
            smooth_ys = pandas.Series({t:the_f(t,_datum.s,a,b,c) for t in _datum.ys.index})
            new_datum = copy.copy(_datum)
            new_datum.ys = smooth_ys
            l.append(new_datum)
        return data(l)


"""
related to hypers
"""
class hypers(keyed_object):
    """
    key is hard coded
    """
    def get_introspection_key(self):
        return 'hyps'

    def __init__(self, c_a, c_b, c_c, l_a, l_b, l_c, l_m):
        self.c_a, self.c_b, self.c_c, self.l_a, self.l_b, self.l_c, self.l_m = c_a, c_b, c_c, l_a, l_b, l_c, l_m


class hypers_save_f(save_factory_base):
    """
    only purpose is to save hypers
    """
        
    def location_f(self, item):
        return '%s/%s' % (global_stuff.data_home, 'hypers')

    print_handler_f = staticmethod(string_adapter(hypers_print_f))

    read_f = staticmethod(hypers_read_f)

"""
related to plotting
"""

class figs_with_avg_error_f(possibly_cached):
    """
    input is pid_iterator.  output is a bunch of figures with total error, avg error printed
    also draw the fitted curve
    """

    def get_introspection_key(self):
        return '%s_%s' % (self.s_f.get_key(), self.actual_ys_f.get_key())

    def key_f(self, pid_iterator):
        return '%s_%s' % (self.get_key(), pid_iterator.get_key())

    def location_f(self, pid_iterator):
        return '%s/%s' % (global_stuff.data_home, 'ys_plots')

    print_handler_f = staticmethod(multiple_figures_to_pdf)

    def __init__(self, s_f, actual_ys_f):
        self.s_f, self.actual_ys_f = s_f, actual_ys_f

#    @memoize
    @save_to_file
    def __call__(self, pid_iterator):
        figs = keyed_list()
        for pid in pid_iterator:
            fig = plt.figure()
            ax = fig.add_subplot(1,1,1)
            try:
                s = self.s_f(pid)
                ys = self.actual_ys_f(pid)
            except Exception, e:
                pass
            else:
                add_series_to_ax(ys, ax, 'black', None, '--')
                ax.plot(-1, s, 'bo')
                ax.set_xlim(-2,50)
                a,b,c = get_curve_abc(s, ys)
                fit_f = functools.partial(the_f, s=s, a=a, b=b, c=c)
                t_vals = range(50)
                y_vals = [fit_f(t) for t in t_vals]
                ax.plot(t_vals, y_vals, color='brown', linestyle='-')
                fitted = pandas.Series(ys.index,index=ys.index).apply(fit_f)
                error = (fitted - ys).abs().sum()
                ax.set_title('%s %.2f %.2f'  % (pid, error, error/len(ys)))
                ax.set_ylim(-0.1,1.1)
                figs.append(fig)
        return figs
    

class abc_vs_attributes_scatter_f(possibly_cached):

    def get_introspection_key(self):
        return '%s_%s_%s_%s' % ('abcplts', self.fs.get_key(), self.s_f.get_key(), self.actual_ys_f.get_key())

    def key_f(self, pid_iterator):
        return '%s_%s' % (self.get_key(), pid_iterator.get_key())

    def location_f(self, pid_iterator):
        return '%s/%s' % (global_stuff.data_home, 'abc_vs_attribute_scatters')

    print_handler_f = staticmethod(multiple_figures_to_pdf)

    def __init__(self, fs, s_f, actual_ys_f):
        self.fs, self.s_f, self.actual_ys_f = fs, s_f, actual_ys_f

    @memoize
    @save_to_file
    def __call__(self, pid_iterator):
        """
        for each feature, get a matrix whose columns are abc's + the feature.  do this by joining 2 columns
        filtering of pids is done outside(pid_iterator may be filtered) 
        """
        # first get the abc's
        
        abc_d = {}
        for pid in pid_iterator:
            s = self.s_f(pid)
            ys = self.actual_ys_f(pid)
            a,b,c = get_curve_abc(s, ys)
            abc_d[pid] = pandas.Series({'a':a, 'b':b,'c':c})
        abc_df = pandas.DataFrame(abc_d)

        # now, get a feature for each dataframe for joining

        figs = []

        for f in self.fs:
            f_series_d = {}
            for pid in pid_iterator:
                try:
                    val = f(pid)
                except my_Exception:
                    pass
                else:
                    f_series_d[pid] = val
            f_series = pandas.Series(f_series_d)
            f_series_df = pandas.DataFrame({f:f_series})
            abcf_df = pandas.concat([abc_df, f_series_df.T], axis = 0, join='inner')
            fig = plt.figure()

            fig.suptitle('%s %s' % (pid_iterator.get_key(), f.get_key()))
            a_ax = fig.add_subplot(2,2,1)
            a_ax.scatter(abcf_df.loc[f],abcf_df.loc['a'])
            a_ax.set_ylabel('a')
            a_ax.set_xlabel(f.get_key())
            b_ax = fig.add_subplot(2,2,2)
            b_ax.scatter(abcf_df.loc[f],abcf_df.loc['b'])
            b_ax.set_ylabel('b')
            b_ax.set_xlabel(f.get_key())
            c_ax = fig.add_subplot(2,2,3)
            c_ax.scatter(abcf_df.loc[f],abcf_df.loc['c'])
            c_ax.set_ylabel('c')
            c_ax.set_xlabel(f.get_key())
            fig.subplots_adjust(hspace=0.3,wspace=0.3)
            figs.append(fig)

        return figs


class aggregate_curve_f(possibly_cached):

    def get_introspection_key(self):
        return 'agg'

    def key_f(self, data):
        return '%s_%s' % (self.get_key(), data.get_key())

    def location_f(self, data):
        return '%s/%s' % (global_stuff.data_home, 'aggregate_curves')

    print_handler_f = staticmethod(write_Series)

    read_f = staticmethod(read_Series)

    @key
    def __call__(self, data):
        all_ys = pandas.DataFrame({datum.pid:datum.ys for datum in data})
        mean = all_ys.apply(np.mean, axis=1)
        return keyed_Series(mean)


class aggregate_shape_f(possibly_cached):

    def get_introspection_key(self):
        return 'aggshape'

    def key_f(self, data):
        return '%s_%s' % (self.get_key(), data.get_key())

    def location_f(self, data):
        return '%s/%s' % (global_stuff.data_home, 'aggregate_shapes')

    print_handler_f = staticmethod(write_Series)

    read_f = staticmethod(read_Series)

    @key
    def __call__(self, data):
        all_ys = pandas.DataFrame({datum.pid:datum.ys/datum.s for datum in data})
        mean_shape = all_ys.apply(np.mean, axis=1)
        return keyed_Series(mean_shape)


class figure_combiner_f(possibly_cached):
    """
    accepts iterator, fig_creator_f, parsing function that takes output of iterator, figures out what is 'data' and what is 'how'
    parsing function returns 2 lists for use as *args.  
    """
    def get_introspection_key(self):
        return self.base_fig_creator.get_key()

    def key_f(self, iterator):
        return iterator.get_key()
        return '%s_%s' % (self.get_key(), iterator.get_key())

    def location_f(self, iterator):
        return '%s/%s' % (global_stuff.data_home, 'multiple_figs')

    print_handler_f = staticmethod(multiple_figures_to_pdf)

    def __init__(self, base_fig_creator, parsing_f):
        self.base_fig_creator, self.parsing_f = base_fig_creator, parsing_f

    def __call__(self, iterator):
        figs = []
        for stuff in iterator:
            how, which = self.parsing_f(stuff)
            figs.append(self.base_fig_creator(*how)(*which))
        return figs


class always_true_f(keyed_object):

    def get_introspection_key(self):
        return 'truef'

    def __call__(self, *args, **kwargs):
        return True

"""
helpers
"""

def add_performances_to_ax(ax, perfs, color, name):
    add_series_to_ax(perfs['mean'], ax, color, name, 'dashed')
    percentiles = perfs[[x for x in perfs.columns if x != 'mean']]
    fixed = functools.partial(add_series_to_ax, ax=ax, color=color, label=None, linestyle='solid')
    percentiles.apply(fixed, axis=0)
    return ax

def add_series_to_ax(s, ax, color, label, linestyle):
    ax.plot(s.index, s, color=color, ls=linestyle, label=label)

def the_f(t, s, a, b, c):
    return s * ( (1.0-a) - (1.0-a)*(b) * math.exp(-1.0*t/c))

def g_a(pop_a, xa, B_a):
    return logistic(logit(pop_a) + xa.dot(B_a))
                 
def g_b(pop_b, xb, B_b):
    return logistic(logit(pop_b) + xb.dot(B_b))

def g_c(pop_c, xc, B_c):
    import math
    return math.exp(math.log(pop_c) + xc.dot(B_c))


class get_curve_abc_f(possibly_cached):

    def key_f(self, s, curve):
        return '%.2f_%s' % (s, curve.tostring())

    @memoize
    def __call__(self, s, curve):
        return get_curve_abc_helper(s, curve)

def get_curve_abc_helper(s, curve):
    import math
    import scipy.optimize
    def obj_f(x):
        error = 0.0
        for time, value in curve.iteritems():
            if not np.isnan(value):
                error += pow(the_f(time, s, x[0], x[1], x[2]) - value, 2)
        return error
    x, f, d = scipy.optimize.fmin_l_bfgs_b(obj_f, np.array([0.5, 0.5, 2.0]), approx_grad = True, bounds = [(0.00,1.0),[0.00,1.0],[0.01,None]])
    return x


get_curve_abc = get_curve_abc_f()

class hard_coded_f(object):

    def __init__(self, val):
        self.val = val

    def __call__(self, *args, **kwargs):
        return self.val


def get_categorical_fs(backing_f, bins):
    """
    given list of bins, backing feature, returns list of bin features
    """
    return keyed_list([bin_f(backing_f, bin) for bin in bins])

def get_percentiles(l, percentiles):
    s_l = sorted(l.dropna())
    num = len(s_l)
    return pandas.Series([s_l[int((p*num)+1)-1] for p in percentiles],index=percentiles)

def logit(x):
    import math
    return math.log(x/(1.0-x))

def logistic(x):
    return 1.0 / (1 + np.exp(-x))

def scaled_logistic_loss(x, c):

    return 2*(logistic(c*x)-0.5)



def train_logistic_model(X, Y):
    """
    each patient is a column.  would like return object to be series whose index matches feature names
    """
    def obj_f(b):
        error_vect = (X.T.dot(b)).apply(logistic) - Y
        return error_vect.dot(error_vect)
    import scipy.optimize

    ans, f, d = scipy.optimize.fmin_l_bfgs_b(obj_f, np.zeros(X.shape[0]), approx_grad = True)
    return pandas.Series(ans, index=X.index)

def train_logistic_shapeunif_model(X, Y, S):
    """
    each patient is a column.  would like return object to be series whose index matches feature names
    """
    S_mat = np.diag(S)
    def obj_f(b):
        error_vect = S_mat.dot((X.T.dot(b)).apply(logistic)) - Y
        return error_vect.dot(error_vect)
    import scipy.optimize

    ans, f, d = scipy.optimize.fmin_l_bfgs_b(obj_f, np.zeros(X.shape[0]), approx_grad = True)
    return pandas.Series(ans, index=X.index)

def get_feature_series(pid, fs):
    return pandas.Series({f:f(pid) for f in fs})    


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

    def __contains__(self, obj):
        if obj in self.in_vals:
            return True
        else:
            return False

    def __repr__(self):
        import string
        return string.join([str(v) for v in self.in_vals], sep='_')


class pp_roll(object):
    """
    object to which you can keep requesting axes, and it fits multiple per page
    it keeps a list of figures you can access
    """

    def __init__(self, num_rows, num_cols, hspace=0.3, wspace=0.3):
        self.num_rows, self.num_cols, self.hspace, self.wspace = num_rows, num_cols, hspace, wspace
        self.figure_limit = num_rows * num_cols
        self.cur_fig_num = -1
        self.figs = []
        self.start_new_page()

    def start_new_page(self):
        if self.cur_fig_num != 0:
            self.figs.append(plt.figure())
            self.figs[-1].subplots_adjust(hspace=self.hspace,wspace=self.wspace)
            self.cur_fig_num = 0

    def get_axes(self):
        if self.cur_fig_num >= self.figure_limit:
            self.start_new_page()
        ax = self.figs[-1].add_subplot(self.num_rows, self.num_cols, self.cur_fig_num+1)
        self.cur_fig_num = self.cur_fig_num + 1
        return ax

def print_traceback():
    import traceback, sys
    for frame in traceback.extract_tb(sys.exc_info()[2]):
        fname,lineno,fn,text = frame
        print "Error in %s on line %d" % (fname, lineno)            

class my_iter_apply(object):

    def __init__(self, f, base_iter):
        self.base_iter, self.f = base_iter, f

    def __iter__(self):
        for x in self.base_iter:
            yield self.f(*x)

def get_feature_set_iterator(*args):
    """
    given list of iterators, each of which iterate over keyed_lists of features, gives iterator over flattened cross_product of those iterators, with the key of returned feature list set to the concatenation of individual keys.  adds offset_feature to feature set
    """
    def f(inner_args):
        return set_hard_coded_key_dec(keyed_list, '_'.join(itertools.imap(lambda x:x.get_key(), inner_args)))(itertools.chain(*inner_args))
    return itertools.imap(f, itertools.product(*args))

def get_gapped_iterable(the_iter, k, n):
    return itertools.imap(lambda x: x[0], itertools.ifilter(lambda x:x[1]%n==k, zip(the_iter, itertools.count(0))))

def run_iter_f_parallel_dec(f, job_n):
    """
    for a function that accepts only a single iterable, runs it in parallel.
    """
    def dec_f(the_iterable):
        jobs = []
        for job_i in xrange(job_n):
            the_iterable_this_job = get_gapped_iterable(the_iterable, job_i, job_n)
            p = multiprocessing.Process(target=f, args=(the_iterable_this_job,))
            jobs.append(p)
            p.start()
        [p.join() for p in jobs]
    
    return dec_f

def randomize_iterable_dec(f):
    """
    for a function that accepts only a single iterable, changes the iterable to a randomized list of the elements returned by iterable
    """
    def dec_f(the_iterable):
        vals = [x for x in the_iterable]
        import random
        random.shuffle(vals)
        return f(vals)
    return dec_f

def override_sysout_dec(f, log_folder):
    """
    if calling a function as a multiprocessing process, want to override sysout so that outputs goes to a process-specific log file
    """
    def dec_f(*args, **kwargs):
        import sys, os
        stdout_log_file = '%s/%s_%s.stdout_log' % (log_folder, f.__name__, os.getpid())
        sys.stdout = open(stdout_log_file, 'w')
        stderr_log_file = '%s/%s_%s.stderr_log' % (log_folder, f.__name__, os.getpid())
        sys.stderr = open(stderr_log_file, 'w')
        return f(*args, **kwargs)

    return dec_f

def get_seq(start, interval, num):
    return [start + i*interval for i in xrange(num)]

def get_rand_beta(m, phi, r=None):
    s = (1.0/phi) - 1
    alpha = 1.0 + s*m
    beta = 1.0 + s*(1-m)
    if r == None:
        return random.betavariate(alpha,beta)
    else:
        return r.betavariate(alpha, beta)

def get_beta_p(x, m, phi):
    s = (1.0/phi) - 1
    alpha = 1.0 + s*m
    beta = 1.0 + s*(1-m)
    import math, scipy.special
    #return scipy.stats.beta.pdf(x,alpha,beta)
    c = 1.0 / scipy.special.beta(alpha,beta)
    #c = math.gamma(alpha+beta)/(math.gamma(alpha)*(math.gamma(beta)))
    return c * math.pow(x,alpha-1) * math.pow(1-x,beta-1)

def get_rand_gamma(m, phi, r):
    alpha = 1.0/phi
    beta = (alpha-1.0)/m
    if r == None:
        return random.gammavariate(alpha,1.0/beta)
    else:
        return r.gammavariate(alpha,1.0/beta)

def get_gamma_p(x, m, phi):
    import math
    alpha = 1.0/phi
    beta = (alpha-1.0)/m
    c = pow(beta,alpha) / math.gamma(alpha)
    return c * math.pow(x,alpha-1) * math.exp(-beta*x)

class returns_whats_given_f(possibly_cached):

    def get_introspection_key(self):
        return 'nothing'

    def __init__(self, x):
        self.x = x

    def __call__(self, *args, **kwargs):
        return self.x





#def plot_curve_distribution(ax, s, xa, xb, xc):
    
