

data_home = './scratch'
train_diffcovs_r_script = './train_diffcovs_model.r'
xs_file = '../raw_data/xs.csv'
ys_folder = '../raw_data/series'
all_pid_file = '../raw_data/pids.csv'
times = [1,2,4,8,12,18,24,30,36,42,48]
percentiles = [.25, .50, .75]


from management_stuff import *
import pandas
import numpy as np
import functools
import math

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
    pids = pd.read_csv(pids_file, header=None, squeeze=True, converters={0:str})
    xas = pd.read_csv(xbs_file, header=0, index_col=0)
    xbs = pd.read_csv(xbs_file, header=0, index_col=0)
    xcs = pd.read_csv(xcs_file, header=0, index_col=0)
    ss = pd.read_csv(ss_file, header=None, index_col=0, squeeze=True)
    ys_folder = '%s/%s' % (folder, 'datapoints')
    l = []
    for pid, xa, xb, xc, s in zip(pids, xas.iteritems(), xbs.iteritems(), xcs.iteritems(), ss):
        p_ys_file = '%s/%s' % (ys_folder, pid)
        p_ys = pd.read_csv(p_ys_file,header=None, index_col = 0, squeeze=True)
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
    traces['B_a'].to_csv(folder+'/out_B_a.csv', header=False, index=False)
    traces['B_b'].to_csv(folder+'/out_B_b.csv', header=False, index=False)
    traces['B_c'].to_csv(folder+'/out_B_c.csv', header=False, index=False)
    traces['phi_a'].to_csv(folder+'/out_phi_a.csv', header=False, index=False)
    traces['phi_b'].to_csv(folder+'/out_phi_b.csv', header=False, index=False)
    traces['phi_c'].to_csv(folder+'/out_phi_c.csv', header=False, index=False)
    traces['phi_m'].to_csv(folder+'/out_phi_m.csv', header=False, index=False)

def hypers_print_f(h):
    return '%.2f,%.2f,%.2f,%.2f,%.2f,%.2f,%.2f' % (h.c_a,h.c_b,h.c_c,h.l_a,h.l_b,h.l_c,h.l_m)

def write_DataFrame(df, full_path):
    df.to_csv(full_path, header=True, index=True)

def write_Series(s, full_path):
    s.to_csv(full_path, header=True, index=True)

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

"""
id_iterators
"""

class all_ucla_pid_iterator(keyed_object):

    def get_introspection_key(self):
        return 'all_ucla'

    def __init__(self):
        self.pids = pandas.read_csv(all_pid_file, header=False, index_col=None, squeeze=True, converters={0:str}).tolist()

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
                    raise Exception
            except Exception:
                pass
            else:
                yield pid

"""
features
"""

class feat(keyed_object):

    def __repr__(self):
        return self.get_key()

    def __hash__(self):
        return self.get_key()

    def __cmp__(self, other):
        try:
            other_key = other.get_key()
        except AttributeError:
            other_key = str(other)
        return self.get_key() == other_key

class ones_f(keyed_object):

    def get_introspection_key(self):
        return 'ones_f'

    def __call__(self, pid):
        return 1

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
        self.all_covs = pandas.read_csv(xs_file, index_col=0)

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



class pops(keyed_object):

    def __repr__(self):
        return '%.2f, %.2f, %.2f' % (self.pop_a, self.pop_b, self.pop_c)

    def __init__(self, pop_a, pop_b, pop_c):
        self.pop_a, self.pop_b, self.pop_c = pop_a, pop_b, pop_c

    @staticmethod
    def print_f(x):
        return '%.4f, %.4f, %.f4' % (x.pop_a, x.pop_b, x.pop_c)

    @staticmethod
    def read_f(full_path):
        f = open(full_path,'r')
        raw = f.readline().strip().split(',')
        return pops(float(raw[0]), float(raw[1]), float(raw[2]))


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

    to_recalculate = False

    @call_and_cache
    @call_and_save
    def __call__(self, data):
        """
        averages curves, and then fits a curve to it
        """
        
        def obj_f(x):
            a,b,c = x[0],x[1],x[2]
            error = 0.0
            for datum in data:
                fit_f = functools.partial(the_f, s=datum.s, a=a, b=b, c=c)
                fitted = pandas.Series(datum.ys.index).apply(fit_f)
                diff_vect = (fitted - datum.ys).dropna()
                this = diff_vect.dot(diff_vect)
                error += this
                """
                for t,v in datum.ys.iteritems():
                    fitted_val = the_f(t,datum.s,a,b,c)
                    error = error + pow(fitted_val - v, 2)
                    if np.isnan(error):
                        pdb.set_trace()
                    print pow(fitted_val - v, 2), error
                """
            print error
            return error

        import scipy.optimize
        x, f, d = scipy.optimize.fmin_l_bfgs_b(obj_f, np.array([0.5, 0.5, 2.0]), approx_grad = True, bounds = [(0.01,.99),[0.01,.99],[0.01,None]])
        return pops(x[0],x[1],x[2])


class prior_predictor(keyed_object):

    def get_introspection_key(self):
        return '%s_%s' % 'priorpred', self.pops.get_key()

    def __init__(self, pops):
        self.pops = pops

    def __call__(self, datum, time):
        return the_f(time, datum.s, self.pops.pop_a, self.pops.pop_b, self.pops.pop_c)


class get_prior_predictor_f(possibly_cached):

    display_color = 'red'

    display_name = 'prior'

    def get_introspection_key(self):
        return '%s_%s' % ('priorpred-f', self.get_pops_f.get_key())

    def key_f(self, data):
        return '%s_%s' % (self.get_key(), data.get_key())

    def __init__(self, get_pops_f):
        self.get_pops_f = get_pops_f

    def __call__(self, data):
        pops = self.get_pops_f(data)
        return prior_predictor(pops)



class get_diffcovs_posterior_f(possibly_cached):
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

    to_recalculate = True

    def __init__(self, get_pops_f, hypers, iters, chains, seed):
        self.r_script = train_diffcovs_r_script
        self.get_pops_f, self.hypers, self.iters, self.chains, self.seed = get_pops_f, hypers, iters, chains, seed

    @call_and_save
    def __call__(self, data):
        pops = self.get_pops_f.call_and_save(data)
        pops_path = self.get_pops_f.full_path_f(data)
        data.get_creator().save(data)
        data_path = data.get_full_path()
        hypers_save_f()(self.hypers)
        hypers_path = hypers_save_f().full_path_f(self.hypers)
        save_path = self.full_path_f(data)
        make_folder(save_path)
        import subprocess
        cmd = '%s %s %s %s %s %d %d %d %s' % ('Rscript', self.r_script, pops_path, data_path, hypers_path, self.iters, self.chains, self.seed, save_path)
        print cmd
        subprocess.call(cmd, shell=True)
        posteriors = read_unheadered_posterior_traces(save_path)
        # set the column names of posterior traces
        a_datum = iter(data).next()
        posteriors['B_a'].columns = a_datum.xa.index
        posteriors['B_b'].columns = a_datum.xb.index
        posteriors['B_c'].columns = a_datum.xc.index
        posteriors['phi_a'].name = 'phi_a'
        posteriors['phi_b'].name = 'phi_b'
        posteriors['phi_c'].name = 'phi_c'
        posteriors['phi_m'].name = 'phi_m'
        return posteriors

class full_model_point_predictor(keyed_object):
    """
    given point estimates of posterior params, does prediction
    """
    def get_introspection_key(self):
        return '%s_%s_%s' % ('full_pred', self.params.get_key(), self.pops.get_key())

    def __init__(self, params, pops):
        self.params, self.pops = params, pops

    def __call__(self, datum, time):
        a = g_a(self.pops.pop_a, self.params['B_a'], datum.xa)
        b = g_a(self.pops.pop_b, self.params['B_b'], datum.xb)
        c = g_a(self.pops.pop_c, self.params['B_c'], datum.xc)
        return the_f(time, datum.s, a, b, c)


class get_param_mean_f(possibly_cached):
    """
    returns dictionary of mean params
    """
    def get_introspection_key(self):
        return 'param_mean'

    def key_f(self, post_param):
        return '%s_%s' % (self.get_key(), post_param.get_key())

    def __call__(self, post_params):
        return keyed_dict({p:v.apply(pandas.Series.mean, axis=0) for p,v in post_params.iteritems()})
        

class get_diffcovs_point_predictor_f(possibly_cached):
    """
    return trained object that makes point predictions
    """
    display_color = 'blue'

    display_name = 'full'

    def get_introspection_key(self):
        return '%s_%s_%s' % ('fullpred_f', self.get_diffcovs_posterior_f.get_key(), self.summarize_posterior_f.get_key())

    def key_f(self, data):
        return '%s_%s' % (self.get_key(), data.get_key())

    def __init__(self, get_diffcovs_posterior_f, summarize_posterior_f):
        self.get_diffcovs_posterior_f, self.summarize_posterior_f = get_diffcovs_posterior_f, summarize_posterior_f

    def __call__(self, data):
        posteriors = self.get_diffcovs_posterior_f(data)
        params = self.summarize_posterior_f(posteriors)
        # assuming that get_diffcovs_posterior_f has a get_pops_f attribute i can call
        pops = self.get_diffcovs_posterior_f.get_pops_f(data)
        return full_model_point_predictor(params, pops)

class logreg_predictor(keyed_object):
    """
    
    """

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
    display_color = 'blue'

    display_name = 'logreg'

    def get_introspection_key(self):
        return 'logpred_f'

    def key_f(self, data):
        return '%s_%s' % (self.get_key(), data.get_key())

    def __init__(self, times):
        self.times = times

    def __call__(self, _data):
        """
        for now, use only xa for prediction
        """
        Bs = {}
        xas = pandas.DataFrame({datum.pid:datum.xa for datum in _data})
        for time in self.times:
            y = pandas.Series({datum.pid:datum.ys[time] for datum in _data})
            y = y[y.notnull()]
            this_xas = xas[y.index]
            Bs[time] = train_logistic_model(this_xas, y)
        return logreg_predictor(pandas.DataFrame(Bs))



class cross_validated_scores_f(possibly_cached):
    """
    get_predictor_f is a factory for trained predictor objects, as in it does training
    return dataframe where columns are patients and rows are times
    """
    def get_introspection_key(self):
        return '%s_%s_%d' % ('cvscore', self.get_predictor_f.get_key(), self.fold_k)

    def key_f(self, data):
        return '%s_%s' % (self.get_key(), data.get_key())

    def location_f(self, data):
        return '%s/%s' % (data_home, 'scores')

    print_handler_f = staticmethod(write_DataFrame)

    read_f = staticmethod(read_DataFrame)

    to_recalculate = True

    def __init__(self, get_predictor_f, fold_k, times):
        self.get_predictor_f, self.fold_k, self.times = get_predictor_f, fold_k, times


    @call_and_cache
    def __call__(self, data):
        fold_scores = []
        for i in range(self.fold_k):
            train_data = get_data_fold_training(i, self.fold_k)(data)
            test_data = get_data_fold_testing(i, self.fold_k)(data)
            predictor = self.get_predictor_f(train_data)
            fold_scores.append(pandas.DataFrame({datum.pid:{time:predictor(datum, time) for time in self.times} for datum in test_data}))
        return keyed_DataFrame(pandas.concat(fold_scores, axis=1))

class scaled_logistic_loss_f(keyed_object):

    def get_introspection_key(self):
        return '%s_%.2f' % ('logloss', self.c)

    def __init__(self, c):
        self.c = c

    def __call__(self, x):
        return scaled_logistic_loss(x, self.c)

class performance_series_f(possibly_cached):
    """
    returns a dataframe.  mean score will be the first column
    """

    def __init__(self, scores_getter_f, loss_f, percentiles):
        self.scores_getter_f, self.loss_f, self.percentiles = scores_getter_f, loss_f, percentiles

    def __call__(self, data):
        # have raw scores for each patient
        scores = self.scores_getter_f(data)
        true_ys = pandas.DataFrame({datum.pid:datum.ys for datum in data})
        diff = (scores - true_ys).abs()
        losses = diff.apply(self.loss_f, axis=1)
        mean_losses = losses.apply(np.mean, axis=1)
        loss_percentiles = losses.apply(functools.partial(get_percentiles, percentiles=self.percentiles), axis=1)
        loss_percentiles['mean'] = mean_losses
        return keyed_DataFrame(loss_percentiles)
        

    def get_introspection_key(self):
        return '%s_%s_%s' % ('perf', self.scores_getter_f.get_key(), self.loss_f.get_key())

    def key_f(self, data):
        return '%s_%s' % self.get_key(), data.get_key()

    def location_f(self, data):
        '%s/%s' % (data.get_location(), tester_f.__class__.__name__)

    print_handler_f = staticmethod(write_DataFrame)

    read_f = staticmethod(read_DataFrame)

    to_recalculate = False

"""
what 'data' means for this project
"""

class datum(keyed_object):

    def __init__(self, pid, xa, xb, xc, s, ys):
        self.pid, self.xa, self.xb, self.xc, self.s, self.ys = pid, xa, xb, xc, s, ys


class data(keyed_list):

    def get_introspection_key(self):
        return 'data'

class s_f(keyed_object):

    def get_introspection_key(self):
        return '%s_%s' % ('s', self.ys_f.get_key())

    def __init__(self, ys_f):
        self.ys_f = ys_f

    @raise_if_na
    def __call__(self, pid):
        return self.ys_f(pid)[0]

class get_data_f(possibly_cached):

    def __init__(self, x_abc_fs, s_f, ys_f):
        self.x_abc_fs, self.s_f, self.ys_f = x_abc_fs, s_f, ys_f

    def get_introspection_key(self):
        return '%s_%s_%s_%s' % ('data', self.x_abc_fs.get_key(), self.s_f.get_key(), self.ys_f.get_key())

    def key_f(self, pid_iterator):
        return '%s_%s' % (self.get_key(), pid_iterator.get_key())

    def location_f(self, pid_iterator):
        return '%s/%s/%s' % (data_home, 'data', self.get_key())

    print_handler_f = staticmethod(folder_adapter(write_diffcovs_data))

    read_f = staticmethod(read_diffcovs_data)

    to_recalculate = False

    @call_and_cache
    @call_and_save
    def __call__(self, pid_iterator):

        l = []

        for pid in pid_iterator:
            try:
                xa = get_feature_series(pid, self.x_abc_fs.xa_fs)
                xb = get_feature_series(pid, self.x_abc_fs.xb_fs)
                xc = get_feature_series(pid, self.x_abc_fs.xc_fs)
                s = self.s_f(pid)
                ys = self.ys_f(pid)
            except Exception, e:
                print e
                pass
            else:
                l.append(datum(pid, xa, xb, xc, s, ys))
        l = sorted(l, key = lambda x: x.pid)
        return data(keyed_list(l))
                
class filtered_get_data_f(keyed_object):

    def get_introspection_key(self):
        return 'filt'

    def key_f(self, data):
        return '%s_%s' % (self.get_key(), data.get_key())

    def location_f(self, data):
        return '%s/%s/%s' % (data_home, 'filtered_data', data.get_key())

    print_handler_f = staticmethod(folder_adapter(write_diffcovs_data))

    read_f = staticmethod(read_diffcovs_data)

    to_recalculate = False

    @call_and_cache
    @call_and_save
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
                    raise Exception
                if sum(datum.ys.notnull()) < 8:
                    raise Exception
                
            except Exception:
                return False
            else:
                return True
        return data(filter(is_ok, _data))

            


class get_data_fold_training(possibly_cached):

    def get_introspection_key(self):
        return '%s_%d_%d' % ('trfld', self.fold_i, self.fold_k)

    def key_f(self, data):
        return '%s_%s' % (self.get_key(), data.get_key())

    def location_f(self, data):
        return '%s/%s_%d_%d' % (data.get_location(), 'training', self.fold_i, self.fold_k)

    print_handler_f = staticmethod(folder_adapter(write_diffcovs_data))

    read_f = staticmethod(read_diffcovs_data)

    to_recalculate = False

    def __init__(self, fold_i, fold_k):
        self.fold_i, self.fold_k = fold_i, fold_k

    @call_and_cache
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

    to_recalculate = False

    def __init__(self, fold_i, fold_k):
        self.fold_i, self.fold_k = fold_i, fold_k

    @call_and_cache
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

    to_recalculate = True

    def __init__(self, fs):
        self.fs = fs

    @call_and_cache
    @call_and_cache
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
        return '%s_%.2f' % ('up', self.c)

    def __init__(self, c):
        self.c = c

    def __call__(self, s):
        return (s+self.c) / (1.0+self.c)


class modified_ys_f(possibly_cached):

    def get_introspection_key(self):
        return '%s_%s_%s' % ('modys', self.ys_f.get_key(), self.score_modifier_f.get_key())

    def __init__(self, ys_f, score_modifier_f):
        self.ys_f, self.score_modifier_f = ys_f, score_modifier_f

    def __call__(self, pid):
        raw = self.ys_f(pid)
        scaled = raw.apply(self.score_modifier_f)
        ans = scaled[scaled.index != 0]
        return pandas.Series({k:v for k,v in ans.iteritems()})
        return ans

class ys_f(keyed_object):

    physical_condition, mental_condition, urinary_function, urinary_bother, bowel_function, bowel_bother, sexual_function, sexual_bother = 2,3,4,5,6,7,8,9

    function_names = {physical_condition:'physical_condition', mental_condition:'mental_condition', urinary_function:'urinary_function', urinary_bother:'urinary_bother', bowel_function:'bowel_function', bowel_bother:'bowel_bother', sexual_function:'sex', sexual_bother:'sexual_bother'}

    def get_introspection_key(self):
        return '%s_%s' % ('ys', ys_f.function_names[self.which_function])

    def __init__(self, which_function):
        import pandas
        function_name = ys_f.function_names[which_function]
        raw_file = '%s/%s.csv' % (ys_folder, function_name)
        self.which_function = which_function
        self.dump = pandas.read_csv(raw_file, index_col=0, header=0)

    def __call__(self, pid):
        import pdb
        return self.dump[pid]


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
        return '%s/%s' % (data_home, 'hypers')

    print_handler_f = staticmethod(string_adapter(hypers_print_f))

    read_f = staticmethod(hypers_read_f)

    to_recalculate = True

"""
related to plotting
"""
class aggregate_curve_f(possibly_cached):

    def get_introspection_key(self):
        return 'agg'

    def key_f(self, data):
        return '%s_%s' % (self.get_key(), data.get_key())

    def location_f(self, data):
        return '%s/%s' % (data_home, 'aggregate_curves')

    print_handler_f = staticmethod(write_Series)

    read_f = staticmethod(read_Series)

    def __call__(self, data):
        all_ys = pandas.DataFrame({datum.pid:datum.ys for datum in data})
        mean = all_ys.apply(np.mean, axis=1)
        return keyed_Series(mean)




"""
helpers
"""

def add_performances_to_ax(ax, perfs, color, name):
    add_series_to_ax(perfs['mean'], ax, color, name, 'dashed')
    percentiles = perfs[[x for x in perfs.columns if x != 'mean']]
    fixed = functools.partial(add_series_to_ax, ax=ax, color=color, label=name, linestyle='solid')
    percentiles.apply(fixed, axis=0)
    return ax

def add_series_to_ax(s, ax, color, label, linestyle):
    ax.plot(s.index, s, color=color, ls=linestyle, label=label)

def the_f(t, s, a, b, c):
    return s * ( (1.0-a) - (1.0-a)*(b) * math.exp(-1.0*t/c))

def g_a(pop_a, B_a, xa):
    return logistic(pop_a + xa.dot(B_a))
                 
def g_b(pop_b, B_b, xb):
    return logistic(pop_b + xb.dot(B_b))

def g_c(pop_c, B_c, xc):
    return logistic(pop_c + xc.dot(B_c))

def get_curve_abc(s, curve):
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

class hard_coded_f(object):

    def __init__(self, val):
        self.val = val

    def __call__(self, *args, **kwargs):
        return self.val


def get_percentiles(l, percentiles):
    s_l = sorted(l.dropna())
    num = len(l)
    return pandas.Series([s_l[int((p*num)+1)-1] for p in percentiles],index=percentiles)

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

def get_feature_series(pid, fs):
    return pandas.Series({f.get_key():f(pid) for f in fs})    


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


