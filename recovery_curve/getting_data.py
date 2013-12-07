from management_stuff import *
from prostate_specifics import *

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
        try:
            raw = self.backing_f(pid)
        except:
            return 0
        return int(raw in self.bin)



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



class reduce_data_f(possibly_cached):

    def __init__(self, proportion):
        self.proportion = proportion

    @key
    def __call__(self, _data):
        to_keep = data([_data[int(i/float(self.proportion))] for i in range(int(self.proportion*len(_data)))])
        return to_keep
        
    def get_introspection_key(self):
        return '%s_%.2f' % ('reduced', self.proportion)

    def key_f(self, _data):
        return '%s_%s' % (self.get_key(), _data.get_key())

    def location_f(self, pid_iterator):
        return '%s/%s/%s' % (global_stuff.data_home, 'data', self.get_key())

    print_handler_f = staticmethod(folder_adapter(write_diffcovs_data))

    read_f = staticmethod(read_diffcovs_data)


class filtered_by_column_label_DataFrame(keyed_object):

    def get_introspection_key(self):
        return '%s_%s' % ('filtered', self.filter_f.get_key())

    def __init__(self, filter_f):
        self.filter_f = filter_f

    def location_f(self, data):
        return data.get_location()

    def key_f(self, data):
        return '%s_%s'% (self.get_key(), data.get_key())
    @key
    def __call__(self, data):
        return keyed_DataFrame(data.loc[:,[pid for pid in data.columns if self.filter_f(pid)]])

class beta_noise(keyed_object):

    def get_introspection_key(self):
        return '%s_%.3f' % ('betanoise', self.phi)

    def __init__(self, phi):
        self.phi = phi

    def __call__(self, m):
        return get_rand_beta(m, self.phi)

class normal_noise(keyed_object):

    def get_introspection_key(self):
        return '%s_%.3f' % ('normalnoise', self.sd)

    def __init__(self, sd):
        self.sd = sd

    def __call__(self, m):
        import random
        return random.normalvariate(m, self.sd)


class simulated_get_data_f(possibly_cached):
    """
    re-seed random number generator at every call
    all random calls from here are fed the freshly seeded random.Random()
    """

    def __init__(self, params, pops, id_to_x_s_f, times, noise_f, seed=0):
        self.params, self.pops, self.id_to_x_s_f, self.times, self.noise_f, self.seed = params, pops, id_to_x_s_f, times, noise_f, seed

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
            ys = pandas.Series([self.noise_f(the_f(t,s,a,b,c)) for t in self.times], index=self.times)
            l.append(simulated_datum(pid, x, x, x, s, ys, a, b, c))
        return data(l)

    def get_introspection_key(self):
        return '%s_%s_%s_%s_%s' % ('sim_data_f', self.params.get_key(), self.pops.get_key(), self.id_to_x_s_f.get_key(), self.noise_f.get_key())

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



class asymptotic_data_from_data(possibly_cached):

    def location_f(self, data):
        return '%s/%s' % (data.get_location(), 'asymp')
    
    def key_f(self, data):
        return '%s_%s' % (self.get_key(), data.get_key())

    def get_introspection_key(self):
        return '%s_%.2f_%.2f' % ('asymp_data', self.c_cutoff, self.quorum)

    def __init__(self, c_cutoff, quorum):
        self.c_cutoff, self.quorum = c_cutoff, quorum

    @key
#    @save_to_file
    def __call__(self, _data):
        """
        for each datapoint, look at c, if it's really big, keep timepoints after some cutoff - 2 x fit c value
        """
        asym_data_l = []
        for _datum in _data:
            a,b,c = get_curve_abc(_datum.s, _datum.ys)
            cutoff = self.c_cutoff * c
            ys_to_use = _datum.ys[_datum.ys.index > cutoff]
            if len(ys_to_use) > self.quorum:
                asym_data_l.append(datum_for_asymptotic(_datum.pid, _datum.xa, _datum.s, ys_to_use))
        return data(asym_data_l)


class initial_value_data(possibly_cached):

    def location_f(self, data):
        return '%s/%s' % (data.get_location(), 'initial')
    
    def key_f(self, data):
        return '%s_%s' % (self.get_key(), data.get_key())

    def get_introspection_key(self):
        return '%s_%.2f_%.2f' % ('initial_data', self.num_points, self.window_width)

    def __init__(self, num_points, window_width):
        self.num_points, self.window_width = num_points, window_width

    @key
    def __call__(self, _data):
        """
        for each datapoint, fit abc to extrapolate value at 0.  then generate several values in small window around it
        """
        data_l = []
        for _datum in _data:
            a,b,c = get_curve_abc(_datum.s, _datum.ys)
            time_0_val = _datum.s * (1.0 - a) * (1.0 - b)
            import random
            ys_to_use = pandas.Series([random.uniform(max(0.0,time_0_val-self.window_width), min(1.0,time_0_val+self.window_width)) for i in xrange(self.num_points)])
            data_l.append(datum_for_asymptotic(_datum.pid, _datum.xa, _datum.s, ys_to_use))
        
        return data(data_l)


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

class avoid_boundaries_ys_modifier(keyed_object):

    def get_introspection_key(self):
        return '%s_%s' % ('perturb', self.delta)

    def __init__(self, delta):
        self.delta = delta

    def __call__(self, t, v):
        if v < 0.5:
            return t, max(v, self.delta)
        if v >= 0.5:
            return t, min(v, 1.0-self.delta)


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
id_iterators
"""

class all_ucla_pid_iterator(keyed_object):

    def get_introspection_key(self):
        return 'all_ucla'

    def __init__(self):
        self.pids = pandas.read_csv(global_stuff.all_pid_file, header=False, index_col=None, squeeze=True, converters={0:str}).tolist()

    def __iter__(self):
        return iter(self.pids)


class single_pid_iterator(keyed_object):

    def __init__(self, it):
        self.it = it

    def get_introspection_key(self):
        return str(self.it)

    def __iter__(self):
        return iter([self.it])

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

class fake_pid_iterator(keyed_object):

    def get_introspection_key(self):
        return '%s_%d' % ('dum', self.num)

    def __init__(self, num):
        self.num = num

    def __iter__(self):
        return iter(xrange(self.num))


class not_flat_pid(keyed_object):
    
    def get_introspection_key(self):
        return 'not_flat'

    def __init__(self):
        self.pids = pandas.read_csv(global_stuff.not_flat_file, header=None,index_col=None, squeeze=True, converters={0:str}).tolist()

    def __call__(self, pid):
        return pid in self.pids

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

def get_categorical_fs(backing_f, bins):
    """
    given list of bins, backing feature, returns list of bin features
    """
    return keyed_list([bin_f(backing_f, bin) for bin in bins[0:-1]])

def get_categorical_fs_all_levels(backing_f, bins):
    """
    given list of bins, backing feature, returns list of bin features
    """
    return keyed_list([bin_f(backing_f, bin) for bin in bins])
