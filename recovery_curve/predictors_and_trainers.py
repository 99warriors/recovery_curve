from management_stuff import *
from prostate_specifics import *
import pdb
import itertools


class abc_point_predictor(keyed_object):
    
    def __call__(self, datum):
        pass

class abc_distribution_predictor(keyed_object):

    def __call__(self, datum):
        pass

class abc_phi_m_distribution_predictor(keyed_object):

    def __call__(self, datum):
        pass

class t_point_predictor(keyed_object):

    def __call__(self, datum, t):
        pass

class t_distribution_predictor(keyed_object):

    def __call__(self, datum, t):
        pass

class logreg_predictor(t_point_predictor):
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

class logreg_trainer(possibly_cached):
    """
    returns trained logreg predictor
    """

    def normal(self):
        return True

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


class logregshape_predictor(t_point_predictor):
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

class logregshape_trainer(possibly_cached):
    """
    returns trained logreg predictor, where logistic regression predicts the value *proportional* to s
    generative error term assumed to have variance proportional to s
    """

    def normal(self):
        return True

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


class logregshapeunif_predictor(t_point_predictor):
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

class logregshapeunif_trainer(possibly_cached):
    """
    returns trained logreg predictor, where logistic regression predicts the value *proportional* to s
    generative error term assumed to have independent of initial value
    """
    
    def normal(self):
        return True

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

class Bs_t_distribution_predictor(t_distribution_predictor):
    """
    takes in distribution of Bs to do prediction
    assume params is dict of 3 dataframes for a,b,c
    """
    display_color = 'red'

    display_name = 'full'

    def get_introspection_key(self):
        return '%s_%s_%s' % ('full_pred', self.params.get_key(), self.pops.get_key())

    def __init__(self, params, pops):
        self.params, self.pops = params, pops

    def __call__(self, datum, time):
        return [the_f(time, datum.s, g_a(self.pops.pop_a, datum.xa, B_a), g_b(self.pops.pop_b, datum.xb, B_b), g_c(self.pops.pop_c, datum.xc, B_c)) for (B_a_num, B_a), (B_b_num, B_b), (B_c_num, B_c) in itertools.izip(self.params['B_a'].iterrows(), self.params['B_b'].iterrows(), self.params['B_c'].iterrows())]


class Bs_t_distribution_trainer(keyed_object):

    def normal(self):
        return self.get_posterior_f.normal()

    def __init__(self, get_posterior_f):
        self.get_posterior_f = get_posterior_f

    def get_introspection_key(self):
        return '%s_%s' % ('Bs_distribution_trainer', self.get_posterior_f.get_key())

    def __call__(self, *args):
        posteriors = self.get_posterior_f(*args)
        pops = posteriors.get_pop_f(train_data)
        params = {'B_a':posteriors['B_a'], 'B_b':posteriors['B_b'], 'B_c':posteriors['B_c']}
        return Bs_t_distribution_predictor(params, pops)




class prior_predictor(t_point_predictor):

    display_color = 'blue'

    display_name = 'prior'

    def get_introspection_key(self):
        return '%s_%s' % 'priorpred', self.pops.get_key()

    def __init__(self, pops):
        self.pops = pops

    def __call__(self, datum, time):
        return the_f(time, datum.s, self.pops.pop_a, self.pops.pop_b, self.pops.pop_c)


class prior_trainer(keyed_object):

    def normal(self):
        return True

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




class builtin_abc_distribution_predictor(abc_distribution_predictor):
    """
    just assumes that it is given posterior samples of a,b,c for some patients.  doesn't care how the model obtained them
    """
    def __init__(self, posteriors, max_samples):
        self.posteriors, self.max_samples = posteriors, max_samples
        #self.test_patient_abc_samples_d = test_patient_abc_samples_d

    def __call__(self, datum):
        return self.posteriors.get_abc_phi_m_df(datum.pid, self.max_samples)[['a','b','c']]
        #return self.test_patient_abc_samples_d[datum.pid]

class builtin_regular_abc_distribution_trainer(keyed_object):
    """
    get_posterior_f has to be one for a model that includes test data as missing data
    assumes that get_posterior returns separate a,b,c's for the test_data
    """

    def get_introspection_key(self):
        return '%s_%s' % ('builtin_reg_abc_dist_trainer', self.get_posterior_f.get_key())

    def normal(self):
        return False

    def __init__(self, get_posterior_f):
        self.get_posterior_f = get_posterior_f

    def __call__(self, train_data, test_data):
        posteriors = self.get_posterior_f(train_data, test_data)
        As_test = posteriors['As_test']
        Bs_test = posteriors['Bs_test']
        Cs_test = posteriors['Cs_test']
        d = {}
        for datum in test_data:
            d[datum.pid] = pandas.DataFrame({'a':As_test[datum.pid], 'b':Bs_test[datum.pid], 'c':Cs_test[datum.pid]})
        return builtin_abc_distribution_predictor(d)


class builtin_auto_abc_distribution_trainer(keyed_object):
    """
    passes to trained model the abc's used for training.  suitable for use with self_cv only
    assumes get_posterior does not take in any test_data
    """

    def get_introspection_key(self):
        return '%s_%s' % ('builtin_auto_abc_dist_trainer', self.get_posterior_f.get_key())

    def normal(self):
        return True

    def __init__(self, get_posterior_f, max_samples):
        self.get_posterior_f, self.max_samples = get_posterior_f, max_samples

    def __call__(self, train_data):
        posteriors = self.get_posterior_f(train_data)
        """
        As = posteriors['As']
        Bs = posteriors['Bs']
        Cs = posteriors['Cs']
        d = {}
        for datum in train_data:
            d[datum.pid] = pandas.DataFrame({'a':As[datum.pid], 'b':As[datum.pid], 'c':Cs[datum.pid]})
        """
        return builtin_abc_distribution_predictor(posteriors, self.max_samples)
#        return builtin_abc_distribution_predictor(d)
            

class generic_auto_abc_distribution_predictor(abc_distribution_predictor):
    """
    just assumes that it is given posterior samples of a,b,c for some patients.  doesn't care how the model obtained them
    """
    def __init__(self, posteriors, max_samples):
        self.posteriors, self.max_samples = posteriors, max_samples

    def __call__(self, datum):
        return self.posteriors.get_abc_phi_m_df(datum.pid, self.max_samples)[['a','b','c']]

class generic_builtin_abc_distribution_predictor(abc_distribution_predictor):
    """
    just assumes that it is given posterior samples of a,b,c for some patients.  doesn't care how the model obtained them
    """
    def __init__(self, posteriors, max_samples):
        self.posteriors, self.max_samples = posteriors, max_samples

    def __call__(self, datum):
        it = self.posteriors.get_test_abc_phi_m_df(datum.pid, self.max_samples)
        return it[['a_test','b_test','c_test']]

class generic_builtin_abc_distribution_trainer(keyed_object):
    """
    passes to trained model the abc's used for training.  suitable for use with self_cv only
    assumes get_posterior does not take in any test_data
    """

    def get_introspection_key(self):
        return '%s_%s' % ('generic_builtin_abc_dist_trainer', self.get_posterior_f.get_key())

    def normal(self):
        return False

    def __init__(self, get_posterior_f, max_samples):
        self.get_posterior_f, self.max_samples = get_posterior_f, max_samples

    def __call__(self, train_data, test_data):
        posteriors = self.get_posterior_f(train_data, test_data)
        return generic_builtin_abc_distribution_predictor(posteriors, self.max_samples)



class generic_auto_abc_distribution_top_k_predictor(abc_distribution_predictor):
    """
    just assumes that it is given posterior samples of a,b,c for some patients.  doesn't care how the model obtained them
    """
    def __init__(self, posteriors, k):
        self.posteriors, self.k = posteriors, k
        self.max_samples = 999999

    def __call__(self, datum):
        return self.posteriors.get_best_k_abc_phi_m(datum.pid, self.max_samples, self.k)['a','b','c']


class generic_auto_abc_distribution_top_k_trainer(keyed_object):
    """
    passes to trained model the abc's used for training.  suitable for use with self_cv only
    assumes get_posterior does not take in any test_data
    """

    def get_introspection_key(self):
        return '%s_%s' % ('generic_auto_abc_dist_top_ktrainer', self.get_posterior_f.get_key())

    def normal(self):
        return True

    def __init__(self, get_posterior_f, k):
        self.get_posterior_f, self.k = get_posterior_f, k

    def __call__(self, train_data):
        posteriors = self.get_posterior_f(train_data)
        return generic_auto_abc_distribution_top_k_predictor(posteriors, self.k)

class generic_auto_abc_phi_m_distribution_predictor(abc_phi_m_distribution_predictor):
    """
    just assumes that it is given posterior samples of a,b,c for some patients.  doesn't care how the model obtained them
    """
    def __init__(self, posteriors, max_samples):
        self.posteriors, self.max_samples = posteriors, max_samples

    def __call__(self, datum):
        ans = self.posteriors.get_abc_phi_m_df(datum.pid, self.max_samples)[['a','b','c','phi_m']]
        return ans


class generic_auto_abc_distribution_trainer(keyed_object):
    """
    passes to trained model the abc's used for training.  suitable for use with self_cv only
    assumes get_posterior does not take in any test_data
    """

    def get_introspection_key(self):
        return '%s_%s' % ('generic_auto_abc_dist_trainer', self.get_posterior_f.get_key())

    def normal(self):
        return True

    def __init__(self, get_posterior_f, max_samples):
        self.get_posterior_f, self.max_samples = get_posterior_f, max_samples

    def __call__(self, train_data):
        posteriors = self.get_posterior_f(train_data)
        return generic_auto_abc_distribution_predictor(posteriors, self.max_samples)

class generic_auto_abc_phi_m_distribution_trainer(keyed_object):
    """
    passes to trained model the abc's used for training.  suitable for use with self_cv only
    assumes get_posterior does not take in any test_data
    """

    def get_introspection_key(self):
        return '%s_%s' % ('generic_auto_abc_phi_m_dist_trainer', self.get_posterior_f.get_key())

    def normal(self):
        return True

    def __init__(self, get_posterior_f, max_samples):
        self.get_posterior_f, self.max_samples = get_posterior_f, max_samples

    def __call__(self, train_data):
        posteriors = self.get_posterior_f(train_data)
        return generic_auto_abc_phi_m_distribution_predictor(posteriors, self.max_samples)

class generic_auto_f_star_distribution_predictor(keyed_object):

    def __init__(self, posteriors, max_samples):
        self.posteriors, self.max_samples = posteriors, max_samples

    def __call__(self, datum, t):
        """
        for that time, calculate some f(t)
        """
        return [get_rand_beta(the_f(t,datum.s,a,b,c), phi_m) for num, (a,b,c,phi_m) in self.posteriors.get_abc_phi_m_df(datum.pid, self.max_samples)[['a','b','c','phi_m']].iterrows()]

class generic_auto_f_star_distribution_trainer(keyed_object):

    def get_introspection_key(self):
        return '%s_%s' % ('generic_auto_fstar_trainer', self.get_posterior_f.get_key())

    def normal(self):
        return True

    def __init__(self, get_posterior_f, max_samples):
        self.get_posterior_f, self.max_samples = get_posterior_f, max_samples

    def __call__(self, train_data):
        posteriors = self.get_posterior_f(train_data)
        return generic_auto_f_star_distribution_predictor(posteriors, self.max_samples)


class f_star_distribution_predictor(keyed_object):

    def __init__(self, posteriors, max_samples):
        self.posteriors, self.max_samples = posteriors, max_samples
        #self.test_patient_abc_samples_d, self.phi_ms, self.max_samples = test_patient_abc_samples_d, phi_ms, max_samples

    def __call__(self, datum, t):
        """
        for that time, calculate some f(t)
        """
        return [get_rand_beta(the_f(t,datum.s,a,b,c), phi_m) for num, (a,b,c,phi_m) in self.posteriors.get_abc_phi_m_df(datum.pid, self.max_samples)['a','b','c','d'].iterrows()]
        N = self.phi_ms.shape[0]
        if self.max_samples < N:
            to_use = [int(i*float(N/self.max_samples)) for i in xrange(self.max_samples)]
        else:
            to_use = range(N)

        return [get_rand_beta(the_f(t,datum.s,a,b,c), phi_m) for (abc_num,(a,b,c)), (phi_m_num, phi_m) in itertools.izip(self.test_patient_abc_samples_d[datum.pid].iloc[to_use,:].iterrows(), self.test_patient_abc_samples_d[datum.pid][to_use,:].iterrows())]

class f_star_nonshared_noise_distribution_predictor(keyed_object):

    def __init__(self, posteriors, max_samples):
        self.posteriors, self.max_samples = posteriors, max_samples


        #self.test_patient_abc_phi_m_samples_d, self.max_samples = test_patient_abc_phi_m_samples_d, max_samples

    def __call__(self, datum, t):
        """
        for that time, calculate some f(t)
        """
        return [get_rand_beta(the_f(t,datum.s,a,b,c), phi_m) for num, (a,b,c,phi_m) in self.posteriors.get_abc_phi_m_df(datum.pid, self.max_samples)['a','b','c','d'].iterrows()]
        N = self.test_patient_abc_phi_m_samples_d.iteritems().next()[1].shape[0]
        if self.max_samples < N:
            to_use = [int(i*float(N/self.max_samples)) for i in xrange(self.max_samples)]
        else:
            to_use = range(N)

        return [get_rand_beta(the_f(t,datum.s,a,b,c), phi_m) for (num,(a,b,c,phi_m)) in self.test_patient_abc_phi_m_samples_d[datum.pid].iloc[to_use,:].iterrows()]

class builtin_auto_f_star_distribution_trainer(keyed_object):

    def get_introspection_key(self):
        return '%s_%s' % ('builtin_fstar_trainer', self.get_posterior_f.get_key())

    def normal(self):
        return True

    def __init__(self, get_posterior_f, max_samples):
        self.get_posterior_f, self.max_samples = get_posterior_f, max_samples

    def __call__(self, train_data):
        posteriors = self.get_posterior_f(train_data)
        """
        As = posteriors['As']
        Bs = posteriors['Bs']
        Cs = posteriors['Cs']
        phi_ms = posteriors['phi_m']
        
        d = {}
        for datum in train_data:
            d[datum.pid] = pandas.DataFrame({'a':As[datum.pid], 'b':As[datum.pid], 'c':Cs[datum.pid]})
        """
        return f_star_distribution_predictor(posteriors, self.max_samples)
        #return f_star_distribution_predictor(d, phi_ms, self.max_samples)

class builtin_auto_f_star_nonshared_noise_distribution_trainer(keyed_object):

    def get_introspection_key(self):
        return '%s_%s' % ('builtin_fstar_trainer', self.get_posterior_f.get_key())

    def normal(self):
        return True

    def __init__(self, get_posterior_f, max_samples):
        self.get_posterior_f, self.max_samples = get_posterior_f, max_samples

    def __call__(self, train_data):
        posteriors = self.get_posterior_f(train_data)
        """
        As = posteriors['As']
        Bs = posteriors['Bs']
        Cs = posteriors['Cs']
        phi_ms = posteriors['phi_ms']
        
        d = {}
        for datum in train_data:
            d[datum.pid] = pandas.DataFrame({'a':As[datum.pid], 'b':As[datum.pid], 'c':Cs[datum.pid], 'phi_m':phi_ms[datum.pid]})
        """
        return f_star_nonshared_noise_distribution_predictor(posteriors, self.max_samples)
        #return f_star_nonshared_noise_distribution_predictor(d, self.max_samples)

class t_point_predictor_from_abc_point_predictor(t_point_predictor):

    def __init__(self, abc_point_predictor):
        self.abc_point_predictor = abc_point_predictor

    def __call__(self, datum, t):
        a,b,c = self.abc_point_predictor(datum)
        return the_f(t, datum.s, a, b, c)

class t_point_trainer_from_abc_point_trainer(keyed_object):
    """
    takes in a abc_trainer.  returns t'ed version of the returned abc_trainer
    """

    def get_introspection_key(self):
        return '%s_%s' % ('t_pt_from_abc_pt', self.abc_point_trainer.get_key())

    def normal(self):
        return self.abc_point_trainer.normal()

    def __init__(self, abc_point_trainer):
        self.abc_point_trainer = abc_point_trainer

    def __call__(self, *args):
        return t_point_predictor_from_abc_point_predictor(self.abc_point_trainer(*args))

class abc_point_predictor_from_abc_distribution_predictor(abc_point_predictor):

    def __init__(self, abc_distribution_predictor, summarize_f):
        self.abc_distribution_predictor, self.summarize_f = abc_distribution_predictor, summarize_f

    def __call__(self, datum):
        abc_s = self.abc_distribution_predictor(datum)
        return self.summarize_f(abc_s['a']), self.summarize_f(abc_s['b']), self.summarize_f(abc_s['c'])


class abc_point_trainer_from_abc_distribution_trainer(keyed_object):

    def get_introspection_key(self):
        return '%s_%s' % ('abc_point_trainer', self.abc_distribution_trainer.get_key())

    def normal(self):
        return self.abc_distribution_trainer.normal()

    def __init__(self, abc_distribution_trainer, summarize_f):
        self.abc_distribution_trainer, self.summarize_f = abc_distribution_trainer, summarize_f

    def __call__(self, *args):
        abc_distribution_predictor = self.abc_distribution_trainer(*args)
        return abc_point_predictor_from_abc_distribution_predictor(abc_distribution_predictor, self.summarize_f)
        


class t_distribution_predictor_from_abc_distribution_predictor(t_distribution_predictor):

    def __init__(self, abc_distribution_predictor):
        self.abc_distribution_predictor = abc_distribution_predictor

    def __call__(self, datum, t):
        abc_s = self.abc_distribution_predictor(datum)
        return [the_f(t,datum.s,a,b,c) for num, (a,b,c) in abc_s.iterrows()]

class t_distribution_trainer_from_abc_distribution_trainer(keyed_object):

    def get_introspection_key(self):
        return '%s_%s' % ('t_dist', self.abc_distribution_trainer.get_key())

    def normal(self):
        return self.abc_distribution_trainer.normal()

    def __init__(self, abc_distribution_trainer):
        self.abc_distribution_trainer = abc_distribution_trainer

    def __call__(self, *args):
        abc_distribution_predictor = self.abc_distribution_trainer(*args)
        return t_distribution_predictor_from_abc_distribution_predictor(abc_distribution_predictor)

class t_point_predictor_from_t_distribution_predictor(t_point_predictor, possibly_cached):


    def __init__(self, t_distribution_predictor, summarize_f):
        self.t_distribution_predictor, self.summarize_f = t_distribution_predictor, summarize_f

    def __call__(self, datum, t):
        if t==1: print datum.pid
        ans = self.summarize_f(self.t_distribution_predictor(datum, t))
        return ans

class t_point_trainer_from_t_distribution_trainer(keyed_object):

    def normal(self):
        return self.t_distribution_trainer.normal()

    display_name = 'pointwise_median'

    display_color = 'red'

    def get_introspection_key(self):
        return '%s_%s_%s' % ('pointwise_trainer', self.t_distribution_trainer.get_key(), self.summarize_f.get_key())

    def __init__(self, t_distribution_trainer, summarize_f):
        self.t_distribution_trainer, self.summarize_f = t_distribution_trainer, summarize_f

    def __call__(self, *args):
        t_distribution_predictor = self.t_distribution_trainer(*args)
        return t_point_predictor_from_t_distribution_predictor(t_distribution_predictor, self.summarize_f)

class beta_hierarchical_obs_val_predictor(keyed_object):

    def __init__(self, Zs, phi_ms, max_num):
        self.Zs, self.phi_ms, self.max_num = Zs, phi_ms, max_num

    def __call__(self, datum):
        this_Zs = filter_by_max_num(self.Zs[datum.pid], self.max_num)
        this_phi_ms = filter_by_max_num(self.phi_ms, self.max_num)
        return [get_rand_beta(z, phi) for (z_num, z),(phi_num, phi) in itertools.izip(this_Zs.iteritems(), this_phi_ms.iteritems())]

class auto_beta_hierarchical_obs_val_distribution_trainer(keyed_object):

    def normal(self):
        return True

    def __init__(self, get_posterior_f, max_num):
        self.get_posterior_f, self.max_num = get_posterior_f, max_num

    def __call__(self, train_data):
        posteriors = self.get_posterior_f(train_data)
        Zs = posteriors['Zs']
        phi_ms = posteriors['phi_m'][0]
        return beta_hierarchical_obs_val_predictor(Zs, phi_ms, self.max_num)

class beta_hierarchical_latent_val_predictor(keyed_object):

    def __init__(self, Zs, max_num):
        self.Zs, self.max_num = Zs, max_num

    def __call__(self, datum):
        return filter_by_max_num(self.Zs[datum.pid], self.max_num)


class auto_beta_hierarchical_latent_val_distribution_trainer(keyed_object):

    def normal(self):
        return True

    def __init__(self, get_posterior_f, max_num):
        self.get_posterior_f, self.max_num = get_posterior_f, max_num

    def __call__(self, train_data):
        posteriors = self.get_posterior_f(train_data)
        Zs = posteriors['Zs']
        return beta_hierarchical_latent_val_predictor(Zs, self.max_num)

class curve_fit_cheating_abc_point_predictor(keyed_object):

    def __call__(self, datum):
        a,b,c = get_curve_abc(datum.s, datum.ys)
        return a,b,c


class curve_fit_cheating_abc_point_trainer(keyed_object):

    def normal(self):
        return True

    def get_introspection_key(self):
        return 'curve_fit_cheating_trainer'

    def __call__(self, train_data):
        return curve_fit_cheating_abc_point_predictor()
