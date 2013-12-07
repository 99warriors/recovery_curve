from recovery_curve.management_stuff import *
from recovery_curve.prostate_specifics import *
import recovery_curve.global_stuff as gs
import pandas

import multiprocessing

patient_K = gs.patient_K

"""
need function that
- takes in pop_a, patient a's, 
"""

def get_starting_pos_beta_with_test(train_data, test_data, pops):

    # get abc for every patient from curve fitting, put into matrix

    abc_d = {}
    for datum in train_data:
        abc_d[datum.pid] = pandas.Series(get_curve_abc(datum.s, datum.ys),index = ['a','b','c'])
    abc_df = pandas.DataFrame(abc_d).T

    # get B's
    X_a = pandas.DataFrame({datum.pid:datum.xa for datum in train_data})
    B_a = train_logistic_model(X_a, abc_df['a'], logit(pops.pop_a))
    X_b = pandas.DataFrame({datum.pid:datum.xb for datum in train_data})
    B_b = train_logistic_model(X_b, abc_df['b'], logit(pops.pop_b))
    X_c = pandas.DataFrame({datum.pid:datum.xc for datum in train_data})
    B_c = train_logistic_model(X_c, abc_df['c'], math.exp(pops.pop_c))

    # store the values
    d = {}
    
    d['B_a'] = B_a
    d['B_b'] = B_b
    d['B_c'] = B_c
    """
    d['as'] = abc_df['a']
    d['bs'] = abc_df['b']
    d['cs'] = abc_df['c']
    d['as_test'] = [g_a(pops.pop_a, datum.xa, B_a) for datum in test_data]
    d['bs_test'] = [g_b(pops.pop_b, datum.xb, B_b) for datum in test_data]
    d['cs_test'] = [g_c(pops.pop_c, datum.xc, B_c) for datum in test_data]
    """

    d['as'] = [0.5 for i in xrange(len(train_data))]
    d['bs'] = [0.5 for i in xrange(len(train_data))]
    d['cs'] = [2.0 for i in xrange(len(train_data))]
    d['as_test'] = [0.5 for i in xrange(len(test_data))]
    d['bs_test'] = [0.5 for i in xrange(len(test_data))] 
    d['cs_test'] = [2.0 for i in xrange(len(test_data))] 

    #pdb.set_trace()

    return d



class posterior(keyed_dict):
    """
    posterior should be able to return a dataframe of a,b,c,phi_m, given a pid.  later on, maybe return those parameters for test data too.
    """
    def get_abc_phi_m_df(self, pid):
        raise NotImplementedError

    def get_log_ps(self, pid, max_samples, k):
        return filter_by_max_num(self['log_ps'], max_samples)

    def get_best_k_abc_phi_m(self, pid, max_samples, k):
        abc_phi_ms = self.get_abc_phi_m_df(pid, max_samples)
        log_ps = np.array(self.get_log_ps(pid, max_samples))
        best_indicies = np.argsort(-1.0*log_ps)[0:k]
        return abc_phi_ms.iloc[best_indicies,:]

class shared_phi_m_posterior(posterior):

    def get_abc_phi_m_df(self, pid, max_samples):
        d = {}
        As = self['As']
        Bs = self['Bs']
        Cs = self['Cs']
        phi_ms = self['phi_m'].iloc[:,0]

        return filter_by_max_num(pandas.DataFrame({'a':As[pid], 'b':Bs[pid], 'c':Cs[pid], 'phi_m':phi_ms}), max_samples)


class ind_phi_m_posterior(posterior):

    def get_abc_phi_m_df(self, pid, max_samples):
        d = {}
        As = self['As']
        Bs = self['Bs']
        Cs = self['Cs']
        # this will just be a single number
        phi_ms = self['phi_ms']
        ans = filter_by_max_num(pandas.DataFrame({'a':As[pid], 'b':Bs[pid], 'c':Cs[pid], 'phi_m':phi_ms[pid]}), max_samples)
        return ans.reindex(np.random.permutation(ans.index))

class plot_ind_phi_m_posterior(possibly_cached):
    """
    for now, just plot m vs a, b, c
    """

    def get_introspection_key(self):
        return 'ind_phi_m_posterior_plots'

    def key_f(self, posterior):
        return '%s_%s' % (self.get_key(), posterior.get_key())

    def location_f(self, posterior):
        return '%s/%s' % (global_stuff.for_dropbox, 'posterior_scatters')

    def __init__(self, max_samples):
        self.max_samples = max_samples

    print_handler_f = staticmethod(multiple_figures_to_pdf)

    @save_to_file
    def __call__(self, posterior):

        pids = posterior['As'].columns

        def plot(pid):
            fig = plt.figure()
            abcps = posterior.get_abc_phi_m_df(pid, self.max_samples)
            print pid
            ax = fig.add_subplot(2,2,1)
            ax.set_xlim([0,1])
            ax.set_ylim([0,1])
            ax.plot(abcps['phi_m'], abcps['a'], linestyle='None', marker=',')
            ax.set_xlabel('phi_m')
            ax.set_ylabel('a')
            
            ax = fig.add_subplot(2,2,2)
            ax.set_xlim([0,1])
            ax.set_ylim([0,1])
            ax.plot(abcps['phi_m'], abcps['b'], linestyle='None', marker=',')
            ax.set_xlabel('phi_m')
            ax.set_ylabel('b')

            ax = fig.add_subplot(2,2,3)
            ax.set_xlim([0,1])
            ax.plot(abcps['phi_m'], abcps['c'], linestyle='None', marker=',')
            ax.set_xlabel('phi_m')
            ax.set_ylabel('c')

            ax = fig.add_subplot(2,2,4)
            ax.set_xlim([0,1])
            ax.plot(abcps['b'], abcps['c'], linestyle='None', marker=',')
            ax.set_xlabel('b')
            ax.set_ylabel('c')

            fig.suptitle(pid)
            fig.subplots_adjust(hspace=0.3,wspace=0.3)

            return fig

        #figs = parallel_map(plot, pids, global_stuff.num_processors)
        figs = map(plot, pids[0:20])

        return figs

class fixed_phi_m_posterior(posterior):

    def get_abc_phi_m_df(self, pid, max_samples):
        d = {}
        As = self['As']
        Bs = self['Bs']
        Cs = self['Cs']
        # this will just be a single number
        phi_m = self['phi_m']
        N = As.shape[0]
        ans = filter_by_max_num(pandas.DataFrame({'a':As[pid], 'b':Bs[pid], 'c':Cs[pid], 'phi_m':pandas.Series([phi_m for x in xrange(N)],index=As.index)}), max_samples)
        return ans

    def get_test_abc_phi_m_df(self, pid, max_samples):
        d = {}
        As = self['As_test']
        Bs = self['Bs_test']
        Cs = self['Cs_test']
        phi_ms = pandas.Series([self['phi_m'] for i in xrange(len(As))],index = As.index)
        return filter_by_max_num(pandas.DataFrame({'a_test':As[pid], 'b_test':Bs[pid], 'c_test':Cs[pid], 'phi_m':phi_ms}), max_samples)


class fixed_a_fixed_phi_m_posterior(posterior):

    def get_abc_phi_m_df(self, pid, max_samples):
        d = {}
        # as will be a series where index are pids
        As = self['As']
        Bs = self['Bs']
        Cs = self['Cs']
        # this will just be a single number
        phi_m = self['phi_m']
        N = Bs.shape[0]
        return filter_by_max_num(pandas.DataFrame({'a':[As[pid] for x in xrange(N)], 'b':Bs[pid], 'c':Cs[pid], 'phi_m':pandas.Series([phi_m for x in xrange(N)])}), max_samples)



class merged_get_posterior_f(possibly_cached):

    def get_introspection_key(self):
        return '%s_%d_%d' % (self.get_posterior_f_cons_partial.get_key(), self.iters, self.chains)

    def key_f(self, data, test_data=None):
        return '%s_%s' % (self.get_key(), data.get_key())

    def location_f(self, data, test_data=None):
        return '%s/%s' % (data.get_location(), 'merged_diffcovs')

    print_handler_f = staticmethod(folder_adapter(write_posterior_traces))

    read_f = staticmethod(read_posterior_traces)

    def __init__(self, get_posterior_f_cons_partial, iters, chains):
        self.get_posterior_f_cons_partial, self.iters, self.chains = get_posterior_f_cons_partial, iters, chains
        self.get_pops_f = self.get_posterior_f_cons_partial.args[0]

    @key
    @cache
    #@read_from_pickle
    @save_to_pickle
    def __call__(self, *args):
        posteriors = []
        for seed in range(self.chains):
            get_posterior_f = self.get_posterior_f_cons_partial(iters=self.iters, chains=1, seed=seed)
            posteriors.append(get_posterior_f(*args))



        return reduce(merge_posteriors, posteriors)
        ans = multichain_posterior(reduce(merge_posteriors, posteriors))
        ans.num_chains = self.chains
        return ans
        #return multichain_posterior(ans, self.chains)
            

class parallel_merged_get_posterior_f(possibly_cached):

    def get_introspection_key(self):
        return '%s_%s_%d_%d' % ('parl', self.get_posterior_f_cons_partial.get_key(), self.iters, self.chains)

    def key_f(self, data, test_data=None):
        return '%s_%s' % (self.get_key(), data.get_key())

    def location_f(self, data, test_data=None):
        return '%s/%s' % (data.get_location(), 'parallel_merged_diffcovs')

    print_handler_f = staticmethod(folder_adapter(write_posterior_traces))

    read_f = staticmethod(read_posterior_traces)

    def __init__(self, get_posterior_f_cons_partial, iters, chains, num_processes):
        self.get_posterior_f_cons_partial, self.iters, self.chains, self.num_processes = get_posterior_f_cons_partial, iters, chains, num_processes
        self.get_pops_f = self.get_posterior_f_cons_partial.args[0]

    @key
    @cache
#    @read_from_pickle
    @save_to_pickle
    def __call__(self, *__args):

        import multiprocessing

        get_posterior_f_queue = multiprocessing.Queue()
        for seed in xrange(self.chains):
            get_posterior_f_queue.put(self.get_posterior_f_cons_partial(iters=self.iters, chains=1, seed=seed))

        for z in xrange(self.num_processes):
            get_posterior_f_queue.put(None)

        posteriors = multiprocessing.Manager().list()

        def worker(_get_posterior_f_queue, _posteriors, _args):
            for _get_posterior_f in iter(_get_posterior_f_queue.get, None):
                #print _get_posterior_f
                #print _get_posterior_f.get_key()
                #print _args
                print 'worker'
                _posteriors.append(_get_posterior_f(*_args))


        #print __args, 'ggggg'
        #for x in iter(get_posterior_f_queue.get, None):
        #    print 'zz', x
        #pdb.set_trace()
        workers = []
        for i in xrange(self.num_processes):
            p = multiprocessing.Process(target=worker, args=(get_posterior_f_queue, posteriors, __args))
            p.start()
            workers.append(p)

        for p in workers:
            p.join()

        import copy
        posteriors_copy = [copy.deepcopy(posterior) for posterior in posteriors]

        for posterior, i in zip(posteriors_copy, xrange(len(posteriors_copy))):
            for p in posterior:
                try:
                    posterior[p].index = [(x,i) for x in posterior[p].index]
                except AttributeError:
                    pass

        ans = reduce(merge_posteriors, posteriors_copy)

        return ans
        ans = multichain_posterior(reduce(merge_posteriors, posteriors))
        ans.num_chains = self.chains

        return ans



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

    pickle_lock = multiprocessing.Lock()
    extract_lock = multiprocessing.Lock()

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

        """
        setting hyperparameters
        """

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



        fit = pystan.stan(file=self.diffcovs_model_file, data=d, iter=self.iters, seed=self.seed+1, chains=self.chains, verbose=True)


        
        traces = fit.extract(permuted=True)

        posteriors = fixed_phi_m_posterior({})
        N = len(data)
        d_A, d_B, d_C = {}, {}, {}
        num_samples = traces['as'].shape[0]
        if patient_K < num_samples:
            to_keep = [int(z*num_samples/patient_K) for z in range(patient_K)]
        else:
            to_keep = range(num_samples)


        for i in xrange(N):
            try:
                d_A[data[i].pid] = pandas.Series(fit.extract()['as'][to_keep,i])
                d_B[data[i].pid] = pandas.Series(fit.extract()['bs'][to_keep,i])
                d_C[data[i].pid] = pandas.Series(fit.extract()['cs'][to_keep,i])
            except:
                d_A[data[i].pid] = pandas.Series(fit.extract()['as'][to_keep])
                d_B[data[i].pid] = pandas.Series(fit.extract()['bs'][to_keep])
                d_C[data[i].pid] = pandas.Series(fit.extract()['cs'][to_keep])



        posteriors['As'] = pandas.DataFrame(d_A)
        posteriors['Bs'] = pandas.DataFrame(d_B)
        posteriors['Cs'] = pandas.DataFrame(d_C)

        
        # need to convert arrays to dataframes, and give them the same indicies as in data

        try:
            posteriors['B_a'] = pandas.DataFrame(traces['B_a'][to_keep,i])
        except:
            posteriors['B_a'] = pandas.DataFrame(traces['B_a'][to_keep])
        posteriors['B_a'].columns = _a_datum.xa.index
        
        try:
            posteriors['B_b'] = pandas.DataFrame(traces['B_b'][to_keep,i])
        except:
            posteriors['B_b'] = pandas.DataFrame(traces['B_b'][to_keep])
        posteriors['B_b'].columns = _a_datum.xb.index

        try:
            posteriors['B_c'] = pandas.DataFrame(traces['B_c'][to_keep,i])
        except:
            posteriors['B_c'] = pandas.DataFrame(traces['B_c'][to_keep])
        posteriors['B_c'].columns = _a_datum.xb.index


        posteriors['phi_a'] = pandas.DataFrame(traces['phi_a'])
        posteriors['phi_a'].columns = ['phi_a']
        posteriors['phi_b'] = pandas.DataFrame(traces['phi_b'])
        posteriors['phi_b'].columns = ['phi_b']
        posteriors['phi_c'] = pandas.DataFrame(traces['phi_c'])
        posteriors['phi_c'].columns = ['phi_c']

        posteriors['phi_m'] = self.phi_m

        #posteriors['phi_m'] = pandas.DataFrame(traces['phi_m'])
        #posteriors['phi_m'].columns = ['phi_m']

        return posteriors


        """

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
        N = len(data)
        d_A, d_B, d_C = {}, {}, {}
        num_samples = len(traces['B_a'])
        if patient_K < num_samples:
            to_keep = [int(z*num_samples/patient_K) for z in range(patient_K)]
        else:
            to_keep = range(num_samples)
        for i in xrange(N):
            d_A[data[i].pid] = pandas.Series(fit.extract()['as'][to_keep,i])
            d_B[data[i].pid] = pandas.Series(fit.extract()['bs'][to_keep,i])
            d_C[data[i].pid] = pandas.Series(fit.extract()['cs'][to_keep,i])
        posteriors['As'] = pandas.DataFrame(d_A)
        posteriors['Bs'] = pandas.DataFrame(d_B)
        posteriors['Cs'] = pandas.DataFrame(d_C)

        return posteriors

        """

class get_pystan_diffcovs_beta_noise_posterior_f(possibly_cached_folder):
    """
    returns posteriors for diffcovs model, using pystan instead of rstan
    """
    
    def get_introspection_key(self):
        return '%s_%s_%s_%d_%d_%d' % ('pydiffcovs_betanoise', self.get_pops_f.get_key(), self.hypers.get_key(), self.iters, self.chains, self.seed)
        
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
        self.diffcovs_model_file = '%s/%s/%s' % (global_stuff.home, 'recovery_curve/stan_files', 'full_model_diffcovs_beta_noise.stan')

    pickle_lock = multiprocessing.Lock()
    extract_lock = multiprocessing.Lock()

    #@save_and_memoize
    @key
#    @read_from_pickle
    #@save_to_file
    @save_to_pickle
    def __call__(self, data):
        """
        need to convert data to proper form.  also need to convert to my form, which is a dictionary of dataframes, where keys are the original function objects
        """
        import pystan

        d = {}

        """
        setting hyperparameters
        """

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



        fit = pystan.stan(file=self.diffcovs_model_file, data=d, iter=self.iters, seed=self.seed+1, chains=self.chains, verbose=True)


        print '\t\t\t\t\tstart'
        import sys
        sys.stdout.flush()


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
        N = len(data)
        d_A, d_B, d_C = {}, {}, {}
        num_samples = len(traces['B_a'])
        if patient_K < num_samples:
            to_keep = [int(z*num_samples/patient_K) for z in range(patient_K)]
        else:
            to_keep = range(num_samples)
        for i in xrange(N):
            d_A[data[i].pid] = pandas.Series(fit.extract()['as'][to_keep,i])
            d_B[data[i].pid] = pandas.Series(fit.extract()['bs'][to_keep,i])
            d_C[data[i].pid] = pandas.Series(fit.extract()['cs'][to_keep,i])
        posteriors['As'] = pandas.DataFrame(d_A)
        posteriors['Bs'] = pandas.DataFrame(d_B)
        posteriors['Cs'] = pandas.DataFrame(d_C)

        
        print '\t\t\t\t\tfinish'
        import sys
        sys.stdout.flush()


        return posteriors

class get_pystan_curve_fit_beta_noise_nonshared_posterior_f(possibly_cached_folder):
    """
    returns posteriors for diffcovs model, using pystan instead of rstan
    """
    
    def get_introspection_key(self):
        return '%s_%.2f_%d_%d_%d' % ('pydiffcovs_curve_fit_betanoise', self.l_m, self.iters, self.chains, self.seed)
        
    def key_f(self, data):
        return '%s_%s' % (self.get_key(), data.get_key())

    def location_f(self, data):
        return '%s/%s' % (data.get_location(), 'pytrained_diffcovs')

    print_handler_f = staticmethod(folder_adapter(write_posterior_traces))

    read_f = staticmethod(read_posterior_traces)

#    def has_file_content(self, data):
#        return os.path.exists('%s/%s' % (self.full_file_path_f(data), 'out_B_a.csv'))

    def __init__(self, l_m, cs_l, iters, chains, seed):
        self.l_m, self.cs_l, self.iters, self.chains, self.seed = l_m, cs_l, iters, chains, seed
        self.diffcovs_model_file = '%s/%s/%s' % (global_stuff.home, 'recovery_curve/stan_files', 'curve_fit_beta_noise_nonshared.stan')

    pickle_lock = multiprocessing.Lock()
    extract_lock = multiprocessing.Lock()

    #@save_and_memoize
    @key
#    @read_from_pickle
    #@save_to_file
    @save_to_pickle
    def __call__(self, data):
        """
        need to convert data to proper form.  also need to convert to my form, which is a dictionary of dataframes, where keys are the original function objects
        """


        import pystan

        d = {}

        """
        setting hyperparameters
        """

        d['l_m'] = self.l_m
        d['cs_l'] = self.cs_l
        d['N'] = len(data)

        d['ss'] = [a_datum.s for a_datum in data]

        ls = reduce(lambda x, a_datum: x + [len(a_datum.ys)], data, [])
        ts = reduce(lambda x, a_datum: x + a_datum.ys.index.tolist(), data, [])
        vs = reduce(lambda x, a_datum: x + a_datum.ys.tolist(), data, [])
        d['ls'] = ls
        d['ts'] = ts
        d['vs'] = vs
        d['L'] = len(ts)
        assert len(ts) == sum(ls)

        abc_d = {}
        for _datum in data:
            abc_d[_datum.pid] = pandas.Series(get_curve_abc(_datum.s, _datum.ys), index=['a','b','c'])

        abc_df = pandas.DataFrame(abc_d)

        init_d = {}
        init_d['as'] = abc_df.loc['a',:]
        init_d['bs'] = abc_df.loc['b',:]
        init_d['cs'] = abc_df.loc['c',:]
                                             
        """
        set all phi_ms to some small number, say 0.1
        """
        init_d['phi_m'] = [0.1 for i in xrange(len(data))]


        fit = pystan.stan(file=self.diffcovs_model_file, data=d, iter=self.iters, seed=self.seed+1, chains=self.chains, verbose=True, init=[init_d for i in xrange(self.chains)])


        print '\t\t\t\t\tstart'
        import sys
        sys.stdout.flush()


        traces = fit.extract(permuted=True)
        
        # need to convert arrays to dataframes, and give them the same indicies as in data
        
        posteriors = ind_phi_m_posterior({})

        #posteriors['phi_m'] = pandas.DataFrame(traces['phi_m'])
        #posteriors['phi_m'].columns = ['phi_m']

        # also extract the A_i,B_i,C_i parameters
        N = len(data)
        d_A, d_B, d_C, d_phi_m = {}, {}, {}, {}
        num_samples = traces['phi_m'].shape[0]
        if patient_K < num_samples:
            to_keep = [int(z*num_samples/patient_K) for z in range(patient_K)]
        else:
            to_keep = range(num_samples)
        for i in xrange(N):
            d_A[data[i].pid] = pandas.Series(fit.extract()['as'][to_keep,i])
            d_B[data[i].pid] = pandas.Series(fit.extract()['bs'][to_keep,i])
            d_C[data[i].pid] = pandas.Series(fit.extract()['cs'][to_keep,i])
            d_phi_m[data[i].pid] = pandas.Series(fit.extract()['phi_m'][to_keep,i])
        posteriors['As'] = pandas.DataFrame(d_A)
        posteriors['Bs'] = pandas.DataFrame(d_B)
        posteriors['Cs'] = pandas.DataFrame(d_C)
        posteriors['phi_ms'] = pandas.DataFrame(d_phi_m)
        
        print '\t\t\t\t\tfinish'
        import sys
        sys.stdout.flush()

        




        return posteriors


class get_pystan_curve_fit_normal_noise_posterior_f(possibly_cached_folder):
    """
    returns posteriors for diffcovs model, using pystan instead of rstan
    """
    
    def get_introspection_key(self):
        return '%s_%.2f_%d_%d_%d' % ('pydiffcovs_curve_fit_normalnoise', self.l_m, self.iters, self.chains, self.seed)
        
    def key_f(self, data):
        return '%s_%s' % (self.get_key(), data.get_key())

    def location_f(self, data):
        return '%s/%s' % (data.get_location(), 'pytrained_diffcovs')

    print_handler_f = staticmethod(folder_adapter(write_posterior_traces))

    read_f = staticmethod(read_posterior_traces)

#    def has_file_content(self, data):
#        return os.path.exists('%s/%s' % (self.full_file_path_f(data), 'out_B_a.csv'))

    def __init__(self, l_m, cs_l, iters, chains, seed):
        self.l_m, self.cs_l, self.iters, self.chains, self.seed = l_m, cs_l, iters, chains, seed
        self.diffcovs_model_file = '%s/%s/%s' % (global_stuff.home, 'recovery_curve/stan_files', 'curve_fit_normal_noise.stan')

    pickle_lock = multiprocessing.Lock()
    extract_lock = multiprocessing.Lock()

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

        """
        setting hyperparameters
        """

        d['l_m'] = self.l_m
        d['cs_l'] = self.cs_l
        d['N'] = len(data)

        d['ss'] = [a_datum.s for a_datum in data]

        ls = reduce(lambda x, a_datum: x + [len(a_datum.ys)], data, [])
        ts = reduce(lambda x, a_datum: x + a_datum.ys.index.tolist(), data, [])
        vs = reduce(lambda x, a_datum: x + a_datum.ys.tolist(), data, [])
        d['ls'] = ls
        d['ts'] = ts
        d['vs'] = vs
        d['L'] = len(ts)
        assert len(ts) == sum(ls)

        abc_d = {}
        for _datum in data:
            abc_d[_datum.pid] = pandas.Series(get_curve_abc(_datum.s, _datum.ys), index=['a','b','c'])

        abc_df = pandas.DataFrame(abc_d)

        init_d = {}
        init_d['as'] = abc_df.loc['a',:]
        init_d['bs'] = abc_df.loc['b',:]
        init_d['cs'] = abc_df.loc['c',:]
                                             


        fit = pystan.stan(file=self.diffcovs_model_file, data=d, iter=self.iters, seed=self.seed+1, chains=self.chains, verbose=True, init=[init_d for i in xrange(self.chains)])


        print '\t\t\t\t\tstart'
        import sys
        sys.stdout.flush()


        traces = fit.extract(permuted=True)
        
        # need to convert arrays to dataframes, and give them the same indicies as in data
        
        posteriors = shared_phi_m_posterior({})

        posteriors['phi_m'] = pandas.DataFrame(traces['phi_m'])
        posteriors['phi_m'].columns = ['phi_m']

        # also extract the A_i,B_i,C_i parameters
        N = len(data)
        d_A, d_B, d_C = {}, {}, {}
        num_samples = len(traces['phi_m'])
        if patient_K < num_samples:
            to_keep = [int(z*num_samples/patient_K) for z in range(patient_K)]
        else:
            to_keep = range(num_samples)
        for i in xrange(N):
            d_A[data[i].pid] = pandas.Series(fit.extract()['as'][to_keep,i])
            d_B[data[i].pid] = pandas.Series(fit.extract()['bs'][to_keep,i])
            d_C[data[i].pid] = pandas.Series(fit.extract()['cs'][to_keep,i])
        posteriors['As'] = pandas.DataFrame(d_A)
        posteriors['Bs'] = pandas.DataFrame(d_B)
        posteriors['Cs'] = pandas.DataFrame(d_C)

        
        print '\t\t\t\t\tfinish'
        import sys
        sys.stdout.flush()


        return posteriors


class get_pystan_curve_fit_normal_noise_fixed_posterior_f(possibly_cached_folder):
    """
    returns posteriors for diffcovs model, using pystan instead of rstan
    """
    
    def get_introspection_key(self):
        return '%s_%.2f_%.2f_%d_%d_%d' % ('pydiffcovs_curve_fit_normalnoise_fixed', self.phi_m, self.cs_l, self.iters, self.chains, self.seed)
        
    def key_f(self, data):
        return '%s_%s' % (self.get_key(), data.get_key())

    def location_f(self, data):
        return '%s/%s' % (data.get_location(), 'pytrained_diffcovs')

    print_handler_f = staticmethod(folder_adapter(write_posterior_traces))

    read_f = staticmethod(read_posterior_traces)

#    def has_file_content(self, data):
#        return os.path.exists('%s/%s' % (self.full_file_path_f(data), 'out_B_a.csv'))

    def __init__(self, phi_m, cs_l, iters, chains, seed):
        self.phi_m, self.cs_l, self.iters, self.chains, self.seed = phi_m, cs_l, iters, chains, seed
        self.diffcovs_model_file = '%s/%s/%s' % (global_stuff.home, 'recovery_curve/stan_files', 'curve_fit_normal_noise_fixed.stan')

    pickle_lock = multiprocessing.Lock()
    extract_lock = multiprocessing.Lock()

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

        """
        setting hyperparameters
        """

        d['phi_m'] = self.phi_m
        d['cs_l'] = self.cs_l
        d['N'] = len(data)

        d['ss'] = [a_datum.s for a_datum in data]

        ls = reduce(lambda x, a_datum: x + [len(a_datum.ys)], data, [])
        ts = reduce(lambda x, a_datum: x + a_datum.ys.index.tolist(), data, [])
        vs = reduce(lambda x, a_datum: x + a_datum.ys.tolist(), data, [])
        d['ls'] = ls
        d['ts'] = ts
        d['vs'] = vs
        d['L'] = len(ts)
        assert len(ts) == sum(ls)

        abc_d = {}
        for _datum in data:
            abc_d[_datum.pid] = pandas.Series(get_curve_abc(_datum.s, _datum.ys), index=['a','b','c'])

        abc_df = pandas.DataFrame(abc_d)

        init_d = {}
        init_d['as'] = abc_df.loc['a',:]
        init_d['bs'] = abc_df.loc['b',:]
        init_d['cs'] = abc_df.loc['c',:]
                                             


        fit = pystan.stan(file=self.diffcovs_model_file, data=d, iter=self.iters, seed=self.seed+1, chains=self.chains, verbose=True, init=[init_d for i in xrange(self.chains)])
#        fit = pystan.stan(file=self.diffcovs_model_file, data=d, iter=self.iters, seed=self.seed+1, chains=self.chains, verbose=True)

        print '\t\t\t\t\tstart'
        import sys
        sys.stdout.flush()


        traces = fit.extract(permuted=True)
        #pdb.set_trace()
        # need to convert arrays to dataframes, and give them the same indicies as in data

        posteriors = fixed_phi_m_posterior({})

        #posteriors['phi_m'] = pandas.DataFrame(traces['phi_m'])
        #posteriors['phi_m'].columns = ['phi_m']

        # also extract the A_i,B_i,C_i parameters
        N = len(data)
        d_A, d_B, d_C, d_logp = {}, {}, {}, {}
        num_samples = traces['as'].shape[0]
        if patient_K < num_samples:
            to_keep = [int(z*num_samples/patient_K) for z in range(patient_K)]
        else:
            to_keep = range(num_samples)


        for i in xrange(N):
            try:
                d_A[data[i].pid] = pandas.Series(fit.extract()['as'][to_keep,i])
                d_B[data[i].pid] = pandas.Series(fit.extract()['bs'][to_keep,i])
                d_C[data[i].pid] = pandas.Series(fit.extract()['cs'][to_keep,i])
            except:
                d_A[data[i].pid] = pandas.Series(fit.extract()['as'][to_keep])
                d_B[data[i].pid] = pandas.Series(fit.extract()['bs'][to_keep])
                d_C[data[i].pid] = pandas.Series(fit.extract()['cs'][to_keep])



        posteriors['As'] = pandas.DataFrame(d_A)
        posteriors['Bs'] = pandas.DataFrame(d_B)
        posteriors['Cs'] = pandas.DataFrame(d_C)

        posteriors['phi_m'] = self.phi_m

        
        print '\t\t\t\t\tfinish'
        import sys
        sys.stdout.flush()


        return posteriors

class get_pystan_diffcovs_posterior_truncated_phi_m_fixed_has_test_f(possibly_cached_folder):
    """
    returns posteriors for diffcovs model, using pystan instead of rstan
    """
    
    def get_introspection_key(self):
        return '%s_%s_%s_%d_%d_%d' % ('pydiffcovs_truncated_phimfixed_has_test', self.get_pops_f.get_key(), self.hypers.get_key(), self.iters, self.chains, self.seed)
        
    def key_f(self, data, test_data):
        return '%s_%s' % (self.get_key(), data.get_key())

    def location_f(self, data, test_data):
        #print data, test_data
        return '%s/%s/%s' % (data.get_location(), 'pytrained_diffcovs', self.get_pops_f.get_key())

    print_handler_f = staticmethod(folder_adapter(write_posterior_traces))

    read_f = staticmethod(read_posterior_traces)

    def has_file_content(self, data):
        return os.path.exists('%s/%s' % (self.full_file_path_f(data), 'out_B_a.csv'))

    def __init__(self, phi_m, get_pops_f, hypers, iters, chains, seed):
        self.get_pops_f, self.hypers, self.iters, self.chains, self.seed = get_pops_f, hypers, iters, chains, seed
        self.phi_m = phi_m
        self.diffcovs_model_file = '%s/%s/%s' % (global_stuff.home, 'recovery_curve/stan_files', 'full_model_diffcovs_truncated_normal_noise_fixed_has_test.stan')

    #@save_and_memoize
    @key
#    @read_from_pickle
#    @save_to_file
    @save_to_pickle
    def __call__(self, data, test_data):
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


        d['N_test'] = len(test_data)

        xas_test = pandas.DataFrame({a_datum.pid:a_datum.xa for a_datum in test_data})
        xbs_test = pandas.DataFrame({a_datum.pid:a_datum.xb for a_datum in test_data})
        xcs_test = pandas.DataFrame({a_datum.pid:a_datum.xc for a_datum in test_data})
        d['xas_test'] = xas_test.T.as_matrix()
        d['xbs_test'] = xbs_test.T.as_matrix()
        d['xcs_test'] = xcs_test.T.as_matrix()

        d['ss_test'] = [a_datum.s for a_datum in test_data]

        """
        ls_test = reduce(lambda x, a_datum: x + [len(a_datum.ys)], test_data, [])
        ts_test = reduce(lambda x, a_datum: x + a_datum.ys.index.tolist(), test_data, [])
        vs_test = reduce(lambda x, a_datum: x + a_datum.ys.tolist(), test_data, [])
        d['ls_test'] = ls_test
        d['ts_test'] = ts_test
        d['vs_test'] = vs_test
        d['L_test'] = len(ts_test)
        """
        
        """
        set initial parameters.  
        """

        init_d = get_starting_pos_beta_with_test(data, test_data, pops)
                                             
        """
        set phi's
        """

        init_d['phi_a'], init_d['phi_b'], init_d['phi_c'] = 0.4, 0.4, 0.4

        pdb.set_trace()

        fit = pystan.stan(file=self.diffcovs_model_file, data=d, iter=self.iters, seed=self.seed+1, chains=self.chains, verbose=True, init = [init_d for i in xrange(self.chains)], algorithm='HMC')
        print pandas.DataFrame(fit.extract(permuted=True)['B_a'][0:50,:])
        traces = fit.extract(permuted=True)

        posteriors = fixed_phi_m_posterior({})
        N = len(data)
        d_A, d_B, d_C = {}, {}, {}
        d_A_test, d_B_test, d_C_test = {}, {}, {}
        num_samples = traces['as'].shape[0]
        if patient_K < num_samples:
            to_keep = [int(z*num_samples/patient_K) for z in range(patient_K)]
        else:
            to_keep = range(num_samples)


        for i in xrange(N):
            try:
                d_A[data[i].pid] = pandas.Series(fit.extract()['as'][to_keep,i])
                d_B[data[i].pid] = pandas.Series(fit.extract()['bs'][to_keep,i])
                d_C[data[i].pid] = pandas.Series(fit.extract()['cs'][to_keep,i])
            except:
                d_A[data[i].pid] = pandas.Series(fit.extract()['as'][to_keep])
                d_B[data[i].pid] = pandas.Series(fit.extract()['bs'][to_keep])
                d_C[data[i].pid] = pandas.Series(fit.extract()['cs'][to_keep])


        N_test = len(test_data)

        for i in xrange(N_test):
            try:
                d_A_test[test_data[i].pid] = pandas.Series(fit.extract()['as_test'][to_keep,i])
                d_B_test[test_data[i].pid] = pandas.Series(fit.extract()['bs_test'][to_keep,i])
                d_C_test[test_data[i].pid] = pandas.Series(fit.extract()['cs_test'][to_keep,i])
            except:
                d_A_test[test_data[i].pid] = pandas.Series(fit.extract()['as_test'][to_keep])
                d_B_test[test_data[i].pid] = pandas.Series(fit.extract()['bs_test'][to_keep])
                d_C_test[test_data[i].pid] = pandas.Series(fit.extract()['cs_test'][to_keep])

        posteriors['As'] = pandas.DataFrame(d_A)
        posteriors['Bs'] = pandas.DataFrame(d_B)
        posteriors['Cs'] = pandas.DataFrame(d_C)

        posteriors['As_test'] = pandas.DataFrame(d_A_test)
        posteriors['Bs_test'] = pandas.DataFrame(d_B_test)
        posteriors['Cs_test'] = pandas.DataFrame(d_C_test)

        
        # need to convert arrays to dataframes, and give them the same indicies as in data

        try:
            posteriors['B_a'] = pandas.DataFrame(traces['B_a'][to_keep,i])
        except:
            posteriors['B_a'] = pandas.DataFrame(traces['B_a'][to_keep])
        posteriors['B_a'].columns = _a_datum.xa.index
        
        try:
            posteriors['B_b'] = pandas.DataFrame(traces['B_b'][to_keep,i])
        except:
            posteriors['B_b'] = pandas.DataFrame(traces['B_b'][to_keep])
        posteriors['B_b'].columns = _a_datum.xb.index

        try:
            posteriors['B_c'] = pandas.DataFrame(traces['B_c'][to_keep,i])
        except:
            posteriors['B_c'] = pandas.DataFrame(traces['B_c'][to_keep])
        posteriors['B_c'].columns = _a_datum.xb.index


        posteriors['phi_a'] = pandas.DataFrame(traces['phi_a'])
        posteriors['phi_a'].columns = ['phi_a']
        posteriors['phi_b'] = pandas.DataFrame(traces['phi_b'])
        posteriors['phi_b'].columns = ['phi_b']
        posteriors['phi_c'] = pandas.DataFrame(traces['phi_c'])
        posteriors['phi_c'].columns = ['phi_c']

        posteriors['phi_m'] = self.phi_m

        #posteriors['phi_m'] = pandas.DataFrame(traces['phi_m'])
        #posteriors['phi_m'].columns = ['phi_m']

        return posteriors


class get_pystan_curve_fit_normal_noise_fixed_as_fixed_posterior_f(possibly_cached_folder):
    """
    returns posteriors for diffcovs model, using pystan instead of rstan
    """
    
    def get_introspection_key(self):
        return '%s_%.2f_%d_%d_%d' % ('pydiffcovs_curve_fit_normalnoise_fixed_as_fixed', self.phi_m, self.iters, self.chains, self.seed)
        
    def key_f(self, data):
        return '%s_%s' % (self.get_key(), data.get_key())

    def location_f(self, data):
        return '%s/%s' % (data.get_location(), 'pytrained_diffcovs')

    print_handler_f = staticmethod(folder_adapter(write_posterior_traces))

    read_f = staticmethod(read_posterior_traces)

#    def has_file_content(self, data):
#        return os.path.exists('%s/%s' % (self.full_file_path_f(data), 'out_B_a.csv'))

    def __init__(self, phi_m, cs_l, iters, chains, seed):
        self.phi_m, self.cs_l, self.iters, self.chains, self.seed = phi_m, cs_l, iters, chains, seed
        self.diffcovs_model_file = '%s/%s/%s' % (global_stuff.home, 'recovery_curve/stan_files', 'curve_fit_normal_noise_fixed_as_fixed.stan')

    pickle_lock = multiprocessing.Lock()
    extract_lock = multiprocessing.Lock()

    #@save_and_memoize
    @key
#    @read_from_pickle
    #@save_to_file
    @save_to_pickle
    def __call__(self, data):
        """
        need to convert data to proper form.  also need to convert to my form, which is a dictionary of dataframes, where keys are the original function objects
        """
        import pystan

        d = {}

        """
        setting hyperparameters
        """

        d['phi_m'] = self.phi_m
        d['cs_l'] = self.cs_l
        d['N'] = len(data)

        d['ss'] = [a_datum.s for a_datum in data]

        ls = reduce(lambda x, a_datum: x + [len(a_datum.ys)], data, [])
        ts = reduce(lambda x, a_datum: x + a_datum.ys.index.tolist(), data, [])
        vs = reduce(lambda x, a_datum: x + a_datum.ys.tolist(), data, [])
        d['ls'] = ls
        d['ts'] = ts
        d['vs'] = vs
        d['L'] = len(ts)
        assert len(ts) == sum(ls)

        abc_d = {}
        for _datum in data:
            abc_d[_datum.pid] = pandas.Series(get_curve_abc(_datum.s, _datum.ys), index=['a','b','c'])

        abc_df = pandas.DataFrame(abc_d)

        init_d = {}
        #init_d['as'] = abc_df.loc['a',:]
        init_d['bs'] = (abc_df.loc['b',:]+1).tolist()[0]
#        init_d['bs']='wer'
        init_d['cs'] = (abc_df.loc['c',:]+.1).tolist()[0]
                                             
        d['as'] = abc_df.loc['a',:]
        #pdb.set_trace()
        fit = pystan.stan(file=self.diffcovs_model_file, data=d, iter=self.iters, chains=self.chains, verbose=True, init=[init_d for i in xrange(self.chains)], save_dso=False)
        #fit = pystan.stan(file=self.diffcovs_model_file, data=d, iter=self.iters, seed=self.seed+11, chains=self.chains, verbose=True, init='random', save_dso=False)
        #fit = pystan.stan(file=self.diffcovs_model_file, data=d, iter=self.iters, seed=self.seed+1, chains=self.chains, verbose=True)
        gg = fit.extract(permuted=False, inc_warmup=True)[:,0,:]
        hh = fit.extract()['bs'][0]
        print '\t\t\t\t\tstart'
        import sys
        sys.stdout.flush()
        pdb.set_trace()

        traces = fit.extract(permuted=True)
        
        # need to convert arrays to dataframes, and give them the same indicies as in data
        
        #posteriors = keyed_dict({})

        posteriors = fixed_a_fixed_phi_m_posterior({})

        #posteriors['phi_m'] = pandas.DataFrame(traces['phi_m'])
        #posteriors['phi_m'].columns = ['phi_m']

        # also extract the A_i,B_i,C_i parameters
        N = len(data)
        d_A, d_B, d_C = {}, {}, {}
        num_samples = traces['bs'].shape[0]
        if patient_K < num_samples:
            to_keep = [int(z*num_samples/patient_K) for z in range(patient_K)]
        else:
            to_keep = range(num_samples)
        for i in xrange(N):
            #d_A[data[i].pid] = pandas.Series(fit.extract()['as'][to_keep,i])
            try:
                d_B[data[i].pid] = pandas.Series(fit.extract()['bs'][to_keep,i])
                d_C[data[i].pid] = pandas.Series(fit.extract()['cs'][to_keep,i])
            except:
                d_B[data[i].pid] = pandas.Series(fit.extract()['bs'][to_keep])
                d_C[data[i].pid] = pandas.Series(fit.extract()['cs'][to_keep])
        # As is fixed
        posteriors['As'] = d['as']
        posteriors['Bs'] = pandas.DataFrame(d_B)
        posteriors['Cs'] = pandas.DataFrame(d_C)

        posteriors['phi_m'] = self.phi_m

        
        print '\t\t\t\t\tfinish'
        import sys
        sys.stdout.flush()


        return posteriors


class get_pystan_curve_fit_normal_nonshared_noise_posterior_f(possibly_cached_folder):
    """
    returns posteriors for diffcovs model, using pystan instead of rstan
    """
    
    def get_introspection_key(self):
        return '%s_%.2f_%.2f_%d_%d_%d' % ('pydiffcovs_curve_fit_nonshared_normalnoise', self.l_m, self.cs_l, self.iters, self.chains, self.seed)
        
    def key_f(self, data):
        return '%s_%s' % (self.get_key(), data.get_key())

    def location_f(self, data):
        return '%s/%s' % (data.get_location(), 'pytrained_diffcovs')

    print_handler_f = staticmethod(folder_adapter(write_posterior_traces))

    read_f = staticmethod(read_posterior_traces)

#    def has_file_content(self, data):
#        return os.path.exists('%s/%s' % (self.full_file_path_f(data), 'out_B_a.csv'))

    def __init__(self, l_m, cs_l, iters, chains, seed):
        self.l_m, self.cs_l, self.iters, self.chains, self.seed = l_m, cs_l, iters, chains, seed
        self.diffcovs_model_file = '%s/%s/%s' % (global_stuff.home, 'recovery_curve/stan_files', 'curve_fit_normal_noise_nonshared.stan')

    pickle_lock = multiprocessing.Lock()
    extract_lock = multiprocessing.Lock()

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

        """
        setting hyperparameters
        """

        d['l_m'] = self.l_m
        d['cs_l'] = self.cs_l
        d['N'] = len(data)

        d['ss'] = [a_datum.s for a_datum in data]

        ls = reduce(lambda x, a_datum: x + [len(a_datum.ys)], data, [])
        ts = reduce(lambda x, a_datum: x + a_datum.ys.index.tolist(), data, [])
        vs = reduce(lambda x, a_datum: x + a_datum.ys.tolist(), data, [])
        d['ls'] = ls
        d['ts'] = ts
        d['vs'] = vs
        d['L'] = len(ts)
        assert len(ts) == sum(ls)

        abc_d = {}
        for _datum in data:
            abc_d[_datum.pid] = pandas.Series(get_curve_abc(_datum.s, _datum.ys), index=['a','b','c'])

        abc_df = pandas.DataFrame(abc_d)

        init_d = {}
        init_d['as'] = abc_df.loc['a',:]
        init_d['bs'] = abc_df.loc['b',:]
        init_d['cs'] = abc_df.loc['c',:]
                                             
        """
        set all phi_ms to some small number, say 0.1
        """
        init_d['phi_m'] = [0.1 for i in xrange(len(data))]


        fit = pystan.stan(file=self.diffcovs_model_file, data=d, iter=self.iters, seed=self.seed+1, chains=self.chains, verbose=True, init=[init_d for i in xrange(self.chains)])


        print '\t\t\t\t\tstart'
        import sys
        sys.stdout.flush()


        traces = fit.extract(permuted=True)
        
        # need to convert arrays to dataframes, and give them the same indicies as in data
        
        posteriors = ind_phi_m_posterior({})

        #posteriors['phi_m'] = pandas.DataFrame(traces['phi_m'])
        #posteriors['phi_m'].columns = ['phi_m']

        # also extract the A_i,B_i,C_i parameters
        N = len(data)
        d_A, d_B, d_C, d_phi_m = {}, {}, {}, {}
        num_samples = traces['phi_m'].shape[0]
        if patient_K < num_samples:
            to_keep = [int(z*num_samples/patient_K) for z in range(patient_K)]
        else:
            to_keep = range(num_samples)
        for i in xrange(N):
            d_A[data[i].pid] = pandas.Series(fit.extract()['as'][to_keep,i])
            d_B[data[i].pid] = pandas.Series(fit.extract()['bs'][to_keep,i])
            d_C[data[i].pid] = pandas.Series(fit.extract()['cs'][to_keep,i])
            d_phi_m[data[i].pid] = pandas.Series(fit.extract()['phi_m'][to_keep,i])
        posteriors['As'] = pandas.DataFrame(d_A)
        posteriors['Bs'] = pandas.DataFrame(d_B)
        posteriors['Cs'] = pandas.DataFrame(d_C)
        posteriors['phi_ms'] = pandas.DataFrame(d_phi_m)
        
        print '\t\t\t\t\tfinish'
        import sys
        sys.stdout.flush()









        return posteriors


class get_pystan_curve_fit_truncated_normal_noise_fixed_posterior_f(possibly_cached_folder):
    """
    returns posteriors for diffcovs model, using pystan instead of rstan
    """
    
    def get_introspection_key(self):
        return '%s_%.2f_%.2f_%d_%d_%d' % ('pydiffcovs_curve_fit_truncated_normalnoise_fixed', self.phi_m, self.cs_l, self.iters, self.chains, self.seed)
        
    def key_f(self, data):
        return '%s_%s' % (self.get_key(), data.get_key())

    def location_f(self, data):
        return '%s/%s' % (data.get_location(), 'pytrained_diffcovs')

    print_handler_f = staticmethod(folder_adapter(write_posterior_traces))

    read_f = staticmethod(read_posterior_traces)

#    def has_file_content(self, data):
#        return os.path.exists('%s/%s' % (self.full_file_path_f(data), 'out_B_a.csv'))

    def __init__(self, phi_m, cs_l, iters, chains, seed):
        self.phi_m, self.cs_l, self.iters, self.chains, self.seed = phi_m, cs_l, iters, chains, seed
        self.diffcovs_model_file = '%s/%s/%s' % (global_stuff.home, 'recovery_curve/stan_files', 'curve_fit_truncated_normal_noise_fixed.stan')

    pickle_lock = multiprocessing.Lock()
    extract_lock = multiprocessing.Lock()

    #@save_and_memoize
    @key
#    @read_from_pickle
    #@save_to_file
    @save_to_pickle
    def __call__(self, data):
        """
        need to convert data to proper form.  also need to convert to my form, which is a dictionary of dataframes, where keys are the original function objects
        """
        import pystan

        d = {}

        """
        setting hyperparameters
        """

        """
        holy
        """
        abc_d = {}

        for _datum in data:
            abc_d[_datum.pid] = pandas.Series(get_curve_abc(_datum.s, _datum.ys), index=['a','b','c'])

        abc_df = pandas.DataFrame(abc_d)

        for pid in abc_df.columns:
            if abc_df.loc['a',pid] < 0.9:
                abc_df.loc['a',pid] += 0.098

        init_d = {}
        init_d['as'] = abc_df.loc['a',:]
        init_d['bs'] = abc_df.loc['b',:]
        init_d['cs'] = abc_df.loc['c',:]



        


        d['phi_m'] = self.phi_m
        d['cs_l'] = self.cs_l
        d['N'] = len(data)

        d['ss'] = [a_datum.s for a_datum in data]

        ls = reduce(lambda x, a_datum: x + [len(a_datum.ys)], data, [])
        ts = reduce(lambda x, a_datum: x + a_datum.ys.index.tolist(), data, [])
        vs = reduce(lambda x, a_datum: x + a_datum.ys.tolist(), data, [])
        d['ls'] = ls
        d['ts'] = ts
        d['vs'] = vs
        d['L'] = len(ts)
        assert len(ts) == sum(ls)

        
                                             


#        fit = pystan.stan(file=self.diffcovs_model_file, data=d, iter=self.iters, seed=self.seed+1, chains=self.chains, verbose=True, init=[init_d for i in xrange(self.chains)])
#        fit = pystan.stan(file=self.diffcovs_model_file, data=d, iter=self.iters, seed=self.seed+1, chains=self.chains, verbose=True, init = [init_d for x in xrange(self.chains)], warmup=0)
        fit = pystan.stan(file=self.diffcovs_model_file, data=d, iter=self.iters, seed=self.seed+1, chains=self.chains, verbose=True)
        print '\t\t\t\t\tstart'
        import sys
        sys.stdout.flush()


        traces = fit.extract(permuted=True)

        # need to convert arrays to dataframes, and give them the same indicies as in data

        posteriors = fixed_phi_m_posterior({})

        #posteriors['phi_m'] = pandas.DataFrame(traces['phi_m'])
        #posteriors['phi_m'].columns = ['phi_m']

        # also extract the A_i,B_i,C_i parameters
        N = len(data)
        d_A, d_B, d_C = {}, {}, {}
        num_samples = traces['as'].shape[0]
        if patient_K < num_samples:
            to_keep = [int(z*num_samples/patient_K) for z in range(patient_K)]
        else:
            to_keep = range(num_samples)
        for i in xrange(N):
            try:
                d_A[data[i].pid] = pandas.Series(fit.extract()['as'][to_keep,i])
                d_B[data[i].pid] = pandas.Series(fit.extract()['bs'][to_keep,i])
                d_C[data[i].pid] = pandas.Series(fit.extract()['cs'][to_keep,i])
            except:
                d_A[data[i].pid] = pandas.Series(fit.extract()['as'][to_keep])
                d_B[data[i].pid] = pandas.Series(fit.extract()['bs'][to_keep])
                d_C[data[i].pid] = pandas.Series(fit.extract()['cs'][to_keep])

        posteriors['As'] = pandas.DataFrame(d_A)
        posteriors['Bs'] = pandas.DataFrame(d_B)
        posteriors['Cs'] = pandas.DataFrame(d_C)

        posteriors['phi_m'] = self.phi_m

        
        print '\t\t\t\t\tfinish'
        import sys
        sys.stdout.flush()


        return posteriors


class get_pystan_curve_fit_truncated_normal_noise_posterior_f(possibly_cached_folder):
    """
    returns posteriors for diffcovs model, using pystan instead of rstan
    """
    
    def get_introspection_key(self):
        return '%s_%.2f_%d_%d_%d' % ('pydiffcovs_curve_fit_truncatednormalnoise', self.l_m, self.iters, self.chains, self.seed)
        
    def key_f(self, data):
        return '%s_%s' % (self.get_key(), data.get_key())

    def location_f(self, data):
        return '%s/%s' % (data.get_location(), 'pytrained_diffcovs')

    print_handler_f = staticmethod(folder_adapter(write_posterior_traces))

    read_f = staticmethod(read_posterior_traces)

#    def has_file_content(self, data):
#        return os.path.exists('%s/%s' % (self.full_file_path_f(data), 'out_B_a.csv'))

    def __init__(self, l_m, cs_l, iters, chains, seed):
        self.l_m, self.cs_l, self.iters, self.chains, self.seed = l_m, cs_l, iters, chains, seed
        self.diffcovs_model_file = '%s/%s/%s' % (global_stuff.home, 'recovery_curve/stan_files', 'curve_fit_truncated_normal_noise.stan')

    pickle_lock = multiprocessing.Lock()
    extract_lock = multiprocessing.Lock()

    #@save_and_memoize
    @key
#    @read_from_pickle
    #@save_to_file
    @save_to_pickle
    def __call__(self, data):
        """
        need to convert data to proper form.  also need to convert to my form, which is a dictionary of dataframes, where keys are the original function objects
        """
        import pystan

        d = {}

        """
        setting hyperparameters
        """
        pdb.set_trace()
        d['l_m'] = self.l_m
        d['cs_l'] = self.cs_l
        d['N'] = len(data)

        d['ss'] = [a_datum.s for a_datum in data]

        ls = reduce(lambda x, a_datum: x + [len(a_datum.ys)], data, [])
        ts = reduce(lambda x, a_datum: x + a_datum.ys.index.tolist(), data, [])
        vs = reduce(lambda x, a_datum: x + a_datum.ys.tolist(), data, [])
        d['ls'] = ls
        d['ts'] = ts
        d['vs'] = vs
        d['L'] = len(ts)
        assert len(ts) == sum(ls)

        abc_d = {}
        for _datum in data:
            abc_d[_datum.pid] = pandas.Series(get_curve_abc(_datum.s, _datum.ys), index=['a','b','c'])

        abc_df = pandas.DataFrame(abc_d)

        init_d = {}
        init_d['as'] = abc_df.loc['a',:]
        init_d['bs'] = abc_df.loc['b',:]
        init_d['cs'] = abc_df.loc['c',:]
                                             


        fit = pystan.stan(file=self.diffcovs_model_file, data=d, iter=self.iters, seed=self.seed+1, chains=self.chains, verbose=True, init=[init_d for i in xrange(self.chains)])


        print '\t\t\t\t\tstart'
        import sys
        sys.stdout.flush()


        traces = fit.extract(permuted=True)
        
        # need to convert arrays to dataframes, and give them the same indicies as in data
        
        posteriors = keyed_dict({})

        posteriors['phi_m'] = pandas.DataFrame(traces['phi_m'])
        posteriors['phi_m'].columns = ['phi_m']

        # also extract the A_i,B_i,C_i parameters
        N = len(data)
        d_A, d_B, d_C = {}, {}, {}
        num_samples = len(traces['phi_m'])
        if patient_K < num_samples:
            to_keep = [int(z*num_samples/patient_K) for z in range(patient_K)]
        else:
            to_keep = range(num_samples)
        for i in xrange(N):
            d_A[data[i].pid] = pandas.Series(fit.extract()['as'][to_keep,i])
            d_B[data[i].pid] = pandas.Series(fit.extract()['bs'][to_keep,i])
            d_C[data[i].pid] = pandas.Series(fit.extract()['cs'][to_keep,i])
        posteriors['As'] = pandas.DataFrame(d_A)
        posteriors['Bs'] = pandas.DataFrame(d_B)
        posteriors['Cs'] = pandas.DataFrame(d_C)

        
        print '\t\t\t\t\tfinish'
        import sys
        sys.stdout.flush()


        return posteriors


class get_pystan_simple_hierarchical_beta_regression_f(possibly_cached_folder):

    def get_introspection_key(self):
        import string
        args = [self.pop_val, self.l, self.l_m, self.c, self.iters, self.chains, self.seed]
        return string.join([self.get_object_key(arg) for arg in args], sep='_')

    def __init__(self, pop_val, l, l_m, c, iters, chains, seed):
        self.pop_val, self.l, self.l_m, self.c = pop_val, l, l_m, c
        self.model_file = '%s/%s/%s' % (global_stuff.home, 'recovery_curve/stan_files', 'simple_beta_hierarchical_model.stan')
        self.iters, self.chains, self.seed = iters, chains, seed

    def key_f(self, data):
        return '%s_%s' % (self.get_key(), data.get_key())

    def location_f(self, data):
        return '%s/%s' % (data.get_location(), 'pytrained_diffcovs')

    @key
    @read_from_pickle
    @save_to_pickle
    def __call__(self, data):
        """
        data attributes: s(initial value), x(covariates), ys(observations) 
        """
        import pystan

        d = {}

        d['pop_val'] = self.pop_val
        d['c'] = self.c
        d['l'] = self.l
        d['l_m'] = self.l_m

        d['N'] = len(data)
        _a_datum = iter(data).next()
        d['K'] = len(_a_datum.x)
        xs = pandas.DataFrame({a_datum.pid:a_datum.x for a_datum in data})
        d['xs'] = xs.T
        ls = reduce(lambda x, a_datum: x + [len(a_datum.ys)], data, [])
        vs = reduce(lambda x, a_datum: x + a_datum.ys.tolist(), data, [])
        d['ls'] = ls
        d['vs'] = vs
        d['L'] = len(vs)

        fit = pystan.stan(file=self.model_file, data=d, iter=self.iters, seed=self.seed+1, chains=self.chains, verbose=True)

        traces = fit.extract(permuted=True)

        posteriors = keyed_dict({})
        posteriors['B'] = pandas.DataFrame(traces['B'])
        posteriors['phi'] = pandas.DataFrame(traces['phi'])
        posteriors['phi_m'] = pandas.DataFrame(traces['phi_m'])

        N = len(data)
        d_Z = {}
        for i in xrange(N):
            d_Z[data[i].pid] = pandas.Series(fit.extract()['zs'][:,i])
        posteriors['Zs'] = pandas.DataFrame(d_Z)

        return posteriors

class get_pystan_diffcovs_truncated_normal_posterior_f(possibly_cached_folder):
    """
    returns posteriors for diffcovs model, using pystan instead of rstan
    """
    
    def get_introspection_key(self):
        return '%s_%s_%s_%d_%d_%d' % ('pydiffcovs_trunnorm', self.get_pops_f.get_key(), self.hypers.get_key(), self.iters, self.chains, self.seed)
        
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
        self.diffcovs_model_file = '%s/%s/%s' % (global_stuff.home, 'recovery_curve/stan_files', 'full_model_diffcovs_truncated.stan')

    pickle_lock = multiprocessing.Lock()
    extract_lock = multiprocessing.Lock()

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

        """
        setting hyperparameters
        """

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



        fit = pystan.stan(file=self.diffcovs_model_file, data=d, iter=self.iters, seed=self.seed+1, chains=self.chains, verbose=True)

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
        N = len(data)
        d_A, d_B, d_C = {}, {}, {}
        num_samples = len(traces['B_a'])
        if patient_K < num_samples:
            to_keep = [int(z*num_samples/patient_K) for z in range(patient_K)]
        else:
            to_keep = range(num_samples)
        for i in xrange(N):
            d_A[data[i].pid] = pandas.Series(fit.extract()['as'][to_keep,i])
            d_B[data[i].pid] = pandas.Series(fit.extract()['bs'][to_keep,i])
            d_C[data[i].pid] = pandas.Series(fit.extract()['cs'][to_keep,i])
        posteriors['As'] = pandas.DataFrame(d_A)
        posteriors['Bs'] = pandas.DataFrame(d_B)
        posteriors['Cs'] = pandas.DataFrame(d_C)

        return posteriors



class get_pystan_diffcovs_with_test_posterior_f(possibly_cached_folder):
    """
    returns posteriors for diffcovs model, using pystan instead of rstan
    includes the test samples
    """
    
    def get_introspection_key(self):
        return '%s_%s_%s_%d_%d_%d' % ('pydiffcovs_test', self.get_pops_f.get_key(), self.hypers.get_key(), self.iters, self.chains, self.seed)
        
    def key_f(self, data, test_data):
        return '%s_%s' % (self.get_key(), data.get_key())

    def location_f(self, data, test_data):
        return '%s/%s/%s' % (data.get_location(), 'pytrained_diffcovs', self.get_pops_f.get_key())

    print_handler_f = staticmethod(folder_adapter(write_posterior_traces))

    read_f = staticmethod(read_posterior_traces)

    def has_file_content(self, data, test_data):
        return os.path.exists('%s/%s' % (self.full_file_path_f(data), 'out_B_a.csv'))

    def __init__(self, get_pops_f, hypers, iters, chains, seed):
        self.get_pops_f, self.hypers, self.iters, self.chains, self.seed = get_pops_f, hypers, iters, chains, seed
        self.diffcovs_model_file = '%s/%s/%s' % (global_stuff.home, 'recovery_curve/stan_files', 'full_model_diffcovs_has_test.stan')

    pickle_lock = multiprocessing.Lock()
    extract_lock = multiprocessing.Lock()

    #@save_and_memoize
    @key
    @read_from_pickle
    #@save_to_file
    @save_to_pickle
    def __call__(self, data, test_data):
        """
        need to convert data to proper form.  also need to convert to my form, which is a dictionary of dataframes, where keys are the original function objects
        """
        import pystan

        d = {}

        """
        setting hyperparameters
        """

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

        """
        setting test data
        """


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


        """
        setting test data
        """


        d['N_test'] = len(test_data)

        xas_test = pandas.DataFrame({a_datum.pid:a_datum.xa for a_datum in test_data})
        xbs_test = pandas.DataFrame({a_datum.pid:a_datum.xb for a_datum in test_data})
        xcs_test = pandas.DataFrame({a_datum.pid:a_datum.xc for a_datum in test_data})
        d['xas_test'] = xas_test.T.as_matrix()
        d['xbs_test'] = xbs_test.T.as_matrix()
        d['xcs_test'] = xcs_test.T.as_matrix()

        d['ss_test'] = [a_datum.s for a_datum in test_data]

        ls_test = reduce(lambda x, a_datum: x + [len(a_datum.ys)], test_data, [])
        ts_test = reduce(lambda x, a_datum: x + a_datum.ys.index.tolist(), test_data, [])
        vs_test = reduce(lambda x, a_datum: x + a_datum.ys.tolist(), test_data, [])
        d['ls_test'] = ls_test
        d['ts_test'] = ts_test
        d['vs_test'] = vs_test
        d['L_test'] = len(ts_test)
        assert len(ts_test) == sum(ls_test)



        fit = pystan.stan(file=self.diffcovs_model_file, data=d, iter=self.iters, seed=self.seed+1, chains=self.chains, verbose=True)

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

        # also extract the A_i,B_i,C_i parameters.  keep training and test parameters separate
        # extract those for training data
        N = len(data)
        d_A, d_B, d_C = {}, {}, {}
        num_samples = len(traces['B_a'])
        if patient_K < num_samples:
            to_keep = [int(z*num_samples/patient_K) for z in range(patient_K)]
        else:
            to_keep = range(num_samples)

        for i in xrange(N):
            d_A[data[i].pid] = pandas.Series(fit.extract()['as'][to_keep,i])
            d_B[data[i].pid] = pandas.Series(fit.extract()['bs'][to_keep,i])
            d_C[data[i].pid] = pandas.Series(fit.extract()['cs'][to_keep,i])
        posteriors['As'] = pandas.DataFrame(d_A)
        posteriors['Bs'] = pandas.DataFrame(d_B)
        posteriors['Cs'] = pandas.DataFrame(d_C)

        N_test = len(test_data)
        d_A_test, d_B_test, d_C_test = {}, {}, {}
        num_samples = len(traces['B_a'])
        if patient_K < num_samples:
            to_keep_test = [int(z*num_samples/patient_K) for z in range(patient_K)]
        else:
            to_keep_test = range(num_samples)

        for i in xrange(N_test):
            d_A_test[test_data[i].pid] = pandas.Series(fit.extract()['as_test'][to_keep_test, i])
            d_B_test[test_data[i].pid] = pandas.Series(fit.extract()['bs_test'][to_keep_test, i])
            d_C_test[test_data[i].pid] = pandas.Series(fit.extract()['cs_test'][to_keep_test, i])
        posteriors['As_test'] = pandas.DataFrame(d_A_test)
        posteriors['Bs_test'] = pandas.DataFrame(d_B_test)
        posteriors['Cs_test'] = pandas.DataFrame(d_C_test)

        return posteriors


class get_pystan_diffcovs_beta_noise_with_test_posterior_f(possibly_cached_folder):
    """
    returns posteriors for diffcovs model, using pystan instead of rstan
    includes the test samples
    """
    
    def get_introspection_key(self):
        return '%s_%s_%s_%d_%d_%d' % ('pydiffcovs_beta_test', self.get_pops_f.get_key(), self.hypers.get_key(), self.iters, self.chains, self.seed)
        
    def key_f(self, data, test_data):
        return '%s_%s' % (self.get_key(), data.get_key())

    def location_f(self, data, test_data):
        return '%s/%s/%s' % (data.get_location(), 'pytrained_diffcovs', self.get_pops_f.get_key())

    print_handler_f = staticmethod(folder_adapter(write_posterior_traces))

    read_f = staticmethod(read_posterior_traces)

    def has_file_content(self, data, test_data):
        return os.path.exists('%s/%s' % (self.full_file_path_f(data), 'out_B_a.csv'))

    def __init__(self, get_pops_f, hypers, iters, chains, seed):
        self.get_pops_f, self.hypers, self.iters, self.chains, self.seed = get_pops_f, hypers, iters, chains, seed
        self.diffcovs_model_file = '%s/%s/%s' % (global_stuff.home, 'recovery_curve/stan_files', 'full_model_diffcovs_beta_noise_has_test.stan')

    pickle_lock = multiprocessing.Lock()
    extract_lock = multiprocessing.Lock()

    #@save_and_memoize
    @key
    @read_from_pickle
    #@save_to_file
    @save_to_pickle
    def __call__(self, data, test_data):
        """
        need to convert data to proper form.  also need to convert to my form, which is a dictionary of dataframes, where keys are the original function objects
        """
        import pystan

        d = {}

        """
        setting hyperparameters
        """

        pops = self.get_pops_f(data)

        init_d = get_starting_pos_beta_with_test(data, test_data, pops)

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

        """
        setting test data
        """


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


        """
        setting test data
        """


        d['N_test'] = len(test_data)

        xas_test = pandas.DataFrame({a_datum.pid:a_datum.xa for a_datum in test_data})
        xbs_test = pandas.DataFrame({a_datum.pid:a_datum.xb for a_datum in test_data})
        xcs_test = pandas.DataFrame({a_datum.pid:a_datum.xc for a_datum in test_data})
        d['xas_test'] = xas_test.T.as_matrix()
        d['xbs_test'] = xbs_test.T.as_matrix()
        d['xcs_test'] = xcs_test.T.as_matrix()

        d['ss_test'] = [a_datum.s for a_datum in test_data]

        ls_test = reduce(lambda x, a_datum: x + [len(a_datum.ys)], test_data, [])
        ts_test = reduce(lambda x, a_datum: x + a_datum.ys.index.tolist(), test_data, [])
        vs_test = reduce(lambda x, a_datum: x + a_datum.ys.tolist(), test_data, [])
        d['ls_test'] = ls_test
        d['ts_test'] = ts_test
        d['vs_test'] = vs_test
        d['L_test'] = len(ts_test)
        assert len(ts_test) == sum(ls_test)



        fit = pystan.stan(file=self.diffcovs_model_file, data=d, iter=self.iters, seed=self.seed+1, chains=self.chains, verbose=False, init=[init_d for z in xrange(self.chains)])

        print '\t\t\t\t\tstart'
        import sys
        sys.stdout.flush()

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

        # also extract the A_i,B_i,C_i parameters.  keep training and test parameters separate
        # extract those for training data
        N = len(data)
        d_A, d_B, d_C = {}, {}, {}
        num_samples = len(traces['B_a'])
        if patient_K < num_samples:
            to_keep = [int(z*num_samples/patient_K) for z in range(patient_K)]
        else:
            to_keep = range(num_samples)

        for i in xrange(N):
            d_A[data[i].pid] = pandas.Series(fit.extract()['as'][to_keep,i])
            d_B[data[i].pid] = pandas.Series(fit.extract()['bs'][to_keep,i])
            d_C[data[i].pid] = pandas.Series(fit.extract()['cs'][to_keep,i])
        posteriors['As'] = pandas.DataFrame(d_A)
        posteriors['Bs'] = pandas.DataFrame(d_B)
        posteriors['Cs'] = pandas.DataFrame(d_C)

        N_test = len(test_data)
        d_A_test, d_B_test, d_C_test = {}, {}, {}
        num_samples = len(traces['B_a'])
        if patient_K < num_samples:
            to_keep_test = [int(z*num_samples/patient_K) for z in range(patient_K)]
        else:
            to_keep_test = range(num_samples)

        for i in xrange(N_test):
            d_A_test[test_data[i].pid] = pandas.Series(fit.extract()['as_test'][to_keep_test, i])
            d_B_test[test_data[i].pid] = pandas.Series(fit.extract()['bs_test'][to_keep_test, i])
            d_C_test[test_data[i].pid] = pandas.Series(fit.extract()['cs_test'][to_keep_test, i])
        posteriors['As_test'] = pandas.DataFrame(d_A_test)
        posteriors['Bs_test'] = pandas.DataFrame(d_B_test)
        posteriors['Cs_test'] = pandas.DataFrame(d_C_test)

        print '\t\t\t\t\tfinish'
        import sys
        sys.stdout.flush()

        return posteriors



class get_pystan_diffcovs_truncated_normal_with_test_posterior_f(possibly_cached_folder):
    """
    returns posteriors for diffcovs model, using pystan instead of rstan
    includes the test samples
    """
    
    def get_introspection_key(self):
        return '%s_%s_%s_%d_%d_%d' % ('pydiffcovs_truncated_normal_test', self.get_pops_f.get_key(), self.hypers.get_key(), self.iters, self.chains, self.seed)
        
    def key_f(self, data, test_data):
        return '%s_%s' % (self.get_key(), data.get_key())

    def location_f(self, data, test_data):
        return '%s/%s/%s' % (data.get_location(), 'pytrained_diffcovs', self.get_pops_f.get_key())

    print_handler_f = staticmethod(folder_adapter(write_posterior_traces))

    read_f = staticmethod(read_posterior_traces)

    def has_file_content(self, data, test_data):
        return os.path.exists('%s/%s' % (self.full_file_path_f(data), 'out_B_a.csv'))

    def __init__(self, get_pops_f, hypers, iters, chains, seed):
        self.get_pops_f, self.hypers, self.iters, self.chains, self.seed = get_pops_f, hypers, iters, chains, seed
        self.diffcovs_model_file = '%s/%s/%s' % (global_stuff.home, 'recovery_curve/stan_files', 'full_model_diffcovs_truncated_has_test.stan')

    pickle_lock = multiprocessing.Lock()
    extract_lock = multiprocessing.Lock()

    #@save_and_memoize
    @key
    @read_from_pickle
    #@save_to_file
    @save_to_pickle
    def __call__(self, data, test_data):
        """
        need to convert data to proper form.  also need to convert to my form, which is a dictionary of dataframes, where keys are the original function objects
        """
        import pystan

        d = {}

        """
        setting hyperparameters
        """

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

        """
        setting test data
        """


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


        """
        setting test data
        """


        d['N_test'] = len(test_data)

        xas_test = pandas.DataFrame({a_datum.pid:a_datum.xa for a_datum in test_data})
        xbs_test = pandas.DataFrame({a_datum.pid:a_datum.xb for a_datum in test_data})
        xcs_test = pandas.DataFrame({a_datum.pid:a_datum.xc for a_datum in test_data})
        d['xas_test'] = xas_test.T.as_matrix()
        d['xbs_test'] = xbs_test.T.as_matrix()
        d['xcs_test'] = xcs_test.T.as_matrix()

        d['ss_test'] = [a_datum.s for a_datum in test_data]

        ls_test = reduce(lambda x, a_datum: x + [len(a_datum.ys)], test_data, [])
        ts_test = reduce(lambda x, a_datum: x + a_datum.ys.index.tolist(), test_data, [])
        vs_test = reduce(lambda x, a_datum: x + a_datum.ys.tolist(), test_data, [])
        d['ls_test'] = ls_test
        d['ts_test'] = ts_test
        d['vs_test'] = vs_test
        d['L_test'] = len(ts_test)
        assert len(ts_test) == sum(ls_test)



        fit = pystan.stan(file=self.diffcovs_model_file, data=d, iter=self.iters, seed=self.seed+1, chains=self.chains, verbose=False)

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

        # also extract the A_i,B_i,C_i parameters.  keep training and test parameters separate
        # extract those for training data
        N = len(data)
        d_A, d_B, d_C = {}, {}, {}
        num_samples = len(traces['B_a'])
        if patient_K < num_samples:
            to_keep = [int(z*num_samples/patient_K) for z in range(patient_K)]
        else:
            to_keep = range(num_samples)

        for i in xrange(N):
            d_A[data[i].pid] = pandas.Series(fit.extract()['as'][to_keep,i])
            d_B[data[i].pid] = pandas.Series(fit.extract()['bs'][to_keep,i])
            d_C[data[i].pid] = pandas.Series(fit.extract()['cs'][to_keep,i])
        posteriors['As'] = pandas.DataFrame(d_A)
        posteriors['Bs'] = pandas.DataFrame(d_B)
        posteriors['Cs'] = pandas.DataFrame(d_C)

        N_test = len(test_data)
        d_A_test, d_B_test, d_C_test = {}, {}, {}
        num_samples = len(traces['B_a'])
        if patient_K < num_samples:
            to_keep_test = [int(z*num_samples/patient_K) for z in range(patient_K)]
        else:
            to_keep_test = range(num_samples)

        for i in xrange(N_test):
            d_A_test[test_data[i].pid] = pandas.Series(fit.extract()['as_test'][to_keep_test, i])
            d_B_test[test_data[i].pid] = pandas.Series(fit.extract()['bs_test'][to_keep_test, i])
            d_C_test[test_data[i].pid] = pandas.Series(fit.extract()['cs_test'][to_keep_test, i])
        posteriors['As_test'] = pandas.DataFrame(d_A_test)
        posteriors['Bs_test'] = pandas.DataFrame(d_B_test)
        posteriors['Cs_test'] = pandas.DataFrame(d_C_test)

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
#    @save_to_file
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



        fit = pystan.stan(file=self.diffcovs_model_file, data=d, iter=self.iters, seed=self.seed+1, chains=self.chains, verbose=True)

        traces = fit.extract(permuted=True)

        posteriors = fixed_phi_m_posterior({})
        N = len(data)
        d_A, d_B, d_C = {}, {}, {}
        num_samples = traces['as'].shape[0]
        if patient_K < num_samples:
            to_keep = [int(z*num_samples/patient_K) for z in range(patient_K)]
        else:
            to_keep = range(num_samples)


        for i in xrange(N):
            try:
                d_A[data[i].pid] = pandas.Series(fit.extract()['as'][to_keep,i])
                d_B[data[i].pid] = pandas.Series(fit.extract()['bs'][to_keep,i])
                d_C[data[i].pid] = pandas.Series(fit.extract()['cs'][to_keep,i])
            except:
                d_A[data[i].pid] = pandas.Series(fit.extract()['as'][to_keep])
                d_B[data[i].pid] = pandas.Series(fit.extract()['bs'][to_keep])
                d_C[data[i].pid] = pandas.Series(fit.extract()['cs'][to_keep])



        posteriors['As'] = pandas.DataFrame(d_A)
        posteriors['Bs'] = pandas.DataFrame(d_B)
        posteriors['Cs'] = pandas.DataFrame(d_C)

        
        # need to convert arrays to dataframes, and give them the same indicies as in data

        try:
            posteriors['B_a'] = pandas.DataFrame(traces['B_a'][to_keep,i])
        except:
            posteriors['B_a'] = pandas.DataFrame(traces['B_a'][to_keep])
        posteriors['B_a'].columns = _a_datum.xa.index
        
        try:
            posteriors['B_b'] = pandas.DataFrame(traces['B_b'][to_keep,i])
        except:
            posteriors['B_b'] = pandas.DataFrame(traces['B_b'][to_keep])
        posteriors['B_b'].columns = _a_datum.xb.index

        try:
            posteriors['B_c'] = pandas.DataFrame(traces['B_c'][to_keep,i])
        except:
            posteriors['B_c'] = pandas.DataFrame(traces['B_c'][to_keep])
        posteriors['B_c'].columns = _a_datum.xb.index


        posteriors['phi_a'] = pandas.DataFrame(traces['phi_a'])
        posteriors['phi_a'].columns = ['phi_a']
        posteriors['phi_b'] = pandas.DataFrame(traces['phi_b'])
        posteriors['phi_b'].columns = ['phi_b']
        posteriors['phi_c'] = pandas.DataFrame(traces['phi_c'])
        posteriors['phi_c'].columns = ['phi_c']

        posteriors['phi_m'] = self.phi_m

        #posteriors['phi_m'] = pandas.DataFrame(traces['phi_m'])
        #posteriors['phi_m'].columns = ['phi_m']

        return posteriors

class get_pystan_diffcovs_posterior_phi_m_fixed_has_test_f(possibly_cached_folder):
    """
    returns posteriors for diffcovs model, using pystan instead of rstan
    """
    
    def get_introspection_key(self):
        return '%s_%s_%s_%d_%d_%d' % ('pydiffcovs_phimfixed_has_test', self.get_pops_f.get_key(), self.hypers.get_key(), self.iters, self.chains, self.seed)
        
    def key_f(self, data, test_data):
        return '%s_%s' % (self.get_key(), data.get_key())

    def location_f(self, data, test_data):
        #print data, test_data
        return '%s/%s/%s' % (data.get_location(), 'pytrained_diffcovs', self.get_pops_f.get_key())

    print_handler_f = staticmethod(folder_adapter(write_posterior_traces))

    read_f = staticmethod(read_posterior_traces)

    def has_file_content(self, data):
        return os.path.exists('%s/%s' % (self.full_file_path_f(data), 'out_B_a.csv'))

    def __init__(self, phi_m, get_pops_f, hypers, iters, chains, seed):
        self.get_pops_f, self.hypers, self.iters, self.chains, self.seed = get_pops_f, hypers, iters, chains, seed
        self.phi_m = phi_m
        self.diffcovs_model_file = '%s/%s/%s' % (global_stuff.home, 'recovery_curve/stan_files', 'full_model_diffcovs_normal_noise_fixed_has_test.stan')

    #@save_and_memoize
    @key
#    @read_from_pickle
#    @save_to_file
    @save_to_pickle
    def __call__(self, data, test_data):
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


        d['N_test'] = len(test_data)

        xas_test = pandas.DataFrame({a_datum.pid:a_datum.xa for a_datum in test_data})
        xbs_test = pandas.DataFrame({a_datum.pid:a_datum.xb for a_datum in test_data})
        xcs_test = pandas.DataFrame({a_datum.pid:a_datum.xc for a_datum in test_data})
        d['xas_test'] = xas_test.T.as_matrix()
        d['xbs_test'] = xbs_test.T.as_matrix()
        d['xcs_test'] = xcs_test.T.as_matrix()

        d['ss_test'] = [a_datum.s for a_datum in test_data]

        ls_test = reduce(lambda x, a_datum: x + [len(a_datum.ys)], test_data, [])
        ts_test = reduce(lambda x, a_datum: x + a_datum.ys.index.tolist(), test_data, [])
        vs_test = reduce(lambda x, a_datum: x + a_datum.ys.tolist(), test_data, [])
        d['ls_test'] = ls_test
        d['ts_test'] = ts_test
        d['vs_test'] = vs_test
        d['L_test'] = len(ts_test)

        pdb.set_trace()

        fit = pystan.stan(file=self.diffcovs_model_file, data=d, iter=self.iters, seed=self.seed+1, chains=self.chains, verbose=True)

        traces = fit.extract(permuted=True)

        posteriors = fixed_phi_m_posterior({})
        N = len(data)
        d_A, d_B, d_C = {}, {}, {}
        d_A_test, d_B_test, d_C_test = {}, {}, {}
        num_samples = traces['as'].shape[0]
        if patient_K < num_samples:
            to_keep = [int(z*num_samples/patient_K) for z in range(patient_K)]
        else:
            to_keep = range(num_samples)


        for i in xrange(N):
            try:
                d_A[data[i].pid] = pandas.Series(fit.extract()['as'][to_keep,i])
                d_B[data[i].pid] = pandas.Series(fit.extract()['bs'][to_keep,i])
                d_C[data[i].pid] = pandas.Series(fit.extract()['cs'][to_keep,i])
            except:
                d_A[data[i].pid] = pandas.Series(fit.extract()['as'][to_keep])
                d_B[data[i].pid] = pandas.Series(fit.extract()['bs'][to_keep])
                d_C[data[i].pid] = pandas.Series(fit.extract()['cs'][to_keep])


        N_test = len(test_data)

        for i in xrange(N_test):
            try:
                d_A_test[test_data[i].pid] = pandas.Series(fit.extract()['as_test'][to_keep,i])
                d_B_test[test_data[i].pid] = pandas.Series(fit.extract()['bs_test'][to_keep,i])
                d_C_test[test_data[i].pid] = pandas.Series(fit.extract()['cs_test'][to_keep,i])
            except:
                d_A_test[test_data[i].pid] = pandas.Series(fit.extract()['as_test'][to_keep])
                d_B_test[test_data[i].pid] = pandas.Series(fit.extract()['bs_test'][to_keep])
                d_C_test[test_data[i].pid] = pandas.Series(fit.extract()['cs_test'][to_keep])

        posteriors['As'] = pandas.DataFrame(d_A)
        posteriors['Bs'] = pandas.DataFrame(d_B)
        posteriors['Cs'] = pandas.DataFrame(d_C)

        posteriors['As_test'] = pandas.DataFrame(d_A_test)
        posteriors['Bs_test'] = pandas.DataFrame(d_B_test)
        posteriors['Cs_test'] = pandas.DataFrame(d_C_test)

        
        # need to convert arrays to dataframes, and give them the same indicies as in data

        try:
            posteriors['B_a'] = pandas.DataFrame(traces['B_a'][to_keep,i])
        except:
            posteriors['B_a'] = pandas.DataFrame(traces['B_a'][to_keep])
        posteriors['B_a'].columns = _a_datum.xa.index
        
        try:
            posteriors['B_b'] = pandas.DataFrame(traces['B_b'][to_keep,i])
        except:
            posteriors['B_b'] = pandas.DataFrame(traces['B_b'][to_keep])
        posteriors['B_b'].columns = _a_datum.xb.index

        try:
            posteriors['B_c'] = pandas.DataFrame(traces['B_c'][to_keep,i])
        except:
            posteriors['B_c'] = pandas.DataFrame(traces['B_c'][to_keep])
        posteriors['B_c'].columns = _a_datum.xb.index


        posteriors['phi_a'] = pandas.DataFrame(traces['phi_a'])
        posteriors['phi_a'].columns = ['phi_a']
        posteriors['phi_b'] = pandas.DataFrame(traces['phi_b'])
        posteriors['phi_b'].columns = ['phi_b']
        posteriors['phi_c'] = pandas.DataFrame(traces['phi_c'])
        posteriors['phi_c'].columns = ['phi_c']

        posteriors['phi_m'] = self.phi_m

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



        fit = pystan.stan(file=self.diffcovs_model_file, data=d, iter=self.iters, seed=self.seed+1, chains=self.chains, verbose=True)

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
