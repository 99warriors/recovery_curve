from recovery_curve.management_stuff import *
from recovery_curve.prostate_specifics import *
import recovery_curve.global_stuff as gs

import multiprocessing

patient_K = gs.patient_K

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
    #@read_from_pickle
    @save_to_pickle
    def __call__(self, *args):

        import multiprocessing

        get_posterior_f_queue = multiprocessing.Queue()
        for seed in xrange(self.chains):
            get_posterior_f_queue.put(self.get_posterior_f_cons_partial(iters=self.iters, chains=1, seed=seed))

        for z in xrange(self.num_processes):
            get_posterior_f_queue.put(None)

        posteriors = multiprocessing.Manager().list()

        def worker(_get_posterior_f_queue, _posteriors, _args):
            for _get_posterior_f in iter(_get_posterior_f_queue.get, None):
                print _get_posterior_f
                print _get_posterior_f.get_key()
                _posteriors.append(_get_posterior_f(*_args))



        #for x in iter(get_posterior_f_queue.get, None):
        #    print 'zz', x
        #pdb.set_trace()
        workers = []
        for i in xrange(self.num_processes):
            p = multiprocessing.Process(target=worker, args=(get_posterior_f_queue, posteriors, args))
            p.start()
            workers.append(p)

        for p in workers:
            p.join()

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
