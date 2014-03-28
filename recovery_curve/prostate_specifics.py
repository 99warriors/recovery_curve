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

class keyed_partial(functools.partial, keyed_object):
    """
    for now, only take un-named arguments
    """
    def get_introspection_key(self):
        import string
        return '%s_%s' % (self.func.__name__, string.join([self.get_object_key(arg) for arg in self.args], sep='_'))
        






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



    


class abc_trace_plots(possibly_cached):
    """
    plot traces for a,b,c parameters use for fitting(not for prediction).  plot a,b,c for a patient on the same page
    """
    def get_introspection_key(self):
        return 'fit_abc_traces'

    def key_f(self, posteriors):
        return '%s_%s' % (self.get_key(), posteriors.get_key())

    def location_f(self, posteriors):
        return '%s/%s' % (global_stuff.for_dropbox, 'trace_plots')

    print_handler_f = staticmethod(multiple_figures_to_pdf)

    @save_to_file
    def __call__(self, posteriors):
        figs = []
        As = posteriors['As']
        Bs = posteriors['Bs']
        Cs = posteriors['Cs']
        pids = As.columns
        spacer=10
        for pid in pids:
            print pid
            fig = plt.figure()
            fig.subplots_adjust(hspace=0.6, wspace=0.3)
            a_ax = fig.add_subplot(3,1,1)
            a_ax.set_title('a_%s' % pid)
            a_ax.set_xlabel('iteration')
            a_trace = As[pid]
            num_traces = a_trace.shape[0]
            a_idx = np.arange(0,num_traces,spacer)
            a_ax.plot(a_idx, a_trace.iloc[a_idx])

            b_ax = fig.add_subplot(3,1,2)
            b_ax.set_title('b_%s' % pid)
            b_ax.set_xlabel('iteration')
            b_trace = Bs[pid]
            num_traces = b_trace.shape[0]
            b_idx = np.arange(0,num_traces,spacer)
            b_ax.plot(b_idx, b_trace.iloc[b_idx])

            c_ax = fig.add_subplot(3,1,3)
            c_ax.set_title('c_%s' % pid)
            c_ax.set_xlabel('iteration')
            c_trace = Cs[pid]
            num_traces = c_trace.shape[0]
            c_idx = np.arange(0,num_traces,spacer)
            c_ax.plot(c_idx, c_trace.iloc[c_idx])
            figs.append(fig)
        
        return figs

class summary_traces(possibly_cached):

    """
    1 roll for each B
    1 roll for phi's
    """
    def get_introspection_key(self):
        return 'fit_param_traces'

    def key_f(self, posteriors):
        return '%s_%s' % (self.get_key(), posteriors.get_key())

    def location_f(self, posteriors):
        return '%s/%s' % (global_stuff.for_dropbox, 'trace_plots')

    print_handler_f = staticmethod(multiple_figures_to_pdf)


    @save_to_file
    def __call__(self, posteriors):

        B_param_names = ['B_a','B_b','B_c']
        phi_param_names = ['phi_a', 'phi_b', 'phi_c', 'phi_m']
        all_pp_rolls = []
        spacer = 10
        for B_param_name in B_param_names:
            traces = posteriors[B_param_name]
            roll = pp_roll(3,1, hspace=0.5,wspace=0.5, title=B_param_name)
            for col_name, trace in traces.iteritems():
                ax = roll.get_axes()
                idx = np.arange(0,traces.shape[0],spacer)
                trace_to_plot = trace.iloc[idx]
                ax.plot(idx, trace_to_plot)
                ax.set_title(col_name)
            all_pp_rolls.append(roll)

        roll = pp_roll(3,1,hspace=0.5,wspace=0.5)
        for phi_param_name in phi_param_names:
            ax = roll.get_axes()
            trace = posteriors[phi_param_name]
            idx = np.arange(0,trace.shape[0],spacer)
            trace_to_plot = trace.iloc[idx]
            ax.plot(idx, trace_to_plot)
            ax.set_title(phi_param_name)
        all_pp_rolls.append(roll)
        return reduce(lambda x,y:x + y.figs, all_pp_rolls, [])




            

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
        return '%s/%s' % (global_stuff.for_dropbox, 'posterior_means')

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
        return '%s/%s' % (global_stuff.for_dropbox, 'posterior_means')

    print_handler_f = staticmethod(multiple_figures_to_pdf)

    def __init__(self, cv_f, get_posterior_f):
        self.cv_f, self.get_posterior_f = cv_f, get_posterior_f

    print_handler_f = staticmethod(multiple_figures_to_pdf)


    @save_to_file
    def __call__(self, data):
        posteriors = []
        for train_data, test_data in self.cv_f(data):
            try:
                posteriors.append(self.get_posterior_f(train_data))
            except TypeError:
                posteriors.append(self.get_posterior_f(train_data, test_data))

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
                fold_posterior_mean = fold_posterior_mean[sorted(fold_posterior_mean.index,key = lambda x:x.get_key())]
                num_feat = len(fold_posterior_mean)
                ax.set_xlim(-1,num_feat)
                ax.set_ylim(-5,5)
                ax.plot(range(num_feat), fold_posterior_mean, linestyle='None', marker='o', color='black')

                #ax.plot(range(num_feat), D.dot(fold_posterior_mean), linestyle='None', marker='o', color='blue')                
                ax.set_xticks(range(num_feat))
                ax.set_xticklabels(fold_posterior_mean.index, rotation='vertical')
                xticks = ax.get_xticklabels()
                for xtick,i in zip(xticks,range(len(xticks))):
                    xtick.set_fontsize(10)
            figs.append(fig)

        return figs

class plot_posterior_boxplots(possibly_cached):
    """
    assumes that there is only 1 covariate, no bias term
    takes in axes, returns the same axes
    """
    def __init__(self):
        self.param_names = ['B_a', 'B_b', 'B_c', 'phi_a', 'phi_b', 'phi_c', 'phi_m']


    def __call__(self, ax, posteriors, true_params):
        """
        put posteriors into list.  then draw boxplots, and add in true values and labels
        """
        from scripts.for_paper import options as options
        def do_suff(param_list, param_names, true_params):
            l = [posteriors[param_name] for param_name in self.param_names]
            ax.boxplot(param_list, sym='')

            num_params = len(param_names)
            ax.set_xticks(range(1,num_params+1))
            ax.set_xticklabels(param_names, rotation='vertical')
            xticks = ax.get_xticklabels()
            for xtick,i in zip(xticks,range(len(xticks))):
                xtick.set_fontsize(options.textsize) 

            for param_name, i in zip(param_names,xrange(num_params)):
                ax.axhline(y = true_params[param_name], xmin=i/float(num_params), xmax=(i+1)/float(num_params), color = 'green')

        return ax



def plot_posterior_boxplots_f(ax, param_list, param_names, param_display_names, true_params):
    #l = [posteriors[param_name] for param_name in param_names]
    from scripts.for_paper import options as options
    ax.boxplot(param_list, sym='')

    num_params = len(param_names)
    ax.set_xticks(range(1,num_params+1))
    ax.set_xticklabels(param_display_names, rotation='horizontal')
    xticks = ax.get_xticklabels()
    for xtick,i in zip(xticks,range(len(xticks))):
        xtick.set_fontsize(options.textsize) 

    for tick in ax.yaxis.get_major_ticks():
        tick.label.set_fontsize(options.textsize+5) 
    for param_name, i in zip(param_names,xrange(num_params)):
        ax.axhline(y = true_params[param_name], xmin=i/float(num_params), xmax=(i+1)/float(num_params), color = 'green')


class plot_diffcovs_posterior_f(possibly_cached):
    """
    takes in traces, plots histogram of training folds from cv on data.  takes in data, cv_f, get_posterior_f
    """
    def get_introspection_key(self):
        return '%s_%s_%s' % ('postplt', self.cv_f.get_key(), self.get_posterior_f.get_key())

    def key_f(self, data):
        return '%s_%s' % (self.get_key(), data.get_key())

    def location_f(self, data):
        return '%s/%s' % (global_stuff.for_dropbox, 'posterior_plots')

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
            try:
                posteriors.append(self.get_posterior_f(train_data))
            except TypeError:
                posteriors.append(self.get_posterior_f(train_data, test_data))

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
        true_vals = [-2,1,2, 0.05, 0.05, 0.05, 0.005]

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
        


class max_f(keyed_object):

    def get_introspection_key(self):
        return 'max'

    def __call__(self, l):
        return max(l)

class mean_f(keyed_object):

    def get_introspection_key(self):
        return 'meanf'

    def __call__(self, l):
        return np.mean(l)
        total = 0.0
        count = 0.0
        for x in l:
            total += x
            count += 1
        return total/count

class median_f(keyed_object):

    def get_introspection_key(self):
        return 'medianf'

    def __call__(self, l):
        return np.median(l)

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
    #@memoize
    def __call__(self, data):
        fold_scores = keyed_list()
        for train_data, test_data in self.cv_f(data):
            if self.get_predictor_f.normal():
                predictor = self.get_predictor_f(train_data)
            else:
                predictor = self.get_predictor_f(train_data, test_data)


            def _per_data_f(_datum):
                return (_datum.pid, pandas.Series({time:predictor(_datum, time) for time in self.times if time in _datum.ys.index}))
            raw = parallel_map(_per_data_f, test_data, global_stuff.num_processors)
            fold_scores.append(pandas.DataFrame({x:y for x,y in raw}))
            #fold_scores.append(pandas.DataFrame({datum.pid:{time:predictor(datum, time) for time in self.times} for datum in test_data if int(datum.pid)%1==0}))
        return keyed_DataFrame(pandas.concat(fold_scores, axis=1))



class cross_validated_scores_notsep_f(possibly_cached):
    """
    get_predictor_f is a factory for trained predictor objects, as in it does training
    returns a list of dataframes
    """
    def get_introspection_key(self):
        return '%s_%s_%s' % ('cvscorens', self.get_predictor_f.get_key(), self.cv_f.get_key())

    def key_f(self, data):
        return '%s_%s' % (self.get_key(), data.get_key())

    def location_f(self, data):
        return '%s/%s' % (global_stuff.data_home, 'scores')

    #print_handler_f = staticmethod(write_DataFrame)

    #read_f = staticmethod(read_DataFrame)

    def __init__(self, get_predictor_f, cv_f, times):
        self.get_predictor_f, self.cv_f, self.times = get_predictor_f, cv_f, times

    @key
    #@memoize
    def __call__(self, data):
        fold_scores = keyed_list()
        for train_data, test_data in self.cv_f(data):
            if self.get_predictor_f.normal():
                predictor = self.get_predictor_f(train_data)
            else:
                predictor = self.get_predictor_f(train_data, test_data)


            def _per_data_f(_datum):
                return (_datum.pid, pandas.Series({time:predictor(_datum, time) for time in self.times if time in _datum.ys.index}))
            raw = parallel_map(_per_data_f, test_data, global_stuff.num_processors)
            fold_scores.append(keyed_DataFrame({x:y for x,y in raw}))
            #fold_scores.append(pandas.DataFrame({datum.pid:{time:predictor(datum, time) for time in self.times} for datum in test_data if int(datum.pid)%1==0}))
        return fold_scores
        #return keyed_DataFrame(pandas.concat(fold_scores, axis=1))


class plot_predictions_fig_f(possibly_cached):
    """
    this function just takes in a bunch of plotters, and calls them on a axis that it creates
    """

    def __init__(self, plotters):
        self.plotters = plotters

    def __call__(self, datum):
        print datum.pid
        fig = plt.figure()
        ax = fig.add_subplot(1,1,1)
        import string

        ax.set_xlabel('time')
        ax.set_ylabel('fxn value')
        for plotter in self.plotters:
            plotter(ax, datum)
        ax.plot(datum.ys.index, datum.ys, color = 'black', linewidth=3, marker='o')
        #add_series_to_ax(datum.ys, ax, 'black', 'true', 'solid', linewidth=3)

        try:
            abc_ys = pandas.Series({time:the_f(time,datum.s,datum.true_a,datum.true_b,datum.true_c) for time in np.linspace(0,50,200)})
            add_series_to_ax(abc_ys, ax, 'green', 'sim', 'solid', linewidth=3)
        except AttributeError:
            pass

        
        a,b,c = get_curve_abc(datum.s, datum.ys)
        """
        ts = np.linspace(0,50,100)
        ys = [the_f(t,datum.s,a,b,c) for t in ts]
        ax.plot(ts, ys, color='black', linestyle='--')
        """
        cov_string = string.join(['%.2f' % v for v in datum.xa],sep=' ')

        fig.suptitle('%s %s a:%.2f b:%.2f c:%.2f'% (datum.pid, cov_string, a, b, c))

        #ax.legend()
        ax.legend(prop={'size':4})
        ax.plot(-1,datum.s,'bo')
        ax.set_xlim(-2,50)
        ax.set_ylim(-0.1, 1.1)
        #ax.set_ylim((0,datum.s))
        return fig

class plot_all_predictions_fig_f(possibly_cached):

    def get_introspection_key(self):
        return '%s_%s' % ('all_preds', self.cv_f.get_key())

    def key_f(self, data):
        return '%s_%s' % (self.get_key(), data.get_key())

    def location_f(self, data):
        return '%s/%s' % (global_stuff.for_dropbox, 'prediction_plots')

    print_handler_f = staticmethod(multiple_figures_to_pdf)

    def __init__(self, trainer_plotter_cons_tuples, cv_f):
        self.trainer_plotter_cons_tuples, self.cv_f = trainer_plotter_cons_tuples, cv_f

    @save_to_file
    def __call__(self, data):
        figs = []
        for train_data, test_data in self.cv_f(data):
            plotters = keyed_list()
            for trainer, plotter_cons in self.trainer_plotter_cons_tuples:
                if trainer.normal():
                    predictor = trainer(train_data)
                else:
                    predictor = trainer(train_data, test_data)
                plotters.append(plotter_cons(predictor))
            prediction_plotter = plot_predictions_fig_f(plotters)
            def asdf(datum):
                return (datum, prediction_plotter(datum))
            figs += parallel_map(asdf, test_data[0:40], global_stuff.num_processors)

        def get_init_bin(_datum):
            if _datum.xa.iloc[2] > 0:
                return 0
            elif _datum.xa.iloc[1] > 0:
                return 1
            elif _datum.xa.iloc[3] > 0:
                return 2
            else:
                return 3

        def get_age_bin(_datum):
            if _datum.xa.iloc[4] > 0:
                return 0
            elif _datum.xa.iloc[5] > 0:
                return 1
            else:
                return 2
            
        def cmp_datum(d1g, d2g):
            d1 = d1g[0]
            d2 = d2g[0]
            age_cmp = cmp(get_age_bin(d1), get_age_bin(d2))
            if age_cmp != 0:
                return age_cmp
            return cmp(d1.s, d2.s)
        return [y[1] for y in figs]
        gg = [y[1] for y in sorted(figs,cmp=cmp_datum)]
        
        return gg
#            figs += [[datum.s,prediction_plotter(datum)] for datum in test_data]
            #figs += [[datum.s,prediction_plotter(datum)] for datum in test_data if int(datum.pid) in [30503,30424,30376,30218,30117,30034]]
#        return [y[1] for y in sorted(figs,key=lambda x:x[0])]
                




            
class plot_scalar_prediction_fig(keyed_object):

    def get_introspection_key(self):
        return 'scalar_pred_fig'

    def __init__(self, plotters):
        self.plotters = plotters

    def __call__(self, datum):
        print datum.pid
        fig = plt.figure()
        ax = fig.add_subplot(1,1,1)
        import string

        ax.set_xlabel('time')
        ax.set_ylabel('fxn value')
        for plotter in self.plotters:
            plotter(ax, datum)

        mean_val = np.mean(datum.ys)
        ax.axvline(mean_val, color = 'black', linewidth=5)
        for y in datum.ys:
            ax.axvline(y+random.uniform(-0.05,0.05), color = 'red')
        return fig


class generic_plot_all_predictions_fig_f(possibly_cached):

    def get_introspection_key(self):
        return '%s_%s_%s' % ('generic_scalar_preds', self.cv_f.get_key(), self.single_fig_plotter.get_key())

    def key_f(self, data):
        return '%s_%s' % (self.get_key(), data.get_key())

    def location_f(self, data):
        return '%s/%s' % (global_stuff.for_dropbox, 'prediction_plots')

    print_handler_f = staticmethod(multiple_figures_to_pdf)

    def __init__(self, trainer_plotter_cons_tuples, cv_f, single_fig_plotter):
        self.trainer_plotter_cons_tuples, self.cv_f, self.single_fig_plotter = trainer_plotter_cons_tuples, cv_f, single_fig_plotter

    @save_to_file
    def __call__(self, data):
        figs = []
        for train_data, test_data in self.cv_f(data):
            plotters = keyed_list()
            for trainer, plotter_cons in self.trainer_plotter_cons_tuples:
                if trainer.normal():
                    predictor = trainer(train_data)
                else:
                    predictor = trainer(train_data, test_data)
                plotters.append(plotter_cons(predictor))
            prediction_plotter = self.single_fig_plotter(plotters)
            
            # sort by age and then initial level

            figs = map(prediction_plotter, test_data[0:20])
            #figs = parallel_map(prediction_plotter, test_data[0:20], global_stuff.num_processors)
            #figs = parallel_map(prediction_plotter, test_data, global_stuff.num_processors)
            #figs += [[datum.s,prediction_plotter(datum)] for datum in test_data]
            #figs += [prediction_plotter(datum) for datum in test_data]
            figs += [(datum,prediction_plotter(datum)) for datum in test_data]
        def cmp_datum(d1, d2):
            age_cmp = cmp(options.get_age_bin(d1), options.get_age_bin(d2))
            if age_cmp != 0:
                return age_cmp
            return cmp(d1.s, d2.s)
        return [y[1] for y in sorted(figs,cmp=cmp_datum)]

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


class beta_loss_f(keyed_object):

    def get_introspection_key(self):
        return 'beta_%.4f' % self.phi

    def __init__(self, phi):
        self.phi = phi

    def __call__(self, pred, truth):
        return -1.0 * get_beta_log_p(truth, pred, self.phi)

class loss_f(keyed_object):

    def get_introspection_key(self):
        return '%s_%s' % (self.get_true_val_f.get_key(), self.distance_f.get_key())

    def __init__(self, get_true_val_f, distance_f):
        self.get_true_val_f, self.distance_f = get_true_val_f, distance_f

    def __call__(self, _datum, time, score):
        print self.get_true_val_f(_datum, time), score, self.distance_f(self.get_true_val_f(_datum, time), score)
        return self.distance_f(self.get_true_val_f(_datum, time), score)


class prop_loss(keyed_object):

    def get_introspection_key(self):
        #return 'proploss'
        return '%s_%s' % ('proploss', self.backing_f.get_key())

    def __init__(self, backing_f):
        self.backing_f = backing_f

    def __call__(self, _datum, time, score):
        scaled_abs_loss = abs(_datum.ys[time] - score) / _datum.s
        return self.backing_f(scaled_abs_loss)


class signed_loss_raw(keyed_object):

    def get_introspection_key(self):
        return 'asdf'

    def __call__(self, x):
        return x

class scaled_loss_f(keyed_object):

    def get_introspection_key(self):
        return '%s_%s' % ('scaled_loss', self.backing_loss_f.get_key())

    def __init__(self, backing_loss_f):
        self.backing_loss_f = backing_loss_f

    def __call__(self, _datum, time, score):
        return self.backing_loss_f(_datum, time, score) / _datum.s

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
        return pred - true_val

def cv_losses(scores, loss_f, data):
    losses_d = {}
    for pid, _scores in scores.iteritems():
        temp_d = {}
        print pid
        for time, score in _scores.iteritems():
            if time in data.d[pid].ys.index:
                temp_d[time] = loss_f(data.d[pid], time, score)
        losses_d[pid] = pandas.Series(temp_d)
    losses = pandas.DataFrame(losses_d)
    return losses

class performance_series_f(possibly_cached):
    """
    returns a dataframe.  mean score will be the first column
    """

    def __init__(self, loss_f, percentiles, data, times):
        self.loss_f, self.percentiles, self.data, self.times = loss_f, percentiles, data, times

    #@key
#    @save_and_memoize
    def __call__(self, scores):

        losses_d = {}
        for pid, _scores in scores.iteritems():
            temp_d = {}
            print pid
            #print scores
            #print self.data.d[pid].ys
            for time, score in _scores.iteritems():
                if time in self.data.d[pid].ys.index:
                    temp_d[time] = self.loss_f(self.data.d[pid], time, score)
                    print temp_d[time]
            losses_d[pid] = pandas.Series(temp_d)
        losses = pandas.DataFrame(losses_d)

        #losses = pandas.DataFrame({pid:{time:self.loss_f(self.data.d[pid], time, score) for time, score in scores.iteritems() if not np.isnan(score)} for pid, scores in scores.iteritems()})


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


def strat_perf_figs(trainers, cv_f, loss_f, percentiles, trainer_colors, trainer_names, data):
    """
    for each of the 12 possible patients, plots the loss over time.  do this agewise and initwise
    """
    init_figs = []
    """
    get the list of pids for each kind
    """
    from scripts.for_paper import options as options
    bin_pids = {(i,j):[] for i in xrange(options.num_init_bins) for j in xrange(options.num_age_bins)}
    from scripts.for_paper import options as options
    for datum in data:
        bin_pids[(options.get_init_bin(datum), options.get_age_bin(datum))].append(datum.pid)

    """
    get the losses for each trainer
    """
    scores = [cross_validated_scores_f(trainer, cv_f, options.real_times)(data) for trainer in trainers]
    losses = [cv_losses(score, loss_f, data) for score in scores]
    
    """
    stratify losses by category
    """
    bin_losses = [{(i,j):loss.loc[:,[pid in bin_pids[(i,j)] for pid in loss.columns]] for i in xrange(options.num_init_bins) for j in xrange(options.num_age_bins)} for loss in losses]

    """
    now, go through each patient type, and figure out what to plot for each losses
    """
    initwise_figs = []
    for i in xrange(options.num_init_bins):
        fig = plt.figure()
        fig.suptitle('age %d' % i)
        for j in xrange(options.num_age_bins):
            ax = fig.add_subplot(2, 2, j+1)
            ax.set_title('age %d size %d' % (j,len(bin_pids[(i,j)])))
            for bin_loss, trainer_color, trainer_name in zip(bin_losses, trainer_colors, trainer_names):
                mean_loss = bin_loss[(i,j)].apply(np.mean, axis=1)
                ax.plot(mean_loss.index, mean_loss, color=trainer_color, label=trainer_name)
            ax.legend(prop={'size':5})
        initwise_figs.append(fig)

    agewise_figs = []
    for j in xrange(options.num_age_bins):
        fig = plt.figure()
        fig.suptitle('age %d' % j)
        for i in xrange(options.num_init_bins):
            ax = fig.add_subplot(2, 2, i+1)
            ax.set_title('init %d size %d' % (j,len(bin_pids[(i,j)])))
            for bin_loss, trainer_color, trainer_name in zip(bin_losses, trainer_colors, trainer_names):
                mean_loss = bin_loss[(i,j)].apply(np.mean, axis=1)
                ax.plot(mean_loss.index, mean_loss, color=trainer_color, label=trainer_name)
            ax.legend(prop={'size':5})
        agewise_figs.append(fig)

    return initwise_figs, agewise_figs



def plot_real_trends(trainer, data):

    def avg_curves(l):
        df = pandas.DataFrame({i:l[i] for i in xrange(len(l))})
        return df.apply(np.mean, axis=1)

    from scripts.for_paper import options as options

    if trainer.normal():
        predictor = trainer(data)
    else:
        predictor = trainer(data, data)

    repr_d = {}
    for datum in data:
        repr_d[(options.get_init_bin(datum),options.get_age_bin(datum))] = datum


    pred_d = {key:pandas.Series([predictor(datum,t) for t in curve_plot_times], index=curve_plot_times)/datum.s for key,datum in repr_d.iteritems()}

    pred_by_init = [[pred_d[(j,i)] for i in xrange(options.num_age_bins)] for j in xrange(options.num_init_bins)]
    agg_by_init = map(avg_curves, pred_by_init)

    pred_by_age = [[pred_d[(j,i)] for j in xrange(options.num_init_bins)] for i in xrange(options.num_age_bins)]
    agg_by_age = map(avg_curves, pred_by_age)


    """
    show trend in init
    """

    init_fig = plt.figure()
    init_fig.suptitle(options.run_name)
    ax = plt.subplot2grid((1,2),(0,0))
    for curve, label, color in zip(agg_by_init, options.init_titles, options.init_colors):
        ax.plot(curve.index, curve, label = label, color=color)
    ax.set_xlabel('months')
    ax.set_ylabel('scaled f')
    ax.set_title('aggregate curve by init')
    ax.set_ylim((0,1))
    ax.legend(prop={'size':5},loc=4)
    for i, age_label in zip(xrange(options.num_age_bins), options.age_titles):
        ax = plt.subplot2grid((options.num_age_bins,2),(i,1))
        ax.set_title(age_label)
        for j,color in zip(xrange(options.num_init_bins), options.init_colors):
            curve = pred_d[(j,i)]
            ax.plot(curve.index, curve, color=color)

    plt.tight_layout()


    """
    show trend in age
    """

    age_fig = plt.figure()
    age_fig.suptitle(options.run_name)
    ax = plt.subplot2grid((1,2),(0,0))
    for curve, label, color in zip(agg_by_age, options.age_titles, options.age_colors):
        ax.plot(curve.index, curve, label = label, color = color)
    ax.set_xlabel('months')
    ax.set_ylabel('scaled f')
    ax.set_ylim((0,1))
    ax.set_title('aggregate curve by age')
    ax.legend(prop={'size':5},loc=4)
    for i, init_label in zip(xrange(options.num_init_bins), options.init_titles):
        ax = plt.subplot2grid((options.num_init_bins,2),(i,1))
        ax.set_title(init_label)
        for j, color in zip(xrange(options.num_age_bins), options.age_colors):
            curve = pred_d[(i,j)]
            ax.plot(curve.index, curve, color=color)

    plt.tight_layout()        

    return init_fig, age_fig

class model_comparer_f(possibly_cached):
    """
    hard code the loss functions used
    """

    def __init__(self, trainers, cv_f, loss_f, percentiles, times, display_colors, display_names, filter_f, y_max = None):
        self.trainers, self.cv_f, self.loss_f, self.percentiles, self.times = trainers, cv_f, loss_f, percentiles, times
        self.display_colors, self.display_names = display_colors, display_names
        self.filter_f = filter_f
        self.y_max = y_max

    #@save_to_file
    #@memoize
    def __call__(self, data):
        from scripts.for_paper import options as options
        fig = plt.figure()
        ax = fig.add_subplot(1,1,1)
        ax.plot([0],[0])
        #fig.suptitle('overall loss under %s, n=%d' % (self.loss_f.get_key(), len(data)))
        if self.y_max != None:
            ax.set_ylim((0,self.y_max))
        for trainer, display_color, display_name in itertools.izip(self.trainers, self.display_colors, self.display_names):
            scores_getter = cross_validated_scores_f(trainer, self.cv_f, self.times)
            scores = scores_getter(data)
            key = scores.hard_coded_key
            loc = scores.location
            #scores = keyed_DataFrame(scores.iloc[:,[self.filter_f(x) for x in scores.columns]])
            #scores.hard_coded_key = key
            #scores.location = loc
            _performance_series_f = performance_series_f(self.loss_f, self.percentiles, data, self.times)
            perfs = _performance_series_f(scores)
            add_performances_to_ax(ax, perfs, display_color, display_name)
            for tick in ax.xaxis.get_major_ticks():
                tick.label.set_fontsize(options.textsize)
            for tick in ax.yaxis.get_major_ticks():
                tick.label.set_fontsize(options.textsize)  
        """
        also add in the scores that come from predicting the median of the filtered dataset
        """
        """
        filtered_ys = pandas.DataFrame({x:data.d[x].ys for x in scores.columns if self.filter_f(x)})
        medians = filtered_ys.apply(np.median, axis=1)
        losses_d = {}
        for pid, _ys in filtered_ys.iteritems():
            temp_d = {}
            #print pid
            for time, score in _ys.iteritems():

                if time in data.d[pid].ys.index:
                    temp_d[time] = self.loss_f(data.d[pid], time, medians[time])
                
            losses_d[pid] = pandas.Series(temp_d)
        losses = pandas.DataFrame(losses_d)
        median_losses = losses.apply(np.median, axis=1)
        """
        #ax.plot(median_losses.index, median_losses, color='magenta')


        ax.set_xlim(-1.0,50)

        ax.set_xlabel('months', fontsize = options.textsize)
        ax.set_ylabel('loss', fontsize = options.textsize)
        ax.legend(prop={'size':options.textsize}, loc=4)
        return fig

    def get_introspection_key(self):
        return 'performances'
        return '%s_%s_%s' % (self.loss_f.get_key(), self.cv_f.get_key(), self.trainers[0].get_key())
        return '%s_%s_%s_%s' % ('bs', self.trainers.get_key(), self.cv_f.get_key(), self.loss_f.get_key())

    def key_f(self, data):
        return '%s_%s' % (self.get_key(), data.get_key())

    def location_f(self, data):
        return '%s/%s' % (global_stuff.for_dropbox,'betweenmodel_perfs')

    print_handler_f = staticmethod(figure_to_pdf)

    read_f = staticmethod(not_implemented_f)

class model_comparer_cv_f(possibly_cached):
    """
    hard code the loss functions used
    for each trainer, plot the performance from each fold
    """

    def __init__(self, trainers, cv_f, loss_f, percentiles, times, display_colors, display_names, show_folds, filter_f):
        self.trainers, self.cv_f, self.loss_f, self.percentiles, self.times, self.show_folds = trainers, cv_f, loss_f, percentiles, times, show_folds
        self.display_colors, self.display_names = display_colors, display_names
        self.filter_f = filter_f

    #@save_to_file
    #@memoize
    def __call__(self, data):
        from scripts.for_paper import options as options
        textsize = options.textsize
        linewidth = options.linewidth
        fig = plt.figure()
        ax = fig.add_subplot(1,1,1)
        ax.plot([0],[0])
        #fig.suptitle('overall loss under %s, n=%d' % (self.loss_f.get_key(), len(data)))
        for trainer, display_color, display_name in itertools.izip(self.trainers, self.display_colors, self.display_names):
            scores_getter = cross_validated_scores_notsep_f(trainer, self.cv_f, self.times)
            scores = scores_getter(data)
            key = scores.hard_coded_key
            loc = scores.location
            #scores = keyed_DataFrame(scores.iloc[:,[self.filter_f(x) for x in scores.columns]])
            #scores.hard_coded_key = key
            #scores.location = loc

            mean_scores = pandas.DataFrame({i:performance_series_f(self.loss_f, [], data, self.times)(fold_score)['mean'] for i,fold_score in list(enumerate(scores))})
            means = mean_scores.apply(pandas.Series.mean, axis=1)
            stds = mean_scores.apply(pandas.Series.std, axis=1)

            assert means.shape == stds.shape

            for tick in ax.yaxis.get_major_ticks():
                tick.label.set_fontsize(textsize) 

            for tick in ax.xaxis.get_major_ticks():
                tick.label.set_fontsize(textsize) 


            ax.errorbar(self.times, means, yerr = stds, label = display_name, color = display_color, linewidth=linewidth, capthick=linewidth, alpha = options.alpha, capsize = 6)

            if self.show_folds:
                for fold_score in scores:
                    _performance_series_f = performance_series_f(self.loss_f, self.percentiles, data, self.times)
                    perfs = _performance_series_f(fold_score)
                    add_performances_to_ax(ax, perfs, display_color, display_name)
        """
        also add in the scores that come from predicting the median of the filtered dataset
        """
        """
        filtered_ys = pandas.DataFrame({x:data.d[x].ys for x in scores.columns if self.filter_f(x)})
        medians = filtered_ys.apply(np.median, axis=1)
        losses_d = {}
        for pid, _ys in filtered_ys.iteritems():
            temp_d = {}
            #print pid
            for time, score in _ys.iteritems():

                if time in data.d[pid].ys.index:
                    temp_d[time] = self.loss_f(data.d[pid], time, medians[time])
                
            losses_d[pid] = pandas.Series(temp_d)
        losses = pandas.DataFrame(losses_d)
        median_losses = losses.apply(np.median, axis=1)
        """
        #ax.plot(median_losses.index, median_losses, color='magenta')


        ax.set_xlim(-1.0,50)

        ax.set_xlabel('months', fontsize = textsize)
        ax.set_ylabel('loss', fontsize = textsize)
        ax.legend(prop={'size':textsize}, loc=4)
        return fig

    def get_introspection_key(self):
        return 'performances_cv'
        return '%s_%s_%s' % (self.loss_f.get_key(), self.cv_f.get_key(), self.trainers[0].get_key())
        return '%s_%s_%s_%s' % ('bs', self.trainers.get_key(), self.cv_f.get_key(), self.loss_f.get_key())

    def key_f(self, data):
        return '%s_%s' % (self.get_key(), data.get_key())

    def location_f(self, data):
        return '%s/%s' % (global_stuff.for_dropbox,'betweenmodel_perfs')

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

def plot_phis_on_fig(posterior, name):

    fig = plt.figure()
    fig.suptitle(name)
    
    as_ax = fig.add_subplot(2,2,1)
    bs_ax = fig.add_subplot(2,2,2)
    cs_ax = fig.add_subplot(2,2,3)
    as_ax.hist(posterior['phi_a'].iloc[:,0], bins=20)
    as_ax.set_title('$phi_a$')
    bs_ax.hist(posterior['phi_b'].iloc[:,0], bins=20)
    bs_ax.set_title('$phi_b$')
    cs_ax.hist(posterior['phi_c'].iloc[:,0], bins=20)
    cs_ax.set_title('$phi_c$')

    return fig

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
most general: specify (something that generates output, something that plots the output
"""





"""
what 'data' means for this project
"""

class datum_base(keyed_object):

    def __repr__(self):
        return self.pid

    def __eq__(self, other):
        if not isinstance(other, datum_base):
            return False
        else:
            return self.pid == other.pid

    def __hash__(self):
        return hash(self.pid)

class datum(datum_base):

    def __init__(self, pid, xa, xb, xc, s, ys):
        self.pid, self.xa, self.xb, self.xc, self.s, self.ys = pid, xa, xb, xc, s, ys



class simulated_datum(datum):

    def __init__(self, pid, xa, xb, xc, s, ys, true_a, true_b, true_c):
        self.true_a, self.true_b, self.true_c = true_a, true_b, true_c
        datum.__init__(self, pid, xa, xb, xc, s, ys)


class datum_for_asymptotic(datum_base):

    def __init__(self, pid, x, s, ys):
        self.pid, self.s, self.x, self.ys = pid, s, x, ys




class data(keyed_list):
    """
    also offer dictionary[pid] capability
    """
    def __init__(self, l):
        self.d = {_datum.pid:_datum for _datum in l}
        keyed_list.__init__(self, l)

    def get_introspection_key(self):
        return 'data'


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








class hypers(keyed_object):
    """
    key is hard coded
    """
    def get_introspection_key(self):
        return '%.2f_%.2f' % (self.c_a, self.l_a)
        return '%.2f_%.2f_%.2f_%.2f_%.2f_%.2f_%.2f' % (self.c_a, self.c_b, self.c_c, self.l_a, self.l_b, self.l_c, self.l_m)
            
    def __init__(self, c_a, c_b, c_c, l_a, l_b, l_c, l_m):
        self.c_a, self.c_b, self.c_c, self.l_a, self.l_b, self.l_c, self.l_m = c_a, c_b, c_c, l_a, l_b, l_c, l_m






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
    

def get_abc_scatter(f, title, f_name, data, xticks):
    """
    for a,b,c, plot the scalar feature f on the x-axis
    have a dataframe whose columns are f value, a, b, c
    """
    import matplotlib.pyplot as pyplot

    fig = plt.figure()
    #fig.suptitle(title)

    y_min, y_max = -0.05, 1.05

    d = {}

    for _datum in data:
        a,b,c = get_curve_abc(_datum.s, _datum.ys)
        f_val = f(_datum.pid)
        d[_datum.pid] = pandas.Series({'f':f_val, 'a':a, 'b':b, 'c':c})

    fabc_df = pandas.DataFrame(d)
    from scripts.for_paper import options as options    
    ax = fig.add_subplot(2,2,1)

    textsize = options.textsize

    ax.scatter(fabc_df.loc['f'],fabc_df.loc['a'])
    ax.set_title('%s vs A' % f_name, fontsize = options.textsize)
    ax.set_ylim((y_min,y_max))

    for tick in ax.xaxis.get_major_ticks():
        tick.label.set_fontsize(textsize) 

    for tick in ax.yaxis.get_major_ticks():
        tick.label.set_fontsize(textsize) 

    #pyplot.locator_params(nbins=5)

    #ax.set_yticks(yticks)

    ax.set_xlabel(f_name, fontsize = textsize)
    ax.set_ylabel('A', fontsize = textsize, rotation = 'horizontal')

    ax.set_xticks(xticks)

    ax = fig.add_subplot(2,2,2)
    ax.scatter(fabc_df.loc['f'],fabc_df.loc['b'])
    ax.set_title('%s vs B' % f_name, fontsize = options.textsize)
    ax.set_ylim((y_min,y_max))

    for tick in ax.xaxis.get_major_ticks():
        tick.label.set_fontsize(textsize) 

    for tick in ax.yaxis.get_major_ticks():
        tick.label.set_fontsize(textsize) 

    #pyplot.locator_params(nbins=5)

    #ax.set_yticks(yticks)
    ax.set_xticks(xticks)

    ax.set_xlabel(f_name, fontsize = textsize)
    ax.set_ylabel('B', fontsize = textsize, rotation = 'horizontal')

    ax = fig.add_subplot(2,2,3)
    ax.scatter(fabc_df.loc['f'],fabc_df.loc['c'])
    ax.set_title('%s vs C' % f_name, fontsize = options.textsize)
    #ax.set_ylim((y_min,y_max))
    ax.set_yscale('log')
    for tick in ax.xaxis.get_major_ticks():
        tick.label.set_fontsize(textsize) 

    for tick in ax.yaxis.get_major_ticks():
        tick.label.set_fontsize(textsize) 

    #pyplot.locator_params(nbins=5)

    #ax.set_yticks(yticks)
    ax.set_xticks(xticks)

    ax.set_xlabel(f_name, fontsize = textsize)
    ax.set_ylabel('C', fontsize = textsize, rotation = 'horizontal')

    plt.tight_layout()

    return fig


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
        return '%s/%s' % (global_stuff.for_dropbox, 'aggregate_curves')

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
        return '%s/%s' % (global_stuff.for_dropbox, 'aggregate_shapes')

    print_handler_f = staticmethod(write_Series)

    read_f = staticmethod(read_Series)

    @key
    def __call__(self, data):
        all_ys = pandas.DataFrame({datum.pid:datum.ys/datum.s for datum in data})
        mean_shape = all_ys.apply(np.mean, axis=1)
        return keyed_Series(mean_shape)

class median_shape_f(possibly_cached):

    def get_introspection_key(self):
        return 'med_shape'

    def key_f(self, data):
        return '%s_%s' % (self.get_key(), data.get_key())

    def location_f(self, data):
        return '%s/%s' % (global_stuff.for_dropbox, 'aggregate_shapes')

    print_handler_f = staticmethod(write_Series)

    read_f = staticmethod(read_Series)

    @key
    def __call__(self, data):
        all_ys = pandas.DataFrame({datum.pid:datum.ys/datum.s for datum in data})
        med_shape = all_ys.apply(nan_median, axis=1)
        return keyed_Series(med_shape)


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
    from scripts.for_paper import options as options
    #add_series_to_ax(perfs['mean'], ax, color, name, '--')
    mean = perfs['mean']
    ax.plot(mean.index, mean, color=color, label=name, linewidth=options.linewidth, alpha=options.alpha, linestyle='solid')
    percentiles = perfs[[x for x in perfs.columns if x != 'mean']]
    fixed = functools.partial(add_series_to_ax, ax=ax, color=color, label=None, linestyle='solid', linewidth = options.linewidth)
    percentiles.apply(fixed, axis=0)
    return ax

def add_series_to_ax(s, ax, color, label, linestyle, linewidth=1.0):
    ax.plot(s.index, s, color=color, ls=linestyle, label=label, alpha=0.8)

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
    x, f, d = scipy.optimize.fmin_l_bfgs_b(obj_f, np.array([0.5, 0.5, 2.0]), approx_grad = True, bounds = [(0,1),[0,1],[0.1,10]])
    return x

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


class logistic_f(keyed_object):

    def __init__(self, c):
        self.c = c

    def get_introspection_key(self):
        return '%s_%.2f' % ('logloss', self.c)

    def __call__(self, x):
        return scaled_logistic_loss(x, self.c)



def train_logistic_model(X, Y, offset=0.0):
    """
    each patient is a column.  would like return object to be series whose index matches feature names
    """
    offset_v = np.ones(X.shape[1]) * offset
    def obj_f(b):
        error_vect = (X.T.dot(b)+offset_v).apply(logistic) - Y
        return error_vect.dot(error_vect)
    import scipy.optimize

    ans, f, d = scipy.optimize.fmin_l_bfgs_b(obj_f, np.zeros(X.shape[0]), approx_grad = True)
    return pandas.Series(ans, index=X.index)

def train_exponential_model(X, Y, offset=0.0):
    """
    each patient is a column.  would like return object to be series whose index matches feature names
    """
    offset_v = np.ones(X.shape[1]) * offset
    def obj_f(b):
        error_vect = (X.T.dot(b)+offset_v).apply(math.exp) - Y
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

class and_bin(object):

    def __init__(self, bin1, bin2):
        self.bin1, self.bin2 = bin1, bin2

    def __contains__(self, obj):
        return obj in self.bin1 and obj in self.bin2

    def __repr__(self):
        return '%s_%s' % (repr(self.bin1), repr(self.bin2))


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

    def __init__(self, num_rows, num_cols, hspace=0.3, wspace=0.3, title=None):
        self.num_rows, self.num_cols, self.hspace, self.wspace = num_rows, num_cols, hspace, wspace
        self.figure_limit = num_rows * num_cols
        self.title=title
        self.cur_fig_num = -1
        self.figs = []
        self.start_new_page()


    def start_new_page(self):
        if self.cur_fig_num != 0:
            self.figs.append(plt.figure())
            self.figs[-1].subplots_adjust(hspace=self.hspace,wspace=self.wspace)
            self.figs[-1].suptitle(self.title)
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

def get_rand_trunc_normal(m, phi):
    import random
    ans = random.gauss(m, phi)
    while ans < 0.0 or ans > 1.0:
        ans = random.gauss(m, phi)
    return ans

def get_rand_beta(m, phi, r=None):
    s = (1.0/phi) - 1
    alpha = 1.0 + s*m
    beta = 1.0 + s*(1-m)
    if r == None:
        return random.betavariate(alpha,beta)
    else:
        return r.betavariate(alpha, beta)

def get_beta_log_p(x, m, phi):
    s = (1.0/phi) - 1
    alpha = 1.0 + s*m
    beta = 1.0 + s*(1-m)
    import math, scipy.special
    #return scipy.stats.beta.pdf(x,alpha,beta)
    c = 1.0 / scipy.special.beta(alpha,beta)
    #c = math.gamma(alpha+beta)/(math.gamma(alpha)*(math.gamma(beta)))
    import math
    return math.log(c) + (alpha-1) * math.log(x) + (beta-1) * math.log(1-x)
    return c * math.pow(x,alpha-1) * math.pow(1-x,beta-1)

def get_beta_log_p_given_mean(x, mean, phi):
    s = (1.0/phi) - 1
    m = (mean*(2+s) - 1) / s
    print mean, m
    alpha = 1.0 + s*m
    beta = 1.0 + s*(1-m)
    import math, scipy.special
    #return scipy.stats.beta.pdf(x,alpha,beta)
    c = 1.0 / scipy.special.beta(alpha,beta)
    #c = math.gamma(alpha+beta)/(math.gamma(alpha)*(math.gamma(beta)))
    import math
    return math.log(c) + (alpha-1) * math.log(x) + (beta-1) * math.log(1-x)

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





def merge_posteriors(p1, p2):
    """
    posteriors are just a dictionary.  merge samples using concat
    """
    p = type(p1)({})
    for param in p1:
        if isinstance(p1[param], pandas.DataFrame) or isinstance(p1[param], pandas.Series):
            p[param] = pandas.concat([p1[param],p2[param]])
        else:
            p[param] = p1[param]
    return p

class multichain_posterior(keyed_dict):
    """
    the object returned by a merged_get_posterior_f
    only this type of object(for now) can have its convergence statistic computed, when the method for getting samples by chain is requested.  later on add the method for original posterior object, which i've just been using a dict for
    assuming that samples from separate chains were just appended to each other
    """
    #def __init__(self, posteriors, num_chains):
    #    self.num_chains = num_chains
    #    pdb.set_trace()
    #    keyed_dict.__init__(posteriors)

    def get_chainwise_posteriors(self):
        pdb.set_trace()
        l = [keyed_dict({}) for i in xrange(self.num_chains)]
        for param, samples in self.iteritems():
            try:
                bin_width = samples.shape[0] / self.num_chains
                for i in xrange(self.num_chains):
                    low, high = i*bin_width, (i+1)*bin_width
                    l[i][param] = samples.iloc[low:high,:]
            except:
                pass
        return l
                

class gelman_statistic_f(possibly_cached):
    """
    returns a dictionary.  calculate the single variable version for each stat
    """

    print_handler_f = staticmethod(write_DataFrame)

    read_f = staticmethod(read_DataFrame)

    def get_introspection_key(self):
        return 'gelmanstat'

    def key_f(self, posteriors):
        return '%s_%s' % (self.get_key(), posteriors.get_key())

    def location_f(self, posteriors):
        return '%s/%s' % (global_stuff.for_dropbox, 'gelmans')

    
    @save_to_file
    def __call__(self, posteriors):
        from rpy import r

        d = pandas.DataFrame(index=['low','hi'])
        chainwise = posteriors.get_chainwise_posteriors()
        r.library('coda')
        # for each variable, run single-variate test
        for param in chainwise.__iter__().next():
            # iterate through corresponding columns of the same multivariate variable
            for all_cols, idx in itertools.izip(itertools.izip(*[itertools.imap(lambda x:x[1],posterior[param].iteritems()) for posterior in chainwise]), itertools.count(0)):
                param_name = '%s_%d' % (param, idx)
                print param_name
                # put into R data structure to feed to R gelman statistic calculator
                # make list of series in python, assign to x in R using assign
                # directly run r code using multiple lines
                r.assign('ls_%s' % param_name, all_cols)
                r('mcmc_l_%s=list()' % param_name)
                r.assign('num_chains_%s' % param_name, posteriors.num_chains)
                r('for(i in 1:num_chains_%s){mcmc_l_%s[[i]]=mcmc(ls_%s[[i]])}' % (param_name, param_name, param_name))
                r('res_%s=gelman.diag(mcmc_l_%s)' % (param_name, param_name))
                res = r['res_%s' % param_name]
                d[param_name] = pandas.Series({'low':res['psrf'][0][0], 'hi':res['psrf'][0][1]})

        def is_important(s):
            ms = ['B_a', 'B_b', 'B_c', 'phi_a', 'phi_b', 'phi_c', 'phi_m']
            return sum([s.__contains__(m) for m in ms]) > 0

        important_columns = [c for c in d.columns if is_important(c)]
        d_new = d.ix[:,important_columns + list(d.columns)]

        return pandas.DataFrame.transpose(d_new)


def filter_by_max_num(df, n):
    N = df.shape[0]
    if n < N:
        to_use = [int(i*float(N/n)) for i in xrange(n)]
    else:
        to_use = range(N)
    if len(df.shape) == 2:
        return df.iloc[to_use,:]
    else:
        return df.iloc[to_use]


def parallel_map(f, iterable, num_processes):
    """
    make a 
    """
    import multiprocessing
    results = multiprocessing.Manager().list()
    iterable_queue = multiprocessing.Queue()

    def worker(_iterable_queue, _f, results_queue):
        for arg in iter(_iterable_queue.get, None):
            results_queue.append(_f(arg))

    for x in iterable:
        iterable_queue.put(x)

    for i in xrange(num_processes):
        iterable_queue.put(None)

    workers = []

    for i in xrange(num_processes):
        p = multiprocessing.Process(target=worker, args=(iterable_queue, f, results))
        p.start()
        workers.append(p)

    for p in workers:
        p.join()

    return [x for x in results]

def truncated_normal_pdf(x, mu, sigma, a, b):
    import scipy.stats
    a = scipy.stats.norm.pdf(x, mu, sigma)
    pct = scipy.stats.norm.cdf(0, mu, sigma) + (1.0 - scipy.stats.norm.cdf(1, mu, sigma))
    return a / pct

def normal_pdf(x, mu, sigma):
    import scipy.stats
    return scipy.stats.norm.pdf(x, mu, sigma)
                                                


class shape_plotter(object):

    def __init__(self, predictor, low, high, num):
        self.predictor = predictor
        self.low, self.high, self.num = low, high, num

    def __call__(self, ax, _datum, color, label, linestyle = '-', linewidth=17):
        ts = np.linspace(self.low, self.high, self.num)
        ys = [self.predictor(_datum, t) / _datum.s for t in ts]
        ax.plot(ts, ys, color=color, label=label, linestyle=linestyle)



class discrete_shape_plotter(object):

    def __init__(self, predictor):
        self.predictor = predictor

    def __call__(self, ax, _datum, color, label, linestyle = ':', marker='.'):
        ts = global_stuff.times
        ys = [self.predictor(_datum, t) / _datum.s for t in ts]
        ax.plot(ts, ys, color=color, label=label, linestyle=linestyle)        

def plot_stuff(plotters, patient_block, color_list, label_list, axis_labels):
    """
    takes in predictor, m x n list of patients, m x n list of colors, m x n list of labels, m list of axis labels
    """
    m = len(axis_labels)

    figs = []

    for i in xrange(m):
        
        fig = plt.figure()
        ax = fig.add_subplot(1,1,1)

        ax.set_title('patients with ' + axis_labels[i])
        ax.set_xlabel('time')
        ax.set_ylabel('scaled fxn value')
        n = len(patient_block[i])

        for j in xrange(n):
            first = True
            for plotter in plotters:
                if first:
                    plotter(ax, patient_block[i][j], color_list[j], label_list[j])
                    first = False
                else:
                    plotter(ax, patient_block[i][j], color_list[j], None)
        ax.legend(prop={'size':5})
        figs.append(fig)

    return figs


def stratify_dataset(_data, bin_f, num_bins):
    strats = [[] for x in xrange(num_bins)]
    for _datum in _data:
        strats[bin_f(_datum)].append(_datum)
    return [data(x) for x in strats]

"""
shitty ass code goes below here
"""

def plot_2_by_2_predictions(data, cv_f, trainer_plotter_cons_tuples, to_plot, plot_truth = False):

    from scripts.for_paper import options as options

    textsize = options.textsize

    fig = plt.figure()
    
    k = 0

    for train_data, test_data in cv_f(data):
        _plotters = []
        for trainer, plotter_cons in trainer_plotter_cons_tuples:
            if trainer.normal():
                predictor = trainer(train_data)
            else:
                predictor = trainer(train_data, test_data)
            _plotters.append(plotter_cons(predictor))
        for datum in test_data:
            if to_plot(datum):
                k = k + 1
                ax = fig.add_subplot(2,2,k)
                #ax.set_title(datum.pid)
                for plotter in _plotters:
                    plotter(ax, datum)
                if plot_truth:
                    ax.plot(datum.ys.index, datum.ys, color='black')
                for tick in ax.yaxis.get_major_ticks():
                    tick.label.set_fontsize(textsize) 

                for tick in ax.xaxis.get_major_ticks():
                    tick.label.set_fontsize(textsize) 
                ax.set_ylim(0,1)
                ax.set_xlim(-1,50)
                if k in [3,4]:
                    ax.set_xlabel('months', fontsize=textsize)
                if k in [1,3]:
                    ax.set_ylabel('scaled fxn value', fontsize=textsize)
                #ax.legend(prop={'size':textsize}, loc=4)
                ax.set_ylim((0,1))

    plt.tight_layout()

    return fig

class nan_median_f(keyed_object):

    def __call__(self, s):
        ok = s.iloc[[not np.isnan(x) for x in s]]
        return np.median(ok)

    def get_introspection_key(self):
        return 'nanmedian'

nan_median = nan_median_f()

def get_num_above(datum, tol):
    count = 0
    return sum([v > datum.s for k,v in datum.ys.iteritems()])

def has_num_straight_zeros(datum, num):
    for i in range(len(datum.ys)-num):
        if sum([datum.ys.iloc[x] < 0.01001 for x in range(i,i+num)]) == num:
            return True
    return False

def asymptote_above_init(datum):
    a,b,c = get_curve_abc(1.0, datum.ys)
    return the_f(48,1,a,b,c) - datum.s
    return (1.0 - a) - datum.s

def draw_curve(ax, s, s_loc, curve, color, label, lw, alpha):
    ax.plot([s_loc,0],[s,s], color=color, linewidth=lw)
    try:
        ax.axvline(0.5, ymin=curve[1], ymax=s, linewidth=2.5, color=color, alpha=0.3)
    except KeyError:
        ax.axvline(0.5, ymin=curve[2], ymax=s, linewidth=2.5, color=color, alpha=0.3)
    ax.plot(curve.index, curve, label = label, color = color, linewidth=lw)
    ax.set_xlim((s_loc,50))
