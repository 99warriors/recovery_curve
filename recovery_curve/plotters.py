from management_stuff import *
from prostate_specifics import *
import pdb

class plotter(keyed_object):
    """
    this function plots the prediction of the predictor it is initialized with.
    init in subclasses also takes other arguments about how the plotting should be done
    """
    def __init__(self, color, label, predictor):
        self.color, self.label, self.predictor = color, label, predictor

    def __call__(self, ax, datum):
        pass

class color_map_plotter(keyed_object):

    def __init__(self, color_map, label, predictor):
        self.color_map, self.label, self.predictor = color_map, label, predictor

    def __call__(self, ax, datum):
        pass



class point_t_discrete_plotter(plotter):
    """
    for predictors that for a given t, return a single point
    does plotting at discrete points - times at which there is data
    """
    def __call__(self, ax, datum):
        s = pandas.Series({t:self.predictor(datum, t) for t in datum.ys.index})
        prob = 0.0
        sigma = 0.01
        import math
        from scipy.stats import norm
        try:
            for t,v in s.iteritems():
                prob = prob + math.log(norm(0,0.01).pdf(s[t] - datum.ys[t]))
        except ValueError:
            pass
        ax.plot(s.index, s, color=self.color, label='%s, P(D,theta): %.2f' % (self.label, prob), linestyle='None', marker='D')

class point_t_discrete_print_prob_plotter(plotter):
    """
    for predictors that for a given t, return a single point
    does plotting at discrete points - times at which there is data
    """
    def __init__(self, error_pdf, color, label, predictor):
        self.error_pdf = error_pdf
        plotter.__init__(self, color, label, predictor)

    def __call__(self, ax, datum):
        s = pandas.Series({t:self.predictor(datum, t) for t in datum.ys.index})
        prob = 0.0
        sigma = 0.01
        import math
        from scipy.stats import norm
        try:
            for t,v in s.iteritems():
                this_prob = self.error_pdf(s[t], datum.ys[t])
                prob = prob + math.log(this_prob)
                #prob = prob + math.log(norm(0,0.01).pdf(s[t] - datum.ys[t]))
        except ValueError:
            pass
        ax.plot(s.index, s, color=self.color, label='%s, P(D,theta): %.2f' % (self.label, prob), linestyle='None', marker='D')
            

class distribution_t_discrete_plotter(plotter):
    """ 
    for predictors that for a given t, returns a distribution (as a series)
    does plotting at discrete points - times at which there is data
    """

    def __init__(self, width, num_points, color, label, predictor):
        self.width, self.num_points = width, num_points
        plotter.__init__(self, color, label, predictor)
        
    def __call__(self, ax, datum):
        for t in datum.ys.index:
            dist = self.predictor(datum, t)
            N = len(dist)
            point_ys = [p for p in dist if random.random() < float(self.num_points)/N]
            point_xs = [random.uniform(t-self.width,t+self.width) for x in xrange(self.num_points)]
            ax.plot(point_xs, point_ys, linestyle='None', marker=',')

class single_curve_plotter(plotter):

    def __init__(self, low_t, high_t, num_t, color, label, predictor):
        self.low_t, self.high_t, self.num_t = low_t, high_t, num_t
        plotter.__init__(self, color, label, predictor)

class t_point_predictor_curve_plotter(single_curve_plotter):
    """
    for predictors that take in any t, and returns a single point
    """
    def __call__(self, ax, datum):
        s = pandas.Series({t:self.predictor(datum, t) for t in np.linspace(self.low_t, self.high_t, self.num_t)})
        ax.plot(s.index, s, color=self.color, label=self.label, linestyle='--')

class abc_point_predictor_curve_plotter(single_curve_plotter):
    """
    for predictors that return a single abc
    """
    def __call__(self, ax, datum):
        a,b,c = self.predictor(datum)
        ts = np.linspace(self.low_t, self.high_t, self.num_t)
        ys = [the_f(t,datum.s,a,b,c) for t in ts]
        ax.plot(ts, ys, color=self.color, label=self.label, linestyle='--')



class abc_distribution_predictor_curve_plotter(plotter):
    """
    for predictors that return a distribution of abc's, which come as 3 tall thin dataframes of the same height
    """

    def __init__(self, alpha, num_curves, low_t, high_t, num_t, color, label, predictor):
        self.alpha, self.num_curves, self.low_t, self.high_t, self.num_t, = alpha, num_curves, low_t, high_t, num_t
        plotter.__init__(self, color, label, predictor)

    def __call__(self, ax, datum):
        print datum.pid
        abcs = self.predictor(datum)
        ts = np.linspace(self.low_t, self.high_t, self.num_t)
        a_label = self.label
        for row_num, (a,b,c) in abcs.iterrows():
            ys = [the_f(t,datum.s,a,b,c) for t in ts]
            ax.plot(ts, ys, color=self.color, label=a_label, linestyle='-', alpha=self.alpha)
            a_label = None



class abc_distribution_predictor_chainwise_curve_plotter(plotter):
    """
    color based on chain
    """

    def __init__(self, alpha, num_curves, low_t, high_t, num_t, color, label, predictor):
        self.alpha, self.num_curves, self.low_t, self.high_t, self.num_t, = alpha, num_curves, low_t, high_t, num_t
        plotter.__init__(self, color, label, predictor)

    def __call__(self, ax, datum):
        print datum.pid
        abcs = self.predictor(datum)
        ts = np.linspace(self.low_t, self.high_t, self.num_t)
        a_label = self.label
        for row_num, (a,b,c) in abcs.iterrows():
            ys = [the_f(t,datum.s,a,b,c) for t in ts]
            color = self.color[row_num[1]]
            ax.plot(ts, ys, color=color, label=a_label, linestyle='-', alpha=self.alpha)
            a_label = None


class abc_phi_m_distribution_predictor_curve_plotter(color_map_plotter):
    """
    for predictors that return a distribution of abc's, which come as 3 tall thin dataframes of the same height
    """

    def __init__(self, alpha, num_curves, low_t, high_t, num_t, color_map, label, predictor):
        self.alpha, self.num_curves, self.low_t, self.high_t, self.num_t, = alpha, num_curves, low_t, high_t, num_t
        color_map_plotter.__init__(self, color_map, label, predictor)

    def __call__(self, ax, datum):
        print datum.pid
        abc_phi_ms = self.predictor(datum)
        ts = np.linspace(self.low_t, self.high_t, self.num_t)
        a_label = self.label
        max_phi_m = max(abc_phi_ms['phi_m'])
        max_phi_m = np.percentile(abc_phi_ms['phi_m'],90)
        min_phi_m = np.percentile(abc_phi_ms['phi_m'],10)
        med_phi_m = (max_phi_m + min_phi_m) / 2.0
        width = max_phi_m - min_phi_m
        for row_num, (a,b,c, phi_m) in abc_phi_ms.iterrows():
            ys = [the_f(t,datum.s,a,b,c) for t in ts]
            scaled_phi_m = (phi_m - min_phi_m) / width
            color_num = max(min(scaled_phi_m, 0.99), 0.01)
            ax.plot(ts, ys, color=self.color_map(1.0*color_num), label=a_label, linestyle='-', alpha=self.alpha)
            a_label = None
            

class t_distribution_predictor_curve_plotter(plotter):
    """
    for predictors that for a given t, return a distribution, plot distribution of points
    """
    def __init__(self, alpha, num_curves, low_t, high_t, num_t, color, label, predictor):
        self.alpha, self.num_curves, self.low_t, self.high_t, self.num_t, = alpha, num_curves, low_t, high_t, num_t
        plotter.__init__(self, color, label, predictor)

    def __call__(self, ax, datum):
        ts = np.linspace(self.low_t, self.high_t, self.num_t)
        d = {}
        for t in ts:
            predictions = self.predictor(datum, t)
            N = len(predictions)
            d[t] = [y for y in predictions if random.random() < float(self.num_curves)/N]
        d = pandas.DataFrame(d)
        a_label = self.label
        for i, ys in d.iteritems():
            ax.plot(ts, ys, color=self.color, label=a_label, linestyle='None', alpha=self.alpha, marker=',')
            a_label = None


class t_distribution_predictor_curve_plotter_perturbed_ts(plotter):
    """
    
    """
    def __init__(self, alpha, num_curves, low_t, high_t, num_t, color, label, predictor):
        self.alpha, self.num_curves, self.low_t, self.high_t, self.num_t, = alpha, num_curves, low_t, high_t, num_t
        plotter.__init__(self, color, label, predictor)

    def __call__(self, ax, datum):
        import random
        ts = np.linspace(self.low_t, self.high_t, self.num_t)
        gap = float(self.high_t - self.low_t) / self.num_t
        a_label = self.label
        for t in ts:
            predictions = self.predictor(datum, t)
            N = len(predictions)
            plot_ys = [y for y in predictions if random.random() < float(self.num_curves)/N]
            plot_ts = [random.uniform(t-gap/2, t+gap/2) for i in xrange(len(plot_ys))]
            ax.plot(plot_ts, plot_ys, color=self.color, label=a_label, linestyle='None', alpha=self.alpha, marker=',')
            a_label = None

class scalar_distribution_predictor_plotter(plotter):
    """
    color, label, predictor
    """
    def __init__(self, alpha, num_breaks, x_min, x_max, color, label, predictor):
        self.alpha, self.num_breaks, self.x_min, self.x_max = alpha, num_breaks, x_min, x_max
        plotter.__init__(self, color, label, predictor)

    def __call__(self, ax, datum):
        vals = self.predictor(datum)
        ax.hist(vals, bins = self.num_breaks, alpha = self.alpha)
        ax.set_xlim([self.x_min, self.x_max])

