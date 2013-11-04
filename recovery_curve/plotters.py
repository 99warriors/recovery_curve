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




class point_t_discrete_plotter(plotter):
    """
    for predictors that for a given t, return a single point
    does plotting at discrete points - times at which there is data
    """
    def __call__(self, ax, datum):
        s = pandas.Series({t:self.predictor(datum, t) for t in datum.ys.index})
        ax.plot(s.index, s, color=self.color, label=self.label, linestyle='None', marker='.')
            

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
        s = pandas.Series({t:self.predictor(datum, t) for t in np.linspace(self.low_t, self.high_t, num_t)})
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
            ax.plot(ts, ys, color=self.color, label=a_label, linestyle='--', alpha=self.alpha)
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
