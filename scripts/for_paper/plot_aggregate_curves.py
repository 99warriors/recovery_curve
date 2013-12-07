import options
import recovery_curve.prostate_specifics as ps
import matplotlib.pyplot as plt

"""
for filtered data, plots aggregate curve, aggregate shape
"""


"""
options
"""
outfile = options.get_outfile_path('aggregate_curves')
data = options.filtered_data


"""
meat
"""

def do_stuff(data, outfile):
    fig = plt.figure()
    ax = fig.add_subplot(1,1,1)
    agg_shape = ps.aggregate_shape_f()(data)
    agg_curve = ps.aggregate_curve_f()(data)
    ax.plot(agg_shape.index, agg_shape, label = 'aggregate shape')
    ax.plot(agg_curve.index, agg_curve, label = 'aggregate curve')
    ax.legend(loc=4)
    ax.set_xlabel('time')
    fig.suptitle('aggregates of filtered data')
    ps.figure_to_pdf(fig, outfile)

do_stuff(data, outfile)
