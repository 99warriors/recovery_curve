import options
import recovery_curve.prostate_specifics as ps
import matplotlib.pyplot as plt
import random

"""
plots a few example curves by randomly choosing patients from filtered dataset that have enough datapoints
"""

"""
options
"""
outfile = options.get_outfile_path('example_curves')
data = options.filtered_data
num_rows = 3
num_cols = 4
min_datapoints_needed = 9
hspace=0.5
wspace=0.5

"""
meat
"""

def do_stuff(outfile, data, num_rows, num_cols, min_datapoints_needed, hspace, wspace):
    fig = plt.figure()
    fig.subplots_adjust(hspace=hspace,wspace=wspace)
    shuffled_data = ps.data(random.sample(data,len(data)))
    shuffled_data_iter = iter(shuffled_data)
    k = 0
    for i in xrange(num_rows):
        for j in xrange(num_cols):
            ax = fig.add_subplot(num_rows,num_cols,k+1)
            k = k + 1
            datum = shuffled_data_iter.next()
            while len(datum.ys) <= min_datapoints_needed:
                datum = shuffled_data_iter.next()
            ax.plot(datum.ys.index, datum.ys, color='black', linewidth=1.5)
            ax.set_xlim((-1,50))
            ax.set_ylim((0,1))
            #ax.set_title(datum.pid)
            ax.set_xlabel('months')
            ax.set_ylabel('f', rotation='horizontal')
            ax.plot(-1, datum.s, 'bo')
    ps.figure_to_pdf(fig, outfile)

do_stuff(outfile, data, num_rows, num_cols, min_datapoints_needed, hspace, wspace)
            
