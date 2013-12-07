import options
import recovery_curve.prostate_specifics as ps
import matplotlib.pyplot as plt

"""
plots aggregate curve of raw data, stratified by age, init.  puts on same plot
"""


"""
options
"""
data = options.filtered_data
outfile = options.get_outfile_path('stratified_real_data')
hspace=0.3
wspace=0.3

"""
meat
"""


strat_by_age = ps.stratify_dataset(data, options.get_age_bin, options.num_age_bins)
strat_by_init = ps.stratify_dataset(data, options.get_init_bin, options.num_init_bins)

agg_by_age = [ps.aggregate_shape_f()(_data) for _data in strat_by_age]
agg_by_init = [ps.aggregate_shape_f()(_data) for _data in strat_by_init]

fig = plt.figure()
fig.subplots_adjust(hspace=hspace,wspace=wspace)
fig.suptitle('filtered data stratified by age, init')

ax = fig.add_subplot(2,1,1)
for curve, title in zip(agg_by_age, options.age_titles):
    ax.plot(curve.index, curve, label = title)
ax.set_xlabel('months')
ax.set_ylabel('scaled f')
ax.legend(prop={'size':5}, loc=4)

ax = fig.add_subplot(2,1,2)
for curve, title in zip(agg_by_init, options.init_titles):
    ax.plot(curve.index, curve, label = title)
ax.set_xlabel('months')
ax.set_ylabel('scaled f')
ax.legend(prop={'size':5}, loc=4)

ps.figure_to_pdf(fig, outfile)
