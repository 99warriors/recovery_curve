import options
import recovery_curve.prostate_specifics as ps
import matplotlib.pyplot as plt
import recovery_curve.global_stuff as global_stuff
import random
import recovery_curve.get_posterior_fs as gp
import recovery_curve.getting_data as gd

"""
gets posterior of parameters given real data, plots on single plot, posterior of phis as box plot
"""

outfile = options.get_outfile_path('real_posterior_phis')
data = options.filtered_data

"""
meat
"""
posteriors = options.real_get_posterior_f(data)
fig = plt.figure()

param_names = ['phi_a', 'phi_b', 'phi_c']
box_input = [posteriors[param_name] for param_name in param_names]

ax = fig.add_subplot(1,1,1)
ax.boxplot(box_input)
ax.xticks(range(3), param_names)

ps.figure_to_pdf(fig, outfile)
