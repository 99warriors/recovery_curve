import recovery_curve.global_stuff
import matplotlib.pyplot as plt
import recovery_curve.prostate_specifics as ps
import numpy as np
"""
for each phi, plot beta pdf with several modes
"""

out_file = '/Users/glareprotector/Documents/lab/glare_remix2/bin/figs/betas'

phis = [0.001, 0.005, 0.05, 0.1, 0.2, 0.6]
ms = [0.01, 0.05, 0.1, 0.2, 0.3, 0.4, 0.5]
#ms = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]



roll = ps.pp_roll(3,2, hspace=0.5,wspace=0.3)

xs = np.linspace(0,1,100)

for phi in phis:
    ax = roll.get_axes()
    ax.set_title('phi: %.4f' % phi)
    for m in ms:
        ax.plot(xs, [ps.get_beta_p(x,m,phi) for x in xs])

ps.multiple_figures_to_pdf(roll.figs, out_file)
