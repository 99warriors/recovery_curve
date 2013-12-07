import options
import recovery_curve.prostate_specifics as ps
import matplotlib.pyplot as plt
import recovery_curve.global_stuff as global_stuff
import random
import recovery_curve.get_posterior_fs as gp
import recovery_curve.getting_data as gd

"""
for 2 parameter settings, plots box plots of posterior distributions
"""

"""
options
"""
outfile_1 = options.get_outfile_path('simulation_posteriors_1')
outfile_2 = options.get_outfile_path('simulation_posteriors_2')
seed = 0
inference_num_chains = 4
inference_iter = 5000


"""
meat
"""

def do_stuff(outfiles, sim_times, N1, N2, params, pops, noise_f, seed, gp_partial, num_chains, num_iters, num_processes):
    #get_posterior_f = gp.parallel_merged_get_posterior_f(gp_partial, num_iters, num_chains, num_processes)
    get_posterior_f = gp.merged_get_posterior_f(gp_partial, num_iters, num_chains)
    id_to_x_s_f = gd.an_id_to_x_s_f(1)
    for outfile, param in zip(outfiles, params):
        
        fig = plt.figure()
        
        ax = fig.add_subplot(2,1,1)
        N = N1
        pid_iterator = gd.fake_pid_iterator(N)
        sim_data = gd.simulated_get_data_f(param, pops, id_to_x_s_f, sim_times, noise_f, seed)(pid_iterator)
        posteriors = get_posterior_f(sim_data)
        ps.plot_posterior_boxplots(ax, posteriors, param)

        ax = fig.add_subplot(2,1,2)
        N = N2
        pid_iterator = gd.fake_pid_iterator(N)
        sim_data = gd.simulated_get_data_f(param, pops, id_to_x_s_f, sim_times, noise_f, seed)(pid_iterator)
        posteriors = get_posterior_f(sim_data)
        ps.plot_posterior_boxplots(ax, posteriors, param)
        
        ps.figure_to_pdf(fig, outfile)


do_stuff([outfile_1, outfile_2], options.sim_times, options.sim_N_1, options.sim_N_2, [options.sim_param_1, options.sim_param_2], options.sim_pops, options.sim_noise_f, seed, options.sim_get_posterior_f_partial, inference_num_chains, inference_iter, global_stuff.num_processors)
