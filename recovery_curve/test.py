from prostate_specifics import *
from management_stuff import *
import matplotlib.pyplot as plt

import pdb

# getting data

pids = all_ucla_pid_iterator()
surgery_pids = set_hard_coded_key_dec(filtered_pid_iterator,'surgpids')(pids, bin_f(ucla_treatment_f(),equals_bin([ucla_treatment_f.surgery])))
xa_fs = keyed_list([ucla_cov_f(ucla_cov_f.age), bin_f(ucla_cov_f(ucla_cov_f.psa), bin(0,20)), s_f(ys_f(ys_f.sexual_function)), ones_f()])
xb_fs = xa_fs
xc_fs = xa_fs
#df = get_dataframe_f(xa_fs).call_and_save(surgery_pids)
init = set_hard_coded_key_dec(s_f, 'init')(ys_f(ys_f.sexual_function))
a_ys = modified_ys_f(ys_f(ys_f.sexual_function), score_modifier_f(0))
gg=set_hard_coded_key_dec(x_abc_fs, 'feat')(xa_fs, xb_fs, xc_fs)
data = get_data_f(gg, init, a_ys)(surgery_pids)
filtered_data = filtered_get_data_f()(data)




# defining trainers
get_pops_f = train_better_pops_f()
h = hypers(1,1,1,15,15,15,10)
get_posterior_f = get_diffcovs_posterior_f(get_pops_f, h, 1000, 1, 1)
diffcovs_trainer = get_diffcovs_point_predictor_f(get_posterior_f, get_param_mean_f())

prior_trainer = get_prior_predictor_f(get_pops_f)

logreg_trainer = get_logreg_predictor_f(times)

trainers = keyed_list([prior_trainer, logreg_trainer, diffcovs_trainer])

trainers = keyed_list([diffcovs_trainer])

# defining other stuff needed for plotting performance

loss_f = scaled_logistic_loss_f(10)

_cv_f = cv_fold_f(3)


# now run it

between_model_performance_fig_f(trainers, _cv_f, loss_f, percentiles, times)(filtered_data)



pdb.set_trace()

_performance_series_f = performance_series_f(crosser, scaled_logistic_loss_f(10), percentiles)

_between_perf_f = between_model_performance_fig_f(_performance_series_f, trainers)

_between_perf_f(data)


pdb.set_trace()

for trainer in trainers:
    crosser = cross_validated_scores_f(trainer, 3, times)
    perfs = performance_series_f(crosser, scaled_logistic_loss_f(10), percentiles)(filtered_data)
    add_performances_to_ax(ax, perfs, trainer.display_color, trainer.display_name)


#trainers = [logreg_trainer]
performances = {}

fig = plt.figure()
ax = fig.add_subplot(1,1,1)

ax.legend()
fig.show()
pdb.set_trace()



agg = aggregate_curve_f()(training_data)

agg_shape = aggregate_shape_f()(filtered_data)

print agg
print agg_shape

get_pops_f = train_better_pops_f()

a_pops = get_pops_f(training_data)

b_pops = train_shape_pops_f()(training_data)

print a_pops
print b_pops


scores = crosser(filtered_data)

pdb.set_trace()

fold = get_data_fold_training(0, 3)(filtered_data)

agg = aggregate_curve_f()(filtered_data)

pdb.set_trace()

pops = train_better_pops_f()(filtered_data)

pdb.set_trace()



pdb.set_trace()

pids = [x for x in all_ucla_pid_iterator()]

print ys_f(ys_f.sexual_function)(pids[0])

class f(object):

    def __repr__(self):
        return 'z'

    def __hash__(self):
        import pdb
        #print 'z'
        #pdb.set_trace()
        return 1

    def __cmp__(self, other):
        print 'ggg'
        return self.__hash__() == other.__hash__()

class g(object):

    def __repr__(self):
        return 'e'

    def __hash__(self):
        import pdb
        #print 'e'
        #pdb.set_trace()
        return 2

a = f()
aa = f()
b = g()
import pandas
d=pandas.DataFrame({a:[3,4,5], b:[7,8,9]})
import pdb
print d[aa]
print d[b]
pdb.set_trace()
