import prostate_specifics as ps
import getting_data as gd
import recovery_curve.hard_coded_objects.feature_sets as hard_coded_feature_sets

upscale_delta = 0.01
post_process_f = ps.normalized_data_f()


pid_iterator = gd.filtered_pid_iterator(set_hard_coded_key_dec(gd.filtered_pid_iterator,'surgpids')(gd.all_ucla_pid_iterator(), gd.bin_f(ps.ucla_treatment_f(),ps.equals_bin([gd.ucla_treatment_f.surgery]))), gd.is_good_pid())

ys_f = gd.modified_ys_f(gd.ys_f(gd.ys_f.sexual_function), gd.push_0_up_ys_modifier(upscale_delta))
init_f = gd.set_hard_coded_key_dec(gd.s_f, 'init')(ys_f)
actual_ys_f = gd.actual_ys_f(ys_f, actual_ys_f_shift)

feature_set = hard_coded_feature_sets.default_simple_indicators
x_abc_fs = set_hard_coded_key_dec(ps.x_abc_fs, feature_set.get_key())(feature_set, feature_set, feature_set)


data = ps.get_data_f(x_abc_f, init_f, actual_ys_f)(pid_iterator)
processed_data = post_process_f(data)
