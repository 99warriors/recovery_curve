import matplotlib
matplotlib.use('Agg')
import random
random.seed(0)

home = '/home/ubuntu/recovery_curve'
data_home = '%s/%s/%s' % (home, 'bin', 'cache_pystan')
train_diffcovs_r_script = '%s/%s/%s' % (home, 'recovery_curve', 'train_diffcovs_model.r')
raw_data_home = '%s/%s' % (home, 'raw_data')
xs_file = '%s/%s' % (raw_data_home, 'xs.csv')
ys_folder = '%s/%s' % (raw_data_home, 'series')
all_pid_file = '%s/%s' % (raw_data_home, 'pids.csv')
