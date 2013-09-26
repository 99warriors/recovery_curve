# trains diffcovs model, reads in data, pops, hyperparameters from text 

args <- commandArgs(trailing=TRUE)
pops_file <- args[1]
data_path <- args[2]
hypers_path <- args[3]
iters <- args[4]
chains <- args[5]
save_path <- args[6]

source('train_helper.r')
library(rstan)

model_file <- '~/Documents/lab/glare/prostate_code/rstan_stuff/full_model_diffcovs.stan'

pops <- read_in_pops(pops_file)

data <- get_real_full_data_diffcovs(data_path)

hypers <- read_in_hypers(hypers_path)

all_data <- c(pops, data, hypers)

fit <- stan(file=model_file, data=all_data, iter=iters, chains=chains, verbose=F)

write_full_posterior_parameters(fit, save_path)