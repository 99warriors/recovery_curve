# trains diffcovs model, reads in data, pops, hyperparameters from text 

args <- commandArgs(trailing=TRUE)
pops_file <- args[1]
data_path <- args[2]
hypers_path <- args[3]
iters <- as.numeric(args[4])
chains <- as.numeric(args[5])
seed <- as.numeric(args[6])
save_path <- args[7]
train_helper_file <- args[8]
model_file <- args[9]

print('iters')
print(iters)
print('chains')
print(chains)

#source('train_helper.r')

source(train_helper_file)

library(rstan)

#model_file <- './full_model_diffcovs.stan'

pops <- read_in_pops(pops_file)

print(pops)

data <- get_real_full_data_diffcovs(data_path)



hypers <- read_in_hypers(hypers_path)

print(hypers)

all_data <- c(pops, data, hypers)

fit <- stan(file=model_file, data=all_data, iter=iters, chains=chains, verbose=F, seed=seed)

write_full_posterior_parameters(fit, save_path)