 /*
this is full model with B_a,B_b,B_c,phi_a,phi_b,phi_c,phi_noise
*/
data{
	int<lower=0> N; # number of patients
#	int<lower=0> N_test; # number of test patients
	int<lower=0> K_A; # number of covariates for A
	int<lower=0> K_B; # number of covariates for B
	int<lower=0> K_C; # number of covariates for C
	int<lower=0> L; # total number of datapoints
#	int<lower=0> L_test; # total number of datapoints

	# store all function values in 1 long vector
	int<lower=0> ls[N]; # number of function values for each patient
	real<lower=0> ts[L]; # the times at which the function values occur
	real vs[L]; # the function values themselve

	# covariates - covariates for regression models for the 3 parameters (a,b,c) can be different
	vector[K_A] xas[N];
	vector[K_B] xbs[N];
	vector[K_C] xcs[N];
	real<lower=0.0,upper=1.0> ss[N];

	# covariates for test patient
#	vector[K_A] xas_test[N_test];	
#	vector[K_B] xbs_test[N_test];
#	vector[K_C] xcs_test[N_test];
#	real<lower=0.0,upper=1.0> ss_test[N_test];

	# population average parameters
	real<lower=0,upper=1> pop_a;	
	real<lower=0,upper=1> pop_b;	
	real<lower=0> pop_c;	

	# parameters specifying prior on B_a, B_b, B_c
	real<lower=0> c_a;
	real<lower=0> c_b;
	real<lower=0> c_c;

	# parameters specifying prior on phi_a, phi_b, phi_c
	real<lower=0> l_a;
	real<lower=0> l_b;
	real<lower=0> l_c;

	real<lower=0> l_m;

	#real<lower=0,upper=1> phi_a;
	#real<lower=0,upper=1> phi_b;
	#real<lower=0,upper=1> phi_c;

	# we fix the noise variance
	#real<lower=0, upper=1> phi_m;

	# values less than this are interpreted to be a 0
	#real<lower=0> eps;
	#eps <- 0.01


}
parameters{
	# regression coefficients
	vector[K_A] B_a;
	vector[K_B] B_b;
	vector[K_C] B_c;
	
	# specify variance of a,b,c conditional on covariates
	real<lower=0,upper=1> phi_a;
	real<lower=0,upper=1> phi_b;
	real<lower=0,upper=1> phi_c;

	real<lower=0,upper=1> phi_m;  

	# patients a,b,c parameters
	real<lower=0,upper=1> as[N];	
	real<lower=0,upper=1> bs[N];
	real<lower=0> cs[N];

	# a,b,c parameters for test patients - this is so i don't have to do any sampling on my own during prediction
#	real<lower=0,upper=1> as_test[N_test];	
#	real<lower=0,upper=1> bs_test[N_test];
#	real<lower=0> cs_test[N_test];

	#real<lower=0,upper=1> pis[N]; # patient-specific probability of going into bernoulli mode MIX

	real<lower=0,upper=1> pi;
	real<lower=0,upper=1> R; # given that they are in 'bernoulli mode', the probability of writing down a 1 MIX

}
transformed parameters{

	# these are transformation of phi parameters
	real<lower=0> s_a;
	real<lower=0> s_b;		
	real<lower=1> s_c;
	real<lower=0> s_m;

	# these are the modes of the a,b,c parameters
	real<lower=0,upper=1> m_as[N];
	real<lower=0,upper=1> m_bs[N];
	real<lower=0> m_cs[N];

	# likewise for test patients
#	real<lower=0,upper=1> m_as_test[N_test];
#	real<lower=0,upper=1> m_bs_test[N_test];
#	real<lower=0> m_cs_test[N_test];

	# link function relates covariates to the mode of a,b,c distributions
	for(i in 1:N){
	      m_as[i] <- inv_logit(logit(pop_a) + dot_product(xas[i], B_a));		      
	      m_bs[i] <- inv_logit(logit(pop_b) + dot_product(xbs[i], B_b));
	      m_cs[i] <- exp(log(pop_c) + dot_product(xcs[i], B_c));
	}

	# likewise for test patients
#	for(i in 1:N_test){
#	      m_as_test[i] <- inv_logit(logit(pop_a) + dot_product(xas_test[i], B_a)); 
#	      m_bs_test[i] <- inv_logit(logit(pop_b) + dot_product(xbs_test[i], B_b));
#	      m_cs_test[i] <- exp(log(pop_c) + dot_product(xcs_test[i], B_c));
#	}

	# the transformation of phi variables
	s_a <- 1.0 / phi_a - 1;
	s_b <- 1.0 / phi_b - 1;
	s_c <- 1.0 / phi_c;
	s_m <- 1.0 / phi_m - 1;
	
	


}
model{
	# 2 temp variables
	int c;
	real m;
	#real eps;
	#eps <- 0.01
	c <- 1;

	B_a ~ normal(0, c_a);
	B_b ~ normal(0, c_b);
	B_c ~ normal(0, c_c);

	# encourage variances of a,b,c to be small
	phi_a ~ exponential(l_a);
	phi_b ~ exponential(l_b);
	phi_c ~ exponential(l_c);
	phi_m ~ exponential(l_m);

	pi ~ beta(1,1); # diffuse beta prior on mixture probability
	R ~ beta(1,1);

	# conditional on covariates, a,b,c follow beta, beta, gamma distributions, respectively
	for(i in 1:N){
	      as[i] ~ beta(1.0 + (s_a * m_as[i]), 1.0 + (s_a * (1.0 - m_as[i])));
	      bs[i] ~ beta(1.0 + (s_b * m_bs[i]), 1.0 + (s_b * (1.0 - m_bs[i])));
	      cs[i] ~ gamma(s_c, (s_c - 1.0) / m_cs[i]);
	}

	# likewise for test patients
#	for(i in 1:N_test){
#	      as_test[i] ~ beta(1.0 + (s_a * m_as_test[i]), 1.0 + (s_a * (1.0 - m_as_test[i])));
#	      bs_test[i] ~ beta(1.0 + (s_b * m_bs_test[i]), 1.0 + (s_b * (1.0 - m_bs_test[i])));
#	      cs_test[i] ~ gamma(s_c, (s_c - 1.0) / m_cs_test[i]);
#	}


	# likelihood function - mixture of bernoulli distribution and that predicted by the curve model
	for(i in 1:N){
	      for(j in 1:ls[i]){
		    m <- ss[i] * (1.0 - as[i] - (bs[i] * (1.0 - as[i]) * exp(-1.0 * ts[c] / cs[i]))); # m is temp variable representing the mode of the distribution of current function value
		    #increment_log_prob(log_sum_exp(log(pis[i]) + bernoulli_log(vs[c], R), log(1.0-pis[i]) + beta_log(vs[c],1.0 + (s_m * m), 1.0 + (s_m * (1.0 - m))))); # likelihood function - doesn't compile because vs[c], the function value, is a float, and needs to be int in order to be input to bernoulli_log MIX
		    if (vs[c] < 0.0001){
		       	 increment_log_prob(log(pi * (1-R)));		    
		    }
		    else if (vs[c] > 1.0 - 0.0001){
		    	 increment_log_prob(log(pi * R));
		    }
		    else{ 
		    	 increment_log_prob(log(1.0-pi) + beta_log(vs[c],1.0 + (s_m * m), 1.0 + (s_m * (1.0 - m))));
		    }
		    c <- c + 1;
	      }
	}

}