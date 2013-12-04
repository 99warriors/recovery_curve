 /*
beta regression.  one datapoint per patient
*/
data{
	int<lower=0> N; # number of patients
	int<lower=0> K; # number of covariates
	int<lower=0> L; # total number of datapoints

	int<lower=0> ls[N];
	real vs[L];

	vector[K] xs[N];

	real<lower=0,upper=1> pop_val;	

	real<lower=0> c;

	real<lower=0> l;

	real<lower=0> l_m;

}
parameters{
	vector[K] B;
	real<lower=0,upper=1> phi;
	real<lower=0,upper=1> phi_m;
	real<lower=0,upper=1> zs[N]; # this is the stage 1 value

}
transformed parameters{

	# these are actual parameters for input into distributions
	real<lower=0> s;
	real<lower=0> s_m;
	real<lower=0,upper=1> m_zs[N];

	for(i in 1:N){
	      m_zs[i] <- inv_logit(logit(pop_val) + dot_product(xs[i], B));		      
	}
	
	s <- 1.0 / phi - 1;
	s_m <- 1.0 / phi_m - 1;
}
model{
	
	int e;
	e <- 1;
	
	
	B ~ normal(0, c);
	
	phi ~ exponential(l);
	phi_m ~ exponential(l_m);

	
	for(i in 1:N){
	      zs[i] ~ beta(1.0 + (s * m_zs[i]), 1.0 + (s * (1.0 - m_zs[i])));
	}
	

	for(i in 1:N){
	      for(j in 1:ls[i]){
	      	    vs[e] ~ beta(1.0 + (s_m * zs[i]), 1.0 + (s_m * (1 - zs[i])));
		    e <- e + 1;
	      }
	}
	
}