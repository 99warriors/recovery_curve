/*
this is full model with B_a,B_b,B_c,phi_a,phi_b,phi_c,phi_noise
*/
data{
	int<lower=0> N; # number of patients
	int<lower=0> L; # total number of datapoints

	int<lower=0> ls[N];
	real<lower=0> ts[L];
	real vs[L];

	real<lower=0.0,upper=1.0> ss[N];

	real<lower=0> phi_m;

	real<lower=0,upper=1> as[N];


	real<lower=0> cs_l;


}
parameters{
	real<lower=0,upper=1> bs[N];
	real<lower=0> cs[N];
}
model{

	int c;
	c <- 1;

	for(i in 1:N){
	      for(j in 1:ls[i]){
		    vs[c] ~ normal(ss[i] * (1.0 - as[i] - (bs[i] * (1.0 - as[i]) * exp(-1.0 * ts[c] / cs[i]))), phi_m);
		    c <- c + 1;
	      }
	}
}