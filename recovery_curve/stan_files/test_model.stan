data{
	int n;
	real<lower=0.0> ts[n];
	real<lower=0.0,upper=1.0> vs[n];
	real s;
	real<lower=0.0> phi_m;
}
parameters{
	real<lower=0.0,upper=1.0> a;
	real<lower=0.0,upper=1.0> b;
	real<lower=0.0> c;
}
model{
	c ~ exponential(.01);
	for(i in 1:n){
	      vs[i] ~ normal(s * ((1.0 - a) - (1.0 - a) * (b) * exp(-1.0 * ts[i] / c)), phi_m);
	}
}
