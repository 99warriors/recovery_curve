data{
	int<lower=0> N;
	real<lower=0, upper=1> vs[N];
	real phi;
}
parameters{
	real<lower=0,upper=1> m;
	real<lower=0,upper=1> test_m;
}
transformed parameters{
	real<lower=0> s;
	s <- 1.0 / phi - 1;
}
model{
	for(i in 1:N){
	      vs[i] ~ beta(1.0 + (s * m), 1.0 + (s * (1.0 - m)));
	}
	
	test_m ~ beta(1.0 + (s * m), 1.0 + (s * (1.0 - m)));

}