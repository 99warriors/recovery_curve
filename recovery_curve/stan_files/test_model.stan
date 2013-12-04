data{
	real x;
}
parameters{
	real mu;
}
model{
	x ~ normal(mu,1);
}
