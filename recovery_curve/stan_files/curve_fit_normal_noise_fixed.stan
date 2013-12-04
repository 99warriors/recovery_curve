/*
this is full model with B_a,B_b,B_c,phi_a,phi_b,phi_c,phi_noise
*/
data{
	int<lower=0> N; # number of patients
	#int<lower=0> K_A; # number of covariates for A
	#int<lower=0> K_B; # number of covariates for B
	#int<lower=0> K_C; # number of covariates for C
	int<lower=0> L; # total number of datapoints

	int<lower=0> ls[N];
	real<lower=0> ts[L];
	real vs[L];

	#vector[K_A] xas[N];
	#vector[K_B] xbs[N];
	#vector[K_C] xcs[N];
	real<lower=0.0,upper=1.0> ss[N];

	#real<lower=0,upper=1> pop_a;	
	#real<lower=0,upper=1> pop_b;	
	#real<lower=0> pop_c;	

	#real<lower=0> c_a;
	#real<lower=0> c_b;
	#real<lower=0> c_c;

	#real<lower=0> l_a;
	#real<lower=0> l_b;
	#real<lower=0> l_c;
	#real<lower=0> l_m;
	real<lower=0,upper=1> phi_m;

	real<lower=0> cs_l;

	#real<lower=0,upper=1> phi_a;
	#real<lower=0,upper=1> phi_b;
	#real<lower=0,upper=1> phi_c;

}
parameters{
	#vector[K_A] B_a;
	#vector[K_B] B_b;
	#vector[K_C] B_c;
	#real B_c[K];
	#real B_c;
	
	#real<lower=0,upper=1> phi_a;
	#real<lower=0,upper=1> phi_b;
	#real<lower=0,upper=1> phi_c;
#	real<lower=0,upper=1> phi_m;
	
	real<lower=0,upper=1> as[N];	
	real<lower=0,upper=1> bs[N];
	real<lower=0> cs[N];
}
transformed parameters{

	# these are actual parameters for input into distributions
	#real<lower=0> s_a;
	#real<lower=0> s_b;		
	#real<lower=1> s_c;
	real<lower=0> s_m;		

	#real<lower=0,upper=1> m_as[N];
	#real<lower=0,upper=1> m_bs[N];
	#real<lower=0> m_cs[N];
	#for(i in 1:N){
	#      m_as[i] <- inv_logit(logit(pop_a) + dot_product(xas[i], B_a));		      
	#      m_bs[i] <- inv_logit(logit(pop_b) + dot_product(xbs[i], B_b));
	#      m_cs[i] <- exp(log(pop_c) + dot_product(xcs[i], B_c));
	      #m_cs[i] <- exp(log(pop_c) + xs[i,1]* B_c[1] + xs[i,1]* B_c[2]);
	      #print(B_c[1],B_c[2],xs[i,1],xs[i,2]);
	      #m_cs[i] <- exp(log(pop_c) + xs[i,2]* B_c[2]);
	      #m_cs[i] <- exp(log(pop_c) + dot_product(xs[i], B_c));
	      #m_cs[i] <- exp(log(pop_c) + dot_product(row(xs,i), B_c));
	      
	#}
	
	#s_a <- 1.0 / phi_a - 1;
	#s_b <- 1.0 / phi_b - 1;
	#s_c <- 1.0 / phi_c;

	s_m <- 1.0 / phi_m - 1;
	


}
model{
	int c;
	c <- 1;

	#B_a ~ normal(0, c_a);
	#B_b ~ normal(0, c_b);
	#B_c ~ normal(0, c_c);

	#phi_a ~ exponential(l_a);
	#phi_b ~ exponential(l_b);
	#phi_c ~ exponential(l_c);
	#phi_m ~ exponential(l_m);

	#for(i in 1:N){
	#      as[i] ~ beta(1.0 + (s_a * m_as[i]), 1.0 + (s_a * (1.0 - m_as[i])));
	#      bs[i] ~ beta(1.0 + (s_b * m_bs[i]), 1.0 + (s_b * (1.0 - m_bs[i])));
	#      cs[i] ~ gamma(s_c, (s_c - 1.0) / m_cs[i]);
	#}

	for(i in 1:N){
	      cs[i] ~ exponential(cs_l) T[0,50];
	}

	for(i in 1:N){
	      for(j in 1:ls[i]){
	      	    #vs[c] ~ normal(0, phi_m);	
		    #lp__ <- lp__ + normal_log(vs[c], as[i], phi_m);
	      	    #vs[c] ~ normal(m_as[i], 0.1);	
	      	    #vs[c] ~ normal(ss[i] * (1.0 - as[i] - (bs[i] * (1.0 - as[i]) * exp(-1.0 * ts[c] / cs[i]))), phi_m);
		    vs[c] ~ normal(ss[i] * (1.0 - as[i] - (bs[i] * (1.0 - as[i]) * exp(-1.0 * ts[c] / cs[i]))), phi_m);
		    #vs[c] ~ beta(1.0 + (s_m * (ss[i] * (1.0 - as[i] - (bs[i] * (1.0 - as[i]) * exp(-1.0 * ts[c] / cs[i]))))), 1.0 + (s_m * (ss[i] * (1.0 - (1.0 - as[i] - (bs[i] * (1.0 - as[i]) * exp(-1.0 * ts[c] / cs[i])))))));
		    #vs[c] ~ gamma(s_c, (s_c - 1.0) / m_cs[i]);
		    c <- c + 1;
	      }
	}

}