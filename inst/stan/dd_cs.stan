data {
    int<lower=1> N;
    int<lower=1> T;
    int<lower=1, upper=T> Tsubj[N];
    real<lower=0> delay_later[N,T];
    real<lower=0> amount_later[N,T];
    real<lower=0> delay_sooner[N,T];
    real<lower=0> amount_sooner[N,T];
    int<lower=0,upper=1> choice[N, T]; # 0 for instant reward, 1 for delayed reward
}
transformed data {
}
parameters {
    real mu_r_pr;   # (exponential) discounting rate
    real mu_s_pr;   # impatience
    real mu_beta_pr;  # inverse temperature
    
    real<lower=0> sd_r;
    real<lower=0> sd_s;
    real<lower=0> sd_beta;
    
    # subjective-level raw parameters, declare as vectors for vectorizing
    vector[N] r_pr;
    vector[N] s_pr;
    vector[N] beta_pr;    
}
transformed parameters {
    vector<lower=0,upper=1>[N] r;
    vector<lower=0,upper=10>[N] s;
    vector<lower=0,upper=5>[N] beta;
        
    for (i in 1:N) {
        r[i]    <- Phi_approx( mu_r_pr + sd_r * r_pr[i] );
        s[i]    <- Phi_approx( mu_s_pr + sd_s * s_pr[i] ) * 10;
        beta[i] <- Phi_approx( mu_beta_pr + sd_beta * beta_pr[i] ) * 5;
    }
}
model {
    # constant-sensitivity model (Ebert & Prelec, 2007)
    # hyperparameters
    mu_r_pr    ~ normal(0, 1);
    mu_s_pr    ~ normal(0, 1); 
    mu_beta_pr ~ normal(0, 1);
    
    sd_r    ~ cauchy(0, 2.5);  # reduced from 5 to 2.5
    sd_s    ~ cauchy(0, 2.5);  # reduced from 5 to 2.5
    sd_beta ~ cauchy(0, 2.5);  # reduced from 5 to 2.5
    
    # individual parameters
    r_pr    ~ normal(0, 1);
    s_pr    ~ normal(0, 1);
    beta_pr ~ normal(0, 1);
    
    for (i in 1:N) {
        real ev_later;
        real ev_sooner;
        
        for (t in 1:(Tsubj[i])) {
          ev_later <- amount_later[i,t] * exp(-1* ( pow(r[i] * delay_later[i,t], s[i]) ) ); 
          ev_sooner <- amount_sooner[i,t] * exp(-1* ( pow(r[i] * delay_sooner[i,t], s[i]) ) ); 
          choice[i,t] ~ bernoulli_logit( beta[i] * (ev_later - ev_sooner) );
        }
    }
}

generated quantities {
    real<lower=0,upper=1> mu_r;
    real<lower=0,upper=10> mu_s;
    real<lower=0,upper=5> mu_beta;
    real log_lik[N];

    mu_r    <- Phi_approx(mu_r_pr);
    mu_s    <- Phi_approx(mu_s_pr) * 10;
    mu_beta <- Phi_approx(mu_beta_pr) * 5;
    
    { # local section, this saves time and space
        for (i in 1:N) {
            real ev_later;
            real ev_sooner;
            
            log_lik[i] <- 0;
            
            for (t in 1:(Tsubj[i])) {
              ev_later <- amount_later[i,t] * exp(-1* ( pow(r[i] * delay_later[i,t], s[i]) ) ); 
              ev_sooner <- amount_sooner[i,t] * exp(-1* ( pow(r[i] * delay_sooner[i,t], s[i]) ) ); 
              log_lik[i] <- log_lik[i] + bernoulli_logit_log( choice[i,t], beta[i] * (ev_later - ev_sooner) );
            }
        }
    }
}
