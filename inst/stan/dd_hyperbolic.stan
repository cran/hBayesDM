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
    real mu_k_pr;   # discounting rate
    real mu_beta_pr;  # inverse temperature
    
    real<lower=0> sd_k;
    real<lower=0> sd_beta;
    
    # subjective-level raw parameters, declare as vectors for vectorizing
    vector[N] k_pr;
    vector[N] beta_pr;    
}
transformed parameters {
    vector<lower=0,upper=1>[N] k;
    vector<lower=0,upper=5>[N] beta;
        
    for (i in 1:N) {
        k[i]    <- Phi_approx( mu_k_pr + sd_k * k_pr[i] );
        beta[i] <- Phi_approx( mu_beta_pr + sd_beta * beta_pr[i] ) * 5;
    }
}
model {
    # hyperparameters
    mu_k_pr    ~ normal(0, 1); 
    mu_beta_pr ~ normal(0, 1);
    
    sd_k    ~ cauchy(0, 5);
    sd_beta ~ cauchy(0, 5);
    
    # individual parameters
    k_pr    ~ normal(0, 1);
    beta_pr ~ normal(0, 1);
    
    for (i in 1:N) {
        real ev_later;
        real ev_sooner;
        
        for (t in 1:(Tsubj[i])) {
          ev_later   <- amount_later[i,t]  / ( 1 + k[i] * delay_later[i,t] );
          ev_sooner  <- amount_sooner[i,t] / ( 1 + k[i] * delay_sooner[i,t] );
          choice[i,t] ~ bernoulli_logit( beta[i] * (ev_later - ev_sooner) );
        }
    }
}

generated quantities {
    real<lower=0,upper=1> mu_k;
    real<lower=0,upper=5> mu_beta;
    real log_lik[N];

    mu_k    <- Phi_approx(mu_k_pr);
    mu_beta <- Phi_approx(mu_beta_pr) * 5;
    
    { # local section, this saves time and space
        for (i in 1:N) {
          real ev_later;
          real ev_sooner;

          log_lik[i] <- 0;
          
          for (t in 1:(Tsubj[i])) {
            ev_later   <- amount_later[i,t]  / ( 1 + k[i] * delay_later[i,t] );
            ev_sooner  <- amount_sooner[i,t] / ( 1 + k[i] * delay_sooner[i,t] );
            log_lik[i] <- log_lik[i] + bernoulli_logit_log( choice[i,t], beta[i] * (ev_later - ev_sooner) );
          }
        }
    }
}
