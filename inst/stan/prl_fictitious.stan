data {
    int<lower=1> N;
    int<lower=1> T;
    int<lower=1, upper=T> Tsubj[N];
    int<lower=1,upper=2> choice[N, T];
    real rewlos[N, T];
}
transformed data {
    vector[2] initV;
    initV  <- rep_vector(0.0, 2);
}
parameters {
    real mu_eta_pr;   # learning rate
    real mu_alpha_pr; # indecision point
    real mu_beta_pr;  # inverse temperature
    
    real<lower=0> sd_eta;
    real<lower=0> sd_alpha;
    real<lower=0> sd_beta;
    
    # subjective-level raw parameters, declare as vectors for vectorizing
    vector[N] eta_pr;
    vector[N] alpha_pr;
    vector[N] beta_pr;    
}

transformed parameters {
    vector<lower=0,upper=1>[N] eta;
    vector<lower=0,upper=1>[N] alpha;
    vector<lower=0,upper=5>[N] beta;
        
    for (i in 1:N) {
      eta[i]    <- Phi_approx( mu_eta_pr + sd_eta * eta_pr[i] );
      alpha[i]  <- Phi_approx( mu_alpha_pr + sd_alpha * alpha_pr[i] );
      beta[i]   <- Phi_approx( mu_beta_pr + sd_beta * beta_pr[i] ) * 5;
    }
}

model {
    # hyperparameters
    mu_eta_pr    ~ normal(0,1); 
    mu_alpha_pr  ~ normal(-1.9, 0.2); # so that Phi_approx(mu_alpha_pr) is about 0.015 - 0.045
    mu_beta_pr   ~ normal(0,1);
    
    sd_eta    ~ cauchy(0,5);
    sd_alpha  ~ cauchy(0,5);
    sd_beta   ~ cauchy(0,5);
    
    # individual parameters
    eta_pr    ~ normal(0,1);
    alpha_pr  ~ normal(0,1);
    beta_pr   ~ normal(0,1);
    
    for (i in 1:N) {
        vector[2] ev;
        vector[2] prob;
        real pe;     # prediction error
        real penc;   # fictitious prediction error (pe-non-chosen)

        ev <- initV; # initial ev values
        
        for (t in 1:(Tsubj[i])) {
            # compute action probabilities
            prob[1] <- 1 / (1 + exp( beta[i] * (alpha[i] - (ev[1] - ev[2])) ));
            prob[2] <- 1 - prob[1];
            choice[i,t] ~ categorical( prob );

            # prediction error
            pe   <-  rewlos[i,t] - ev[choice[i,t]];
            penc <- -rewlos[i,t] - ev[3-choice[i,t]];

            # value updating (learning)
            ev[choice[i,t]]   <- ev[choice[i,t]]   + eta[i] * pe; 
            ev[3-choice[i,t]] <- ev[3-choice[i,t]] + eta[i] * penc;
        }
    }
}

generated quantities {
    real<lower=0,upper=1> mu_eta;
    real<lower=0,upper=1> mu_alpha;
    real<lower=0,upper=5> mu_beta;
    real log_lik[N];

    mu_eta    <- Phi_approx(mu_eta_pr);
    mu_alpha  <- Phi_approx(mu_alpha_pr);
    mu_beta   <- Phi_approx(mu_beta_pr) * 5;
    
    { # local section, this saves time and space
        for (i in 1:N) {
            vector[2] ev;
            vector[2] prob;
            real pe;     # prediction error
            real penc;   # fictitious prediction error (pe-non-chosen)

            log_lik[i] <- 0;
            ev <- initV; # initial ev values
            
            for (t in 1:(Tsubj[i])) {
                # compute action probabilities
                prob[1] <- 1 / (1 + exp( beta[i] * (alpha[i] - (ev[1] - ev[2])) ));
                prob[2] <- 1 - prob[1];

                log_lik[i] <- log_lik[i] + categorical_log( choice[i, t], prob );

                # prediction error
                pe   <-  rewlos[i,t] - ev[choice[i,t]];
                penc <- -rewlos[i,t] - ev[3-choice[i,t]];

                # value updating (learning)
                ev[choice[i,t]]   <- ev[choice[i,t]]   + eta[i] * pe; 
                ev[3-choice[i,t]] <- ev[3-choice[i,t]] + eta[i] * penc;
            }
        }
    }
}
