data {
  int<lower=1> N;
  int<lower=1> T;               
  int<lower=1, upper=T> Tsubj[N];                 
  int<lower=1,upper=2> choice[N,T];     
  real rewlos[N,T];  # no lower and upper bounds   
}

transformed data {
  vector[2] initV;  # initial values for V
  initV <- rep_vector(0.0, 2);
}

parameters {
  # group-level parameters
  real mu_lr_pr;    # learning rate
  real mu_tau_pr;   # inverse temperature

  real<lower=0> sd_lr;
  real<lower=0> sd_tau;
  
  # subject-level raw parameters, Matt Trick
  vector[N] lr_raw;
  vector[N] tau_raw;
}

transformed parameters {
  # subject-level parameters
  vector<lower=0,upper=1>[N] lr;
  vector<lower=0,upper=5>[N] tau;
  
  for (i in 1:N) {
    lr[i]  <- Phi_approx( mu_lr_pr  + sd_lr  * lr_raw[i] );
    tau[i] <- Phi_approx( mu_tau_pr + sd_tau * tau_raw[i] ) * 5;
  }
}

model {
  # hyperparameters
  mu_lr_pr  ~ normal(0,1);
  mu_tau_pr ~ normal(0,1);
  sd_lr     ~ cauchy(0,5);
  sd_tau    ~ cauchy(0,5);
  
  # individual parameters
  lr_raw  ~ normal(0,1);
  tau_raw ~ normal(0,1);
  
  # subject loop and trial loop
  for (i in 1:N) {
    vector[2] v; # expected value
    real pe;     # prediction error

    v <- initV;

    for (t in 1:(Tsubj[i])) {        
      # compute action probabilities
      choice[i,t] ~ categorical_logit( tau[i] * v );

      # prediction error 
      pe <- rewlos[i,t] - v[choice[i,t]];

      # value updating (learning) 
      v[choice[i,t]] <- v[choice[i,t]] + lr[i] * pe; 
    }
  }
}

generated quantities {
  real<lower=0,upper=1> mu_lr; 
  real<lower=0,upper=5> mu_tau;
  real log_lik[N]; 

  mu_lr  <- Phi_approx(mu_lr_pr);
  mu_tau <- Phi_approx(mu_tau_pr) * 5;

  { # local section, this saves time and space
    for (i in 1:N) {
      vector[2] v; # expected value
      real pe;     # prediction error

      log_lik[i] <- 0;
      v <- initV;

      for (t in 1:(Tsubj[i])) {
        # compute action probabilities
        log_lik[i] <- log_lik[i] + categorical_logit_log(choice[i,t], tau[i] * v);
        
        # prediction error 
        pe <-  rewlos[i,t] - v[choice[i,t]];

        # value updating (learning) 
        v[choice[i,t]] <- v[choice[i,t]] + lr[i] * pe; 
      }
    }   
  }
}
