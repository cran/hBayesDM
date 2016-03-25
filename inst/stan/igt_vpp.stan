data {
  int<lower=1> N;
  int<lower=1> T;
  int<lower=1, upper=T> Tsubj[N];
  real rewlos[N, T];
  int ydata[N, T];
}
transformed data {
  vector[4] initV;
  initV  <- rep_vector(0.0, 4);
}
parameters {
  real mu_A_pr;
  real mu_alpha_pr;
  real mu_cons_pr;
  real mu_lambda_pr;
  real mu_epP_pr;
  real mu_epN_pr;
  real mu_K_pr;
  real mu_w_pr;

  real<lower=0> sd_A;
  real<lower=0> sd_alpha;
  real<lower=0> sd_cons;
  real<lower=0> sd_lambda;
  real<lower=0> sd_epP;
  real<lower=0> sd_epN;
  real<lower=0> sd_K;
  real<lower=0> sd_w;

  vector[N] A_pr;
  vector[N] alpha_pr;
  vector[N] cons_pr;
  vector[N] lambda_pr;
  vector[N] epP_pr;
  vector[N] epN_pr;
  vector[N] K_pr;
  vector[N] w_pr;
}
transformed parameters {
  vector<lower=0, upper=1>[N] A;
  vector<lower=0, upper=2>[N] alpha;
  vector<lower=0, upper=5>[N] cons;
  vector<lower=0, upper=10>[N] lambda;
  vector[N] epP;
  vector[N] epN;
  vector<lower=0, upper=1>[N] K;
  vector<lower=0, upper=1>[N] w;

  for (i in 1:N) {
    A[i] <- Phi_approx( mu_A_pr + sd_A * A_pr[i] );
    alpha[i] <- 2 * Phi_approx( mu_alpha_pr + sd_alpha * alpha_pr[i] );
    cons[i] <- 5 * Phi_approx( mu_cons_pr + sd_cons * cons_pr[i] );
    lambda[i] <- 10 * Phi_approx( mu_lambda_pr + sd_lambda * lambda_pr[i] );
    K[i] <- Phi_approx( mu_K_pr + sd_K * K_pr[i] );
    w[i] <- Phi_approx( mu_w_pr + sd_w * w_pr[i] );
  }
  epP <- mu_epP_pr + sd_epP * epP_pr;
  epN <- mu_epN_pr + sd_epN * epN_pr;
}
model {
  mu_A_pr ~ normal(0, 1);
  mu_alpha_pr ~ normal(0, 1);
  mu_cons_pr ~ normal(0, 1);
  mu_lambda_pr ~ normal(0, 1);
  mu_epP_pr ~ normal(0, 5);
  mu_epN_pr ~ normal(0, 5);
  mu_K_pr ~ normal(0, 1);
  mu_w_pr ~ normal(0, 1);

  sd_A ~ cauchy(0,5);
  sd_alpha ~ cauchy(0,5);
  sd_cons ~ cauchy(0,5);
  sd_lambda ~ cauchy(0,5);
  sd_epP ~ cauchy(0,5);
  sd_epN ~ cauchy(0,5);
  sd_K ~ cauchy(0,5);
  sd_w ~ cauchy(0,5);

  # Matt trick: all dists. should be normal(0,1)
  A_pr ~ normal(0, 1.0);
  alpha_pr ~ normal(0, 1.0);
  cons_pr ~ normal(0, 1.0);
  lambda_pr ~ normal(0, 1.0);
  epP_pr ~ normal(0, 1.0);
  epN_pr ~ normal(0, 1.0);
  K_pr ~ normal(0, 1.0);
  w_pr ~ normal(0, 1.0);

  for (i in 1:N) {
    vector[4] ev;
    vector[4] p_next;
    vector[4] str;
    vector[4] pers;   # perseverance
    vector[4] V;   # weighted sum of ev and pers

    real curUtil;     # utility of curFb
    real theta;       # theta = 3^c - 1
    
    theta <- pow(3, cons[i]) -1;
    ev <- initV; # initial ev values
    pers <- initV; # initial pers values

    for (t in 1:(Tsubj[i]-1)) {
      pers <- pers * K[i]; # decay
      #for (d in 1:4) {
      #  pers[d] <- pers[d] * K[i];   # decay
      #}
      if ( rewlos[i,t] >= 0) {  # x(t) >= 0
        curUtil <- pow(rewlos[i,t], alpha[i]);
        pers[ ydata[i,t] ] <- pers[ ydata[i,t] ] + epP[i];  # perseverance term
      } else {                  # x(t) < 0
        curUtil <- -1 * lambda[i] * pow( -1*rewlos[i,t], alpha[i]);
        pers[ ydata[i,t] ] <- pers[ ydata[i,t] ] + epN[i];  # perseverance term
      }

      ev[ ydata[i, t] ] <- ev[ ydata[i, t] ] + A[i] * (curUtil - ev[ ydata[i, t] ] );
      # calculate V
      V <- w[i]*ev + (1-w[i])*pers;
      
      #for (d in 1:4) {
      #  str[d] <- exp( theta * (w[i]*ev[d]+(1-w[i])*pers[d]) );
      #  #str[d] <- exp( theta * ev[d] );
      #}
      #for (d in 1:4) {
      #  p_next[d] <- str[d] / sum(str);
      #}
      #ydata[i, t+1] ~ categorical( p_next );
      
      # softmax choice
      ydata[i, t+1] ~ categorical_logit( theta * V );
      
    }
  }
}

generated quantities {
  real<lower=0,upper=1> mu_A;
  real<lower=0,upper=2> mu_alpha;
  real<lower=0,upper=5> mu_cons;
  real<lower=0,upper=10> mu_lambda;
  real mu_epP;
  real mu_epN;
  real<lower=0,upper=1> mu_K;
  real<lower=0,upper=1> mu_w;
  real log_lik[N];

  mu_A <- Phi_approx(mu_A_pr);
  mu_alpha <- Phi_approx(mu_alpha_pr) * 2;
  mu_cons <- Phi_approx(mu_cons_pr) * 5;
  mu_lambda <- Phi_approx(mu_lambda_pr) * 10;
  mu_epP <- mu_epP_pr;
  mu_epN <- mu_epN_pr;
  mu_K <- Phi_approx(mu_K_pr);
  mu_w <- Phi_approx(mu_w_pr);
  
  { # local section, this saves time and space
    for (i in 1:N) {
      vector[4] ev;
      vector[4] p_next;
      vector[4] str;
      vector[4] pers;   # perseverance
      vector[4] V;   # weighted sum of ev and pers
  
      real curUtil;     # utility of curFb
      real theta;       # theta = 3^c - 1
      log_lik[i] <- 0;
            
      theta <- pow(3, cons[i]) -1;
      ev <- initV; # initial ev values
      pers <- initV; # initial pers values
  
      for (t in 1:(Tsubj[i]-1)) {
        pers <- pers * K[i]; # decay
        #for (d in 1:4) {
        #  pers[d] <- pers[d] * K[i];   # decay
        #}
        if ( rewlos[i,t] >= 0) {  # x(t) >= 0
          curUtil <- pow(rewlos[i,t], alpha[i]);
          pers[ ydata[i,t] ] <- pers[ ydata[i,t] ] + epP[i];  # perseverance term
        } else {                  # x(t) < 0
          curUtil <- -1 * lambda[i] * pow( -1*rewlos[i,t], alpha[i]);
          pers[ ydata[i,t] ] <- pers[ ydata[i,t] ] + epN[i];  # perseverance term
        }
  
        ev[ ydata[i, t] ] <- ev[ ydata[i, t] ] + A[i] * (curUtil - ev[ ydata[i, t] ] );
        # calculate V
        V <- w[i]*ev + (1-w[i])*pers;
        
        #for (d in 1:4) {
        #  str[d] <- exp( theta * (w[i]*ev[d]+(1-w[i])*pers[d]) );
        #  #str[d] <- exp( theta * ev[d] );
        #}
        #for (d in 1:4) {
        #  p_next[d] <- str[d] / sum(str);
        #}
        #ydata[i, t+1] ~ categorical( p_next );
        
        # softmax choice
        #ydata[i, t+1] ~ categorical_logit( theta * V );
        # softmax choice
        log_lik[i] <- log_lik[i] + categorical_logit_log( ydata[i, t+1], theta * V );
      }
    }
  }  
}
