// Robust correlation model in Stan
// Adrian Baez-Ortega, 2018
//
// To learn how to use this model, see
//    https://github.com/baezortega/bayes/tree/master/robust_correlation
//
// This model is based on Rasmus Bååth's, see
//    http://www.sumsar.net/blog/2013/08/robust-bayesian-estimation-of-correlation/
//    http://www.sumsar.net/blog/2013/08/bayesian-estimation-of-correlation/
//
// Thanks are due to Aki Vehtari for providing invaluable advice on how to improve this model

data {
    int<lower=1> N;  // number of observations
    vector[2] x[N];  // input data: rows are observations, columns are the two variables
}

parameters {
    vector[2] mu;                 // locations of the marginal t distributions
    real<lower=0> sigma[2];       // scales of the marginal t distributions
    real<lower=1> nu;             // degrees of freedom of the marginal t distributions
    real<lower=-1, upper=1> rho;  // correlation coefficient
}

transformed parameters {
    // Covariance matrix
    cov_matrix[2] cov = [[      sigma[1] ^ 2       , sigma[1] * sigma[2] * rho],
                         [sigma[1] * sigma[2] * rho,       sigma[2] ^ 2       ]];
}

model {
  // Likelihood
  // Bivariate Student's t-distribution instead of normal for robustness
  x ~ multi_student_t(nu, mu, cov);
    
  // Noninformative priors on all parameters
  sigma ~ normal(0,100);
  mu ~ normal(0, 100);
  nu ~ gamma(2, 0.1);
}

generated quantities {
  // Random samples from the estimated bivariate t-distribution (for assessment of fit)
  vector[2] x_rand;
  x_rand = multi_student_t_rng(nu, mu, cov);
}
