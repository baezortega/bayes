// Robust correlation model in Stan
// Adrian Baez-Ortega, 2018
//
// To learn how to use this model, see
//    https://github.com/baezortega/bayes/tree/master/robust_correlation
//
// This model is based on Rasmus Bååth's, see
//    http://www.sumsar.net/blog/2013/08/robust-bayesian-estimation-of-correlation/
//    http://www.sumsar.net/blog/2013/08/bayesian-estimation-of-correlation/

data {
    int<lower=1> N;  // number of observations
    matrix[N, 2] x;  // input data: rows are observations, columns are the two variables
}

parameters {
    vector[2] mu;                 // locations of the marginal t distributions
    real<lower=0> sigma[2];       // scales of the marginal t distributions
    real<lower=0> nu;             // degrees of freedom of the marginal t distributions
    real<lower=-1, upper=1> rho;  // correlation coefficient
    vector[2] x_rand;             // random samples from the bivariate t distribution
}

transformed parameters {
    // Covariance matrix
    cov_matrix[2] cov = [[      sigma[1] ^ 2       , sigma[1] * sigma[2] * rho],
                         [sigma[1] * sigma[2] * rho,       sigma[2] ^ 2       ]];
}

model {
    // Likelihood
    // Bivariate Student's t-distribution instead of normal for robustness
    for (n in 1:N) {
        x[n] ~ multi_student_t(nu, mu, cov);
    }
    
    // Uninformative priors on all parameters
    sigma ~ uniform(0, 100000);
    rho ~ uniform(-1, 1);
    mu ~ normal(0, 100000);
    nu ~ exponential(1/30.0);   // see http://doingbayesiandataanalysis.blogspot.co.uk/2015/12/prior-on-df-normality-parameter-in-t.html
    
    // Draw samples from the estimated bivariate t distribution (for assessment of fit)
    x_rand ~ multi_student_t(nu, mu, cov);
}
