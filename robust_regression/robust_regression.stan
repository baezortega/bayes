// Robust regression model in Stan
// Adrian Baez-Ortega, 2018
//
// To learn how to use this model, see
//    https://github.com/baezortega/bayes/tree/master/robust_regression

data {
    int<lower=1> N;  // number of observations
    real x[N];       // input data for the explanatory/independent variable
    real y[N];       // input data for the response/dependent variable
}

parameters {
    real alpha;           // intercept
    real beta;            // coefficient
    real<lower=0> sigma;  // scale of the t-distribution
    real<lower=0> nu;     // degrees of freedom of the t-distribution
    real y_rand;          // random samples from the t-distribution
}

transformed parameters {
    real mu[N] = alpha + beta * x;  // mean response
}

model {
    // Likelihood
    // Student's t-distribution instead of normal for robustness
    y ~ student_t(nu, mu, sigma);
    x ~ normal(0, 100000);
    
    // Uninformative priors on all parameters
    alpha ~ normal(0, 100000);
    beta ~ normal(0, 100000);
    sigma ~ uniform(0, 100000);
    nu ~ exponential(1/30.0);
    
    // Draw samples from the estimated t-distribution (for assessment of fit)
    y ~ student_t(nu, mu, sigma);
}
