// Robust regression model in Stan
// Adrian Baez-Ortega, 2018
//
// To learn how to use this model, see
//    https://github.com/baezortega/bayes/tree/master/robust_regression

data {
    int<lower=1> N;    // number of observations
    int<lower=0> M;    // number of values for credible interval estimation
    int<lower=0> P;    // number of values to predict
    vector[N] x;       // input data for the explanatory (independent) variable
    vector[N] y;       // input data for the response (dependent) variable
    vector[M] x_cred;  // x-values for credible interval estimation (should cover range of x)
    vector[P] x_pred;  // x-values for prediction
}

parameters {
    real alpha;           // intercept
    real beta;            // coefficient
    real<lower=0> sigma;  // scale of the t-distribution
    real<lower=1> nu;     // degrees of freedom of the t-distribution
}

transformed parameters {
    vector[N] mu = alpha + beta * x;            // mean response
    vector[M] mu_cred = alpha + beta * x_cred;  // mean response for credible interval estimation
    vector[P] mu_pred = alpha + beta * x_pred;  // mean response for prediction
}

model {
    // Likelihood
    // Student's t-distribution instead of normal for robustness
    y ~ student_t(nu, mu, sigma);
    
    // Uninformative priors on all parameters
    alpha ~ normal(0, 1000);
    beta ~ normal(0, 1000);
    sigma ~ normal(0, 1000);
    nu ~ gamma(2, 0.1);
}

generated quantities {
    // Sample from the t-distribution at the values to predict (for prediction)
    real y_pred[P];
    for (p in 1:P) {
        y_pred[p] = student_t_rng(nu, mu_pred[p], sigma);
    }
}
