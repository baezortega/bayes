# Function for running a robust correlation model in Stan
# Adrian Baez-Ortega, 2018
#
# To learn how to use this model, see
#    https://github.com/baezortega/bayes/tree/master/robust_correlation
#
# This model is based on Rasmus Bååth's, see 
#    http://www.sumsar.net/blog/2013/08/robust-bayesian-estimation-of-correlation/
#    http://www.sumsar.net/blog/2013/08/bayesian-estimation-of-correlation/

rob.cor.mcmc = function(x, iter = 6000, warmup = 1000, chains = 1) {
    
    library(rstan)
    library(coda)
    library(gridExtra)
    stopifnot(ncol(x) == 2)

    # Set up model data
    model.data = list(N=nrow(x), x=x)
    
    # Use robust estimates of the parameters as initial values
    model.init = list(mu = apply(x, 2, median),
                      sigma = apply(x, 2, mad),
                      rho = cor(x[, 1], x[, 2], method="spearman"))

    # Stan model definition
    stan.model = "
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
        // Bivariate Student-t distribution used instead of normal for robustness
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
    }"
    
    # Run the model
    stan.cor = stan(model_name="robust_regression",
                    model_code=stan.model,
                    data=model.data, init=rep(list(model.init), chains), 
                    seed=210191, iter=iter, warmup=warmup, chains=chains)
    
    # Obtain the MCMC samples of rho
    stan.rho = extract(stan.cor, "rho")[[1]]
    hpd95 = HPDinterval(as.mcmc(as.numeric(stan.rho)), prob=0.95)
    hpd99 = HPDinterval(as.mcmc(as.numeric(stan.rho)), prob=0.99)
    
    # Plot trace and posterior
    p1 = stan_trace(stan.cor, pars="rho")
    p2 = stan_dens(stan.cor, pars="rho") +
        geom_line(data=data.frame(t(hpd95)), 
                  aes(x=var1, y=c(0,0), col="95% HPD"), size=2) +
        scale_colour_manual(name="", values=c(`95% HPD`="steelblue3"))
    grid.arrange(p1, p2)

    # Write some descriptive statistics
    cat("POSTERIOR STATISTICS OF RHO\n",
        "Posterior mean and standard deviation:     Mean = ",
          mean(stan.rho), ", SD = ", sd(stan.rho), "\n",
        "Posterior median and MAD:                  Median = ",
          median(stan.rho), ", MAD = ", mad(stan.rho), "\n",
        "Rho values with 99% posterior probability: 99% HPDI = [", 
          hpd99[,"lower"], ", ", hpd99[,"upper"], "]\n",
        "Rho values with 95% posterior probability: 95% HPDI = [", 
          hpd95[,"lower"], ", ", hpd95[,"upper"], "]\n",
        "Posterior probability that rho is ≤0:      P(rho ≤ 0) = ", 
          mean(stan.rho <= 0), "\n",
        "Posterior probability that rho is ≥0:      P(rho ≥ 0) = ", 
          mean(stan.rho >= 0), "\n",
        "Posterior probability that rho is weak:    P(-0.1 < rho < 0.1) = ", 
          mean(stan.rho > -0.1 & stan.rho < 0.1), "\n\n",
        sep="")
    
    # Return stanfit object
    return(stan.cor)
    
}