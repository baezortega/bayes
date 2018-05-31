# Function for running a robust correlation model in Stan
# Adrian Baez-Ortega, 2018
#
# To learn how to use this model, see
#    https://github.com/baezortega/bayes/tree/master/robust_correlation
#
# This model is based on Rasmus Bååth's, see 
#    http://www.sumsar.net/blog/2013/08/robust-bayesian-estimation-of-correlation/
#    http://www.sumsar.net/blog/2013/08/bayesian-estimation-of-correlation/
#
# Thanks are due to Aki Vehtari for providing invaluable advice on how to 
# improve this model
#
# Note: the model is compiled when sampling is done for the first time.
# Some unimportant warning messages might show up during compilation.

rob.cor.mcmc = function(x, iter = 2000, warmup = 500, chains = 4) {
    
    library(rstan)
    library(coda)
    library(gridExtra)
    stopifnot(ncol(x) == 2)

    # Set up model data
    model.data = list(N=nrow(x), x=x)

    # Stan model definition
    stan.model = "
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
            sigma ~ normal(0, 1000);
            mu ~ normal(0, 1000);
            nu ~ gamma(2, 0.1);
        }
        
        generated quantities {
            // Random samples from the estimated bivariate t-distribution (for assessment of fit)
            vector[2] x_rand;
            x_rand = multi_student_t_rng(nu, mu, cov);
        }"
    
    # Run the model
    stan.cor = stan(model_name="robust_regression",
                    model_code=stan.model, data=model.data,
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