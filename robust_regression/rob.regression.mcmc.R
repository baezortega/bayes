# Function for running a robust regression model in Stan
# Adrian Baez-Ortega, 2018
#
# To learn how to use this model, see
#    https://github.com/baezortega/bayes/tree/master/robust_regression
#
# Note: the model is compiled when sampling is done for the first time.
# Some unimportant warning messages might show up during compilation.

rob.regression.mcmc = function(x, y, x.pred = NULL, pred.int = 90, cred.int = 95, iter = 2000, 
                               warmup = 500, chains = 4, seed = 911, xlab = "x", ylab = "y") {
    
    library(rstan)
    library(coda)
    
    predicting = TRUE
    if (!(pred.int %in% c(90, 95, 99))) {
        stop("`pred.int` only admits values 90, 95 or 99.")
    }
    if (!(cred.int %in% c(90, 95, 99))) {
        stop("`cred.int` only admits values 90, 95 or 99.")
    }
    if (is.null(x.pred)) {
        x.pred = numeric(0)
        predicting = FALSE
    }
    if (length(x.pred) == 1) {
        x.pred = as.array(x.pred)
    }
    
    # Generate range of x values for estimation of credible interval of the mean response
    M = 20
    x.cred = seq(min(x), max(x), length.out=M)
    
    # Set up model data
    model.data = list(x=x, y=y, N=length(x),
                      x_cred=x.cred, M=M,
                      x_pred=x.pred, P=length(x.pred))

    # Stan model definition
    stan.model = "
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
    }"
    
    # Run the model
    stan.reg = stan(model_name="robust_regression",
                    model_code=stan.model,
                    data=model.data, seed=seed, 
                    iter=iter, warmup=warmup, chains=chains)
    
    # Extract MCMC samples
    stan.alpha = extract(stan.reg, "alpha")[[1]]
    stan.beta = extract(stan.reg, "beta")[[1]]
    stan.cred = extract(stan.reg, "mu_cred")[[1]]
    if (predicting) {
        stan.ypred = extract(stan.reg, "y_pred")[[1]]
    }
    
    # Obtain HPD intervals
    hpd.alpha = list("95"=NULL, "99"=NULL)
    hpd.beta = list("95"=NULL, "99"=NULL)
    hpd.alpha$`95` = HPDinterval(as.mcmc(as.numeric(stan.alpha)), prob=0.95)
    hpd.alpha$`99` = HPDinterval(as.mcmc(as.numeric(stan.alpha)), prob=0.99)
    hpd.beta$`95` = HPDinterval(as.mcmc(as.numeric(stan.beta)), prob=0.95)
    hpd.beta$`99` = HPDinterval(as.mcmc(as.numeric(stan.beta)), prob=0.99)
    
    if (predicting) {
        median.ypred = apply(stan.ypred, 2, median)
        hpd.ypred = list("90"=NULL, "95"=NULL, "99"=NULL)
        hpd.ypred$`90` = apply(stan.ypred, 2, function(y) HPDinterval(as.mcmc(as.numeric(y)), prob=0.90))
        hpd.ypred$`95` = apply(stan.ypred, 2, function(y) HPDinterval(as.mcmc(as.numeric(y)), prob=0.95))
        hpd.ypred$`99` = apply(stan.ypred, 2, function(y) HPDinterval(as.mcmc(as.numeric(y)), prob=0.99))
    }
    
    hpd95.mu = list(upper=NULL, lower=NULL)
    for (j in 1:ncol(stan.cred)) {
        hpd = HPDinterval(as.mcmc(as.numeric(stan.cred[,j])), prob=cred.int / 100)
        hpd95.mu$upper = c(hpd95.mu$upper, hpd[,"upper"])
        hpd95.mu$lower = c(hpd95.mu$lower, hpd[,"lower"])
    }
    
    # Plot regression line, credible intervals and prediction intervals
    if (predicting) {
        y.min = min(y, hpd95.mu$lower, hpd.ypred$`90`[1,])
        y.max = max(y, hpd95.mu$upper, hpd.ypred$`90`[2,])
    }
    else {
        y.min = min(y, hpd95.mu$lower)
        y.max = max(y, hpd95.mu$upper)
    }
    plot(x, y, ylim=c(y.min, y.max), type="n", xlab=xlab, ylab=ylab)
    polygon(x=c(x.cred, rev(x.cred)), y=c(hpd95.mu$lower, rev(hpd95.mu$upper)), border=NA, col="grey80")
    segments(x0=min(x), y0=median(stan.alpha) + median(stan.beta) * min(x), 
             x1=max(x), y1=median(stan.alpha) + median(stan.beta) * max(x), lwd=2.5)
    points(x, y, pch=16, col="dodgerblue4", cex=1.2)
    if (predicting) {
        points(x=x.pred, y=median.ypred, pch=16, col="darkorange2", cex=1.2)
        arrows(x0=x.pred, y0=hpd.ypred[[as.character(pred.int)]][1,], 
               x1=x.pred, y1=hpd.ypred[[as.character(pred.int)]][2,], 
               length=0, col="darkorange2", lwd=2.5)
    }
    
    # Write some descriptive statistics
    cat("SIMPLE LINEAR REGRESSION: POSTERIOR STATISTICS\n",
        "Intercept (alpha)\n",
        "   Posterior mean and standard deviation: Mean = ", mean(stan.alpha), ", SD = ", sd(stan.alpha), "\n",
        "   Posterior median and MAD:              Median = ", median(stan.alpha), ", MAD = ", mad(stan.alpha), "\n",
        "   Values with 95% posterior probability: 95% HPDI = [", hpd.alpha$`95`[,"lower"], ", ", hpd.alpha$`95`[,"upper"], "]\n",
        "   Values with 99% posterior probability: 99% HPDI = [", hpd.alpha$`99`[,"lower"], ", ", hpd.alpha$`99`[,"upper"], "]\n\n",
        "Slope (beta)\n",
        "   Posterior mean and standard deviation: Mean = ", mean(stan.beta), ", SD = ", sd(stan.beta), "\n",
        "   Posterior median and MAD:              Median = ", median(stan.beta), ", MAD = ", mad(stan.beta), "\n",
        "   Values with 95% posterior probability: 95% HPDI = [", hpd.beta$`95`[,"lower"], ", ", hpd.beta$`95`[,"upper"], "]\n",
        "   Values with 99% posterior probability: 99% HPDI = [", hpd.beta$`99`[,"lower"], ", ", hpd.beta$`99`[,"upper"], "]\n\n",
        sep="")

    if (predicting) {
        for (j in 1:length(x.pred)) {
            cat("Predicted value #", j, " (x = ", x.pred[j], ")\n",
                "   Posterior median predicted response:   Median = ", median.ypred[j], "\n",
                "   Values with 90% posterior probability: 90% HPDI = [", hpd.ypred$`90`[1, j], ", ", hpd.ypred$`90`[2, j], "]\n",
                "   Values with 95% posterior probability: 95% HPDI = [", hpd.ypred$`95`[1, j], ", ", hpd.ypred$`95`[2, j], "]\n",
                "   Values with 99% posterior probability: 99% HPDI = [", hpd.ypred$`99`[1, j], ", ", hpd.ypred$`99`[2, j], "]\n\n",
                sep="")
        }
    }
    
    # Return stanfit object
    return(stan.reg)
    
}