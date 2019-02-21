#add following things before run
#conda install theano, conda install pygpu, pip install pymc3
#conda install -c anaconda numpy

import matplotlib.pyplot as plt
import numpy as np
import pymc3
import scipy.stats as stats
     
     
plt.style.use("ggplot")

#parameters 
n = 100
z = 30
alpha = 12
beta  = 12
alpha_post = 42
beta_post = 82

iterations = 100000

#use pymc3 to construct a model 
basic_model = pymc3.Model()
with basic_model:
    #prior belief
    theta = pymc3.Beta("theta", alpha = alpha, beta = beta)
    #Bernoulli likelihood 
    y = pymc3.Binomial("y", n=n, p = theta, observed=z)
    #carry out MCMC analysis with the Metropolis algorithm
    #using Maximum A Posteriori optimisation as initial values
    start = pymc3.find_MAP()
    
    #use the metropolis algorithm
    step = pymc3.Metropolis()
    
    #calculate the trace
    trace = pymc3.sample(iterations, step, start, random_seed =1, progressbar=True)
    
    #plot the posterior histogram from MCMC analysis
    bins = 50
    plt.hist(
        trace["theta"], bins,
        histtype="step", normed=True,
        label = "posterior(MCMC)", color = "red"
    )
    
    #plot the analytic prior and posterior beta distributions
    x = np.linspace(0, 1, 100)
    plt.plot(
        x, stats.beta.pdf(x, alpha, beta),
        "--", label="prior", color="blue"
    )
    
    plt.plot(
        x, stats.beta.pdf(x, alpha_post, beta_post),
        label="posterior(Analytic)",color="green"
    )
    
    #update the graph labesl
    plt.legend(title="parameters", loc="best")
    plt.xlabel("$\\theta$, Fairness")
    plt.ylabel("density")
    plt.show()


