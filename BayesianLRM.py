import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import pymc3 as pm

sns.set(style = "darkgrid", palette = "muted")

def simulate_linear_data(N, beta_0, beta_1, eps_sigma_sq):
    df = pd.DataFrame(
     {"x":
        np.random.RandomState(42).choice(
            map(lambda x: float(x)/100.0, 
                np.arange(N)
                ), N, replace=False)
     }   
    )

    eps_mean = 0.0
    df["y"] = beta_0 + beta_1*df["x"] + np.random.RandomState(42).normal(eps_mean, eps_sigma_sq, N)

    return df

def glm_mcmc_inference(df, iterations=5000):
        basic_model = pm.Model()
        with basic_model:
            pm.glm.GLM.from_formula("y ~ x", df, family=pm.glm.families.Normal())
            
            start = pm.find_MAP()
            
            step = pm.NUTS()
            
            trace = pm.sample(
                iterations, step, start, 
                random_seed=42, progressbar=True
            )
        return trace    
    
if __name__ == "__main__":

    beta_0 = 1.0
    beta_1 = 2.0
    N = 200
    eps_sigma_sq = 0.5

    df = simulate_linear_data(N, beta_0, beta_1, eps_sigma_sq)

    sns.lmplot(x="x", y="y", data=df, size=10)
    plt.xlim(0.0, 1.0)
    
    trace = glm_mcmc_inference(df, iterations=5000)
    pm.traceplot(trace[500:])
    plt.show()
    
    sns.lmplot(x="x", y="y", data=df, size=10, fit_reg=False)
    plt.xlim(0.0, 1.0)
    plt.ylim(0.0, 4.0)
    pm.plot_posterior_predictive_glm(trace, samples=100)  
    x = np.linspace(0, 1, N)
    y = beta_0 + beta_1 * x 
    plt.plot(x, y, label="True Regression Line", lw=3., c ="green")
    plt.legend(loc=0)
    plt.show()
    