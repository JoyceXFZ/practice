from math import exp, log, pi

def norm_pdf(x):
    return (1.0/((2*pi)**0.5))*exp(-0.5*x*x)
    
def norm_cdf(x):
    k = 1.0 / (1.0 + 0.2316419 * x)
    k_sum = k * (0.319381530 + k * (-0.356563782 + 
    k * (1.781477937 + k * (-1.821255978 + 1.330274429 * k))))
    
    if x >= 0.0:
        return (1.0 - (1.0 / ((2*pi)**0.5)) * exp(-0.5*x*x) * k_sum)
    else:
        return 1.0 - norm_cdf(-x)
       