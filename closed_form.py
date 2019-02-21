from math import exp, log, sqrt
from statistics import norm_pdf, norm_cdf

def d_j(j, S, K, r, v, T):
    
    return ((log(S/K) + (r + (-1)**(j-1)*0.5*v*v) * T)/(v*(T**0.5)))
    
def vanilla_call_prices(S, K, r, v, T):
    return S * norm_cdf(d_j(1, S, K, r, v, T)) - \
        K*exp(-r*T) * norm_cdf(d_j(2, S, K, r, v, T))

def vanilla_put_prices(S, K, r, v, T):
    return -S * norm_cdf(-d_j(1, S, K, r, v, T)) + \
        K*exp(-r*T) * norm_cdf(-d_j(2, S, K, r, v, T))
        
S = 100.0
K = 100.0
r = 0.025
v = 0.2
T = 1

c = vanilla_call_prices(S, K, r, v, T)
p = vanilla_put_prices(S, K, r, v, T)
print c, p, (c-p), (S - K*exp(-r*T))