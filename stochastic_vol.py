#stochastic volatility model

import numpy as np 
import matplotlib.pyplot as plt
import seaborn as sns
sns.set_context('talk')
import pymc3 as pm
from pymc3.distribution.timeseries import GaussianRandomWalk

from scipy import optimize

