import numpy as np
import pandas as pd

import tensorflow as tf
from keras import backend as K

import matplotlib.pyplot as plt
from lifelines import CoxPHFitter
from lifelines.plotting import plot_lifetimes

from efrontf import efron_estimator_tf

# Dummy data:
observed_times = np.array([5,1,3,7,2,5,4,1,1])
censoring      = np.array([1,1,0,1,1,1,1,0,1])
predicted_risk = np.array([0.1,0.4,-0.2,0.2,-0.3,0.0,-0.1,0.3,-0.4])

# Plotting:
plt.xlim(0, 10)
plt.xlabel("time")
plot_lifetimes(observed_times, event_observed=censoring)

# get likelihood value:
K.eval(
    efron_estimator_tf(
        K.variable(observed_times),
        K.variable(censoring),
        K.variable(predicted_risk) )
)

# done
