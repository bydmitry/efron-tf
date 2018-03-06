"""
Validation of TensorFlow Efron likelihood with R survival package and Lifelines.

author: bydmitry
date: 25.02.2018
"""

import numpy as np
import pandas as pd

import tensorflow as tf
from keras import backend as K

from efrontf import efron_estimator_tf
from lifelines import CoxPHFitter
from rpy2 import robjects
from rpy2.robjects import pandas2ri
from rpy2.robjects.vectors import DataFrame
from rpy2 import robjects as ro
pandas2ri.activate()


r_code = '''
    library(survival)
    function(test.d){
        fit <- coxph(Surv(time, status) ~ x1, test.d, init=c(0.543), method = 'efron', iter.max=0)
        out = list( fit$linear.predictors, fit$loglik )
    }
'''
rfunc  = robjects.r(r_code)

N            = 1000
tie_ratio    = 0.7  # [0,1]
set_size     = int(np.round(N*tie_ratio))
censor_rate  = 0.3

for k in range(10):
    ts     = np.linspace(1, N, N)
    ts     = ts[ np.random.choice(N, set_size, replace=True) ]
    es     = np.random.binomial(1, (1-censor_rate), set_size)

    # Create a data-frame for R:
    df = pd.DataFrame({
            'time'   : ts,
            'status' : es,
            'x1'     : np.random.uniform(-1.0, 1.0, set_size)})


    # Normalize:
    df['x1'] = (df['x1'] - df['x1'].mean()) / df['x1'].std()

    # Compute likelihood with R:
    r_out  = rfunc( df )
    preds, r_lik  = np.asarray(r_out[0]), np.negative(np.round(r_out[1][0],4))
    tf_lik_r = K.eval( efron_estimator_tf(K.variable(ts), K.variable(es), K.variable(preds)) )

    # Compute ll with Lifelines:
    cp = CoxPHFitter()
    cp.fit(df, 'time', 'status', initial_beta=np.ones((1,1))*0.543, step_size=0.0)
    preds = cp.predict_log_partial_hazard(df.drop(['time', 'status'], axis=1)).values[:, 0]
    tf_lik_lifelines = K.eval( efron_estimator_tf(K.variable(ts), K.variable(es), K.variable(preds)) )

    print( 'TensorFlow w/ R: ', tf_lik_r )
    print( 'R-survival : ', r_lik )
    print( 'TensorFlow w/ lifelines: ', tf_lik_lifelines )
    print( 'Lifelines : ', np.negative(cp._log_likelihood), end='\n\n')

# done.
