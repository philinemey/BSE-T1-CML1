#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Nov 28 13:41:18 2021

@author: Philine
"""

import pandas as pd
import numpy as np


"""
Model helper functions
"""

### Hyperparameter optimization ###
def tune_hyperparams(model, X, y, param_grid, n_iter=10, cv=5, scoring='accuracy'):
    import time
    from sklearn.model_selection import RandomizedSearchCV
    
    rs = RandomizedSearchCV(
        estimator = model,
        param_distributions = param_grid,
        n_iter = n_iter,
        cv=cv,
        n_jobs=-1,
        scoring=scoring,
        random_state=42
        )
    
    start_time = time.time()
    hyperparams = rs.fit(X, y)
    
    means = rs.cv_results_['mean_test_score']
    stds = rs.cv_results_['std_test_score']
    params = rs.cv_results_['params']
    for mean, stdev, param in zip(means, stds, params):
        print("%f (%f) with: %r" % (mean, stdev, param))
        
    # Summarize results:
    print("Best: %f using %s" % (hyperparams.best_score_, hyperparams.best_params_))
    print("Execution time: " + str((time.time() - start_time)) + ' ms')


### Correct for class imbalance ###
def reweight_binary(pi, q1=0.5, r1=0.5):
    r0 = 1-r1
    q0 = 1-q1
    tot = pi*(q1/r1)+(1-pi)*(q0/r0)
    w = pi*(q1/r1)
    w /= tot
    return w
    
def reweight_multi(pi,q,r=1/7):
    w = []
    q_r = [x / r for x in q]
    for n in range(0, len(pi+1)):
        tot = pi.loc[n]*pd.Series(q_r)
        tot_s = sum(tot)
        b = [x / tot_s for x in tot]
        w.append(b)
    w = np.array(w)
    return w
