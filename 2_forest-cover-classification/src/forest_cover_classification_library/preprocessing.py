#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Nov 28 13:40:43 2021

@author: Philine
"""

# @Roger: For this project I decided to focus on function-oriented programming

import pandas as pd


"""
Preprocessing helper functions
"""

### Get the same dummies for training and test data ###
def get_dummies(train_df, test_df, features):
        train_dummies = pd.get_dummies(train_df, columns = features, drop_first=True)
        test_dummies = pd.get_dummies(test_df, columns = features, drop_first=True)
        
        # Ensure that columns in training and test set are the same
        missing_cols = set(train_dummies.columns) - set(test_dummies.columns)
        
        # Add a missing column in test set with default value equal to 0
        for c in missing_cols:
          test_dummies[c] = 0
        
        # Ensure the order of column in the test set is in the same order than in train set
        test_dummies = test_dummies[train_dummies.columns]
        test_dummies = test_dummies.drop(columns=['Cover_Type'])
        
        return train_dummies, test_dummies
    

### Scale data ###
def scale(X_train, X_test):
    from sklearn.preprocessing import StandardScaler

    scaler = StandardScaler()
    scaler.fit(X_train) ## Fit to training set

    X_train_scaled = scaler.transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    return X_train_scaled, X_test_scaled