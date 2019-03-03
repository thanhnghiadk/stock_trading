#!/usr/bin/env python2
# -*- coding: utf-8 -*-

import numpy as np
import pandas as pd


def rank(alpha_df):
    a = alpha_df.iloc[:,1:]
    a = a.rank(axis=1)
    for i in range(len(a)):
        if a.loc[i,:].count() > 1:
            a.loc[i,:] = [(ii-1)*1.0/(a.loc[i,:].count()-1) for ii in a.loc[i,:]]
    temp_df = alpha_df.copy()
    temp_df.iloc[:,1:] = a
    return temp_df

def winsorize(alpha_df, x):
    import scipy.stats
    a = alpha_df.iloc[:,1:]
    a = pd.DataFrame(scipy.stats.mstats.winsorize(a.values, limits=x).data, columns = a.columns)
    temp_df = alpha_df.copy()
    temp_df.iloc[:,1:] = a
    return temp_df

def exponential(alpha_df, x):
    a = alpha_df.iloc[:,1:]
    a = a.abs().pow(x).values*np.sign(a).values
    temp_df = alpha_df.copy()
    temp_df.iloc[:,1:] = a
    return temp_df

# TODO: update when alpha = 0
def normalize(alpha_df):
    a = alpha_df.iloc[:,1:]
#    a = a.sub(a.min(axis=1), axis=0)
    a = a.div(a.sum(axis=1), axis=0)
    temp_df = alpha_df.copy()
    temp_df.iloc[:,1:] = a
    return temp_df

def opt_decay(alpha_df,n):
    a = alpha_df.iloc[:,1:].reset_index(drop = True)
    a = a.fillna(0)
    temp = [1.0]
    for i in range(1,n):
        temp += [1.0/(i+1)]
    temp = pd.Series(temp)
    for i in range(n, a.shape[0]):
        a.iloc[i,:] = a.iloc[(i-n):i,:].multiply(temp, axis = 0).sum()
    temp_df = alpha_df.copy()
    temp_df.iloc[:,1:] = a
    return temp_df
        