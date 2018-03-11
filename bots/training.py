# -*- coding: utf-8 -*-
"""
Created on Sun Mar 11 13:41:01 2018

@author: raphael
"""

import numpy as np

def get_train():
    a = np.loadtxt('train_data.csv')
    D,H,M,T,W = a.T
    x = np.array([D,H,M,T])
    y = np.array(W)
    return x,y

