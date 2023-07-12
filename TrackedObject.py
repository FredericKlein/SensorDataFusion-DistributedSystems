# -*- coding: utf-8 -*-
"""
Created on Fri May 26 11:11:42 2023

@author: {name: frederic klein, matriculation_number: 2735589}
"""
# This file contains the information about the tracked objects
import numpy as np

class TrackedObject():
    def __init__(self, F=np.array([[1, 0, 1, 0],
                                   [0, 1, 0, 1],
                                   [0 ,0, 1, 0],
                                   [0, 0, 0, 1]]), Q=100*np.eye(4)):
        self.F = F
        self.Q = Q
    
    def create_true_track(self, T=10):
        x = np.array([[-1000,-4500,20,200]]).T
        gt = [x]
        
        for i in range(T):
            gt.append(np.array([[1, 0, 1, 0],
                                [0, 1, 0, 1],
                                [0, 0, 1, 0],
                                [0, 0, 0, 1]]) @ gt[-1])
        self.gt = gt
        
    def get_xy_from_states(self):
        x = []
        y = []
        for t in self.gt:
            x.append(t[0])
            y.append(t[1])
        return x,y