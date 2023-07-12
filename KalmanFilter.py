# -*- coding: utf-8 -*-
"""
Created on Fri May 26 11:15:52 2023

@author: {name: frederic klein, matriculation_number: 2735589}
"""
# This file contains the Kalman Filtering algorithms
import numpy as np

class KalmanFilter():
    def __init__(self):
        pass
    
    def predicting(self, pos, P, obj):
        # standatd Kalman Filter Prediction step
        new_pos = obj.F @ pos
        new_P = obj.F @ P @ obj.F.T + obj.Q
        return new_pos, new_P
    
    def filtering(self, pos, P, z, H, R):
        # standard Kalman Filter Filtering step
        ny = z - H @ pos # innovation
        S = H @ P @ H.T + R
        W = P @ H.T @ np.linalg.inv(S)
        new_pos = pos + W @ ny
        new_P = P - W @ S @ W.T
        return new_pos, new_P

class DistributedKalmanFilter():
    def __init__(self):
        pass
    
    def predicting(self, pos, P, obj, P_bar, sensor_count):
        # Distributed/Federated Kalman Filter Prediction step
        new_pos = obj.F @ pos
        new_P = obj.F @ P_bar @ obj.F.T + sensor_count*obj.Q
        return new_pos, new_P
        
    def filtering(self, pos, P, z, H, R):
        # Distributed/Federated Kalman Filter Filtering step
        ny = z - H @ pos # innovation
        S = H @ P @ H.T + R
        W = P @ H.T @ np.linalg.inv(S)
        new_pos = pos + W @ ny
        new_P = P - W @ S @ W.T
        return new_pos, new_P