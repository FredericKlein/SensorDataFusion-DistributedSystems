# -*- coding: utf-8 -*-
"""
Created on Fri May 12 10:01:03 2023

@author: {name: frederic klein, matriculation_number: 2735589}
"""
# This file contains the information about the sensors
import numpy as np
from TrackedObject import TrackedObject
from KalmanFilter import KalmanFilter, DistributedKalmanFilter

class Sensor():
    def __init__(self, H=np.array([[1, 0, 0, 0],
                                   [0, 1, 0, 0],
                                   [0, 0, 0, 0], 
                                   [0, 0, 0, 0]]), v=(np.array([0, 0]), 100*np.eye(4)), tracked_object=TrackedObject()):
        self.H = H
        self.v = v
        self.P = v[1]
        self.P_bar = v[1]
        self.pos = None
        self.KF = KalmanFilter()
        self.DKF = DistributedKalmanFilter()
        self.obj = tracked_object

    def get_measurement(self, gt):
        m = self.H @ gt
        n = self.get_noise(m)
        return m + n
    
    def get_noise(self, m):
        n = np.zeros_like(m)
        for i in range(self.H.shape[0]):
            if self.H[i, i] == 1:
                n[i] = np.random.normal(loc=[self.v[0][i]], scale=self.v[1][i,i])
        return n
        # we always carry around that the velocity in x and y is zero, so we cannot use this form:
        #return np.random.multivariate_normal(mean=np.zeros, cov=self.v)
    
    
    def single_step(self, gt=np.array([[0,0,10,12]]).T):
        info = {}
        z = self.get_measurement(gt)
        # get the kalman prediction
        if self.pos is None:
            # initialize first position of tracked object
            self.pos = z
        pred = self.KF.predicting(pos=self.pos, P=self.P, obj=self.obj)
        self.pos = pred[0]
        self.P = pred[1]
        info["predicted"] = {"x": self.pos, "P": self.P}
        
        # get the kalman filtration
        filtration = self.KF.filtering(pos=self.pos, P=self.P, z=z, H=self.H, R=self.v[1])
        self.pos = filtration[0]
        self.P = filtration[1]
        info["filtered"] = {"x": self.pos, "P": self.P}
        
        return info
    
    def single_step_federated(self, gt=np.array([[0,0,10,12]]).T, sensor_count=1):
        info = {}
        z = self.get_measurement(gt)
        # get the kalman predicition
        if self.pos is None:
            # initialize pposition of tracked object
            self.pos = z
        pred = self.DKF.predicting(pos=self.pos, P=self.P, obj=self.obj, P_bar=self.P, sensor_count=sensor_count)
        self.pos = pred[0]
        self.P = pred[1]
        info["predicted"] = {"x": self.pos, "P": self.P}
        
        # get the kalman filtration
        filtration = self.DKF.filtering(pos=self.pos, P=self.P, z=z, H=self.H, R=self.v[1])
        self.pos = filtration[0]
        self.P = filtration[1]
        info["filtered"] = {"x": self.pos, "P": self.P}
        
        return info
    
    def single_step_distributed(self, gt=np.array([[0,0,10,12]]).T, sensor_count=1):
        info = {}
        z = self.get_measurement(gt)
        # get the kalman predicition
        if self.pos is None:
            # initialize pposition of tracked object
            self.pos = z
        pred = self.DKF.predicting(pos=self.pos, P=self.P, obj=self.obj, P_bar=self.P_bar, sensor_count=sensor_count)
        self.pos = pred[0]
        self.P = pred[1]
        info["predicted"] = {"x": self.pos, "P": self.P}
        
        # get the kalman filtration
        filtration = self.DKF.filtering(pos=self.pos, P=self.P, z=z, H=self.H, R=self.v[1])
        self.pos = filtration[0]
        self.P = filtration[1]
        info["filtered"] = {"x": self.pos, "P": self.P}
        
        return info
    def globalization_step(self, sensor_count, fc_p):
        self.P_bar = sensor_count*fc_p
        self.pos = self.P_bar @ np.linalg.inv(self.P) @ self.pos
        
        
    def get_drawable_information(self, gt_array):
        original_info = self.get_information(gt_array)
        info_x = []
        info_y = []
        for f in original_info["filtered"]:
            info_x.append(f["x"][0][0])
            info_y.append(f["x"][1][0])
        return info_x, info_y
    
    def globalization(self, P_hat):
        self.pos = P_hat @ np.linalg.inv(self.P) @ self.pos
        self.P_hat = P_hat




