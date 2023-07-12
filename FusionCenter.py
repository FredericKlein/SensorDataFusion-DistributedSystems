# -*- coding: utf-8 -*-
"""
Created on Fri May 26 11:07:25 2023

@author: {name: frederic klein, matriculation_number: 2735589}
"""
import numpy as np
from tqdm import tqdm
import itertools

class FusionCenter():
    def __init__(self):
        pass

    def get_xy_from_track(self, track):
        x = []
        y = []
        for t in track:
            x.append(t["x"][0])
            y.append(t["x"][1])
        return x, y
    
    def naive_fusion(self, T, car, sensors):
        track = []
        for t in tqdm(range(T)):
            # initialize x and P for timestep t
            fc_x = "None" #np.zeros_like(sensor_results[0]["filtered"][t]["x"])
            fc_p = "None" #np.zeros_like(sensor_results[0]["filtered"][t]["P"])
            
            # calculate x and P with Naive Fusion
            for sensor in sensors:
                # get the sensor information for timestep t
                sensor_result = sensor.single_step(gt=car.gt[t])
                knowledge_s = np.linalg.inv(sensor_result["filtered"]["P"]) # inverse of P_s
                if fc_p == "None":
                    # initialize
                    fc_p = knowledge_s
                    fc_x = knowledge_s @ sensor_result["filtered"]["x"]
                else:
                    fc_p += knowledge_s
                    fc_x += knowledge_s @ sensor_result["filtered"]["x"]
            fc_p = np.linalg.inv(fc_p)
            fc_x = fc_p @ fc_x
            
            track.append({"P": fc_p, "x": fc_x})
        return track
    
    def tracklet_fusion(self, T, car, sensors):
        track = []
        matrices = []
        states = []
        
        # initialize covariance
        P_K_K = np.zeros_like(sensors[0].v[1])
        
        # TODO: Initialize X_K_K without ground truth value
        X_K_K = car.gt[0]
        # X_K_K = sensor_results[0]["filtered"][0]["x"]
        for s in sensors:
            P_K_K += s.P
        P_K_K = 1/len(sensors) * P_K_K
        P_K_K = np.linalg.inv(P_K_K)
        matrices.append(P_K_K)
        X_K_K = P_K_K @ X_K_K
        for t in tqdm(range(T)):
            # predicting P is always the Fisher information matrix, not the covariance!
            P_K_K1 = car.F @ np.linalg.inv(P_K_K) @ car.F.T + car.Q
            P_K_K1 = np.linalg.inv(P_K_K1)
            x_K_K1 = car.F @ np.linalg.inv(P_K_K) @ X_K_K
                
            # filtering
            P_K_K = P_K_K1
            X_K_K = P_K_K1 @ x_K_K1
            for sensor in sensors:
                sensor_result = sensor.single_step(gt=car.gt[t])
                P_s_K_K  = np.linalg.inv(sensor_result["filtered"]["P"])
                P_s_K_K1 = np.linalg.inv(sensor_result["predicted"]["P"])
                X_s_K_K  = sensor_result["filtered"]["x"] 
                X_s_K_K1 = sensor_result["predicted"]["x"]
                I_s = P_s_K_K - P_s_K_K1
                i_s = P_s_K_K @ X_s_K_K - P_s_K_K1 @ X_s_K_K1
                
                P_K_K += I_s
                X_K_K += i_s
            
            p_k = np.linalg.inv(P_K_K)
            states.append(p_k @ X_K_K)
            
            matrices.append(p_k)
            track.append({"P": p_k, "x": states[-1]})
        return track
    
    
    def federated_naive_fusion(self, T, car, sensors):
        track = []
        for t in tqdm(range(T)):
            # initialize x and P
            fc_x = "None" #np.zeros_like(sensor_results[0]["filtered"][t]["x"])
            fc_p = "None" #np.zeros_like(sensor_results[0]["filtered"][t]["P"])
            
            # calculate x and P with Naive Fusion
            for sensor in sensors:
                sensor_result = sensor.single_step_federated(sensor_count=len(sensors), gt=car.gt[t])
                knowledge_s = np.linalg.inv(sensor_result["filtered"]["P"]) # inverse of P_s
                if fc_x == "None":
                    fc_p = knowledge_s
                    fc_x = knowledge_s @ sensor_result["filtered"]["x"]
                else:
                    fc_p += knowledge_s
                    fc_x += knowledge_s @ sensor_result["filtered"]["x"]
            fc_p = np.linalg.inv(fc_p)
            fc_x = fc_p @ fc_x
            
            track.append({"P": fc_p, "x": fc_x})
        return track
    
    def distributed_naive_fusion(self, T, car, sensors):
        track = []
        for t in tqdm(range(T)):
            # initialize x and P
            fc_x = "None" #np.zeros_like(sensor_results[0]["filtered"][t]["x"])
            fc_p = "None" #np.zeros_like(sensor_results[0]["filtered"][t]["P"])
            
            # calculate x and P with Naive Fusion
            for sensor in sensors:
                sensor_result = sensor.single_step_distributed(sensor_count=len(sensors), gt=car.gt[t])
                knowledge_s = np.linalg.inv(sensor_result["filtered"]["P"]) # inverse of P_s
                if fc_x == "None":
                    fc_p = knowledge_s
                    fc_x = knowledge_s @ sensor_result["filtered"]["x"]
                else:
                    fc_p += knowledge_s
                    fc_x += knowledge_s @ sensor_result["filtered"]["x"]
            fc_p = np.linalg.inv(fc_p)
            fc_x = fc_p @ fc_x
            
            # globalization step:
            for sensor in sensors:
                sensor.globalization_step(len(sensors), fc_p)
                

            track.append({"P": fc_p, "x": fc_x})
        return track
    
    def covariance_intersection(self, T, car, sensors):
        track = []
        for t in tqdm(range(T)):
            # get sensor results for time step t
            sensor_results = []
            for sensor in sensors:
                sensor_results.append(sensor.single_step(gt=car.gt[t]))
                
            # search for optimal vector lambda
            lambda_options = [i*0.1 for i in range(11)]
            best_value = None
            best_P = None
            best_x = None
            for l in itertools.product(lambda_options, repeat=len(sensors)):
                value = 0
                # the sum of all lamdas has to be equal to one, we can ignore all other cases
                if np.sum(l) == 1:
                    # calculate the covariance matrix and the state vector
                    P_hat = np.zeros_like(sensor_results[0]["filtered"]["P"])
                    x_hat = np.zeros_like(sensor_results[0]["filtered"]["x"])
                    for s_index in range(len(sensors)):
                        P_s_inv = np.linalg.inv(sensor_results[s_index]["filtered"]["P"])
                        x_s = sensor_results[s_index]["filtered"]["x"]
                        P_hat += l[s_index] * P_s_inv
                        x_hat += l[s_index] * P_s_inv @ x_s
                    P_hat = np.linalg.inv(P_hat)
                    x_hat = P_hat @ x_hat
                    
                    # evaluate the lambda vector
                    # as mentioned in the lecture we choose the trace of P_hat instead of the determinant
                    for i in range(P_hat.shape[0]):
                        value += P_hat[i,i]
                    # check if the new P_hat is better than previous versions
                    if best_value == None or value > best_value:
                        best_value = value
                        best_P = P_hat
                        best_x = x_hat                            
                    
                               
            track.append({"P": best_P, "x": best_x})
        return track
                
                
            