# -*- coding: utf-8 -*-
"""
Created on Fri May 26 11:09:06 2023

@author: {name: frederic klein, matriculation_number: 2735589}
"""
# This file is used to combine all other parts together.
from Sensor import Sensor
from TrackedObject import TrackedObject
from FusionCenter import FusionCenter 
from Painter import animate_xy
sensor_count = 3 # number of sensors in the simulation
T = 45 # number of timesteps this simulation runs for

def main():
    # create an object that can be tracked
    car = TrackedObject()
    car.create_true_track(T)
    x_true, y_true = car.get_xy_from_states()
    
    # create sensors and append them to the array
    sensor_array = []
    for _ in range(sensor_count):
        sensor_array.append(Sensor())
    
    # create a Fusion Center
    FC = FusionCenter()
        
    # use sensor data to form a track with FC
    # IMPORTANT: only use one method! There is no reset on the sensors. 
    #track = FC.tracklet_fusion(T, car, sensor_array)
    #track = FC.naive_fusion(T, car, sensor_array)
    #track = FC.federated_naive_fusion(T, car, sensor_array)
    #track = FC.distributed_naive_fusion(T, car, sensor_array)
    track = FC.covariance_intersection(T, car, sensor_array)
    # extract x and y coordinates from the fusion center track:
    x,y = FC.get_xy_from_track(track)
    animate_xy(x, y, x_true, y_true, T)
    
if __name__ == "__main__":
    main()