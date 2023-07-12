# -*- coding: utf-8 -*-
"""
Created on Fri May 26 17:09:35 2023
 
@author: {name: frederic klein, matriculation_number: 2735589}
"""
# This File contains the functions used to produce visualizations
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation

def animate_xy(x, y, x_true, y_true, T):
    #Multi-Kalman ANIMATE
    fig, ax = plt.subplots()
    xdata, ydata = [],[]
    xdata_gt, ydata_gt = [], []
    ln, =plt.plot([],[], 'rx', label="Filter")
    ln_gt, =plt.plot([],[], 'bx', label="Ground Truth")
    
    def init():
    	ax.set_xlim(-6000, 6000)
    	ax.set_ylim(-6000, 6000)
    	return ln, ln_gt,
    
    def update(frame):
        xdata.append(x[frame])
        ydata.append(y[frame])
        xdata_gt.append(x_true[frame])
        ydata_gt.append(y_true[frame])
        ln_gt.set_data(xdata_gt, ydata_gt)
        ln.set_data(xdata, ydata)
        
        return ln, ln_gt,
    
    ani = FuncAnimation(fig, update, frames=range(0, T), init_func=init, interval = 250, save_count = 10, blit=True, repeat=False)
    #plt.scatter(x_true, y_true, label= 'Ground Truth')
    plt.legend()
    plt.title('Standard Kalman Filter with Covariance Intersection')
    plt.show()
    # saving the animation as gif
    file_path = "C://Users/frede/Programming/AdvancedSDF/Results/StandardKalmanFilter_CovarianceIntersection.gif" 
    ani.save(file_path, writer="mencoder")
    #Animation ENDE
