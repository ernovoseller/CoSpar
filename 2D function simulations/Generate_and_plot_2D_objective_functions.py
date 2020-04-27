# -*- coding: utf-8 -*-
"""
Generate set of 2D objective functions by sampling from a GP prior.
These are the objective functions used in the 2D simulations.
"""

import numpy as np
import matplotlib.pyplot as plt
import os
import scipy.io as io

from mpl_toolkits.mplot3d import Axes3D
from matplotlib import cm

plt.close('all')

# Parameters of kernel and noise. This is assuming a squared exponential kernel
# in two dimensions.
variance = 0.1        # Amplitude
lengthscales = [0.15, 0.15]     # Wavyness of signal
noise_var = 1e-10    # Noise

state_dim = len(lengthscales)
num_pts = [30, 30]   # 30-by-30 grid

x_vals = np.linspace(0, 1, num_pts[0])
y_vals = np.linspace(0, 1, num_pts[1])

# Define grid of points over which to evaluate the objective function:

num_sample_points = np.prod(num_pts)  # Total number of points in grid
    
# Put the points into list format:
points_to_sample = np.empty((num_pts[0] * num_pts[1], state_dim))

for i, x_val in enumerate(x_vals):

    for j, y_val in enumerate(y_vals):

        points_to_sample[num_pts[1] * i + j, :] = [x_val, y_val]

# Gaussian process prior mean:
mean = 0.5 * np.ones(num_sample_points)

# Instantiate the prior covariance matrix, using a squared exponential
# kernel in each dimension of the input space:
GP_prior_cov =  variance * np.ones((num_sample_points, num_sample_points))   

for i in range(num_sample_points):

    pt1 = points_to_sample[i, :]
    
    for j in range(num_sample_points):
        
        pt2 = points_to_sample[j, :]
        
        for dim in range(state_dim):
            
            lengthscale = lengthscales[dim]
            
            if lengthscale > 0:
                GP_prior_cov[i, j] *= np.exp(-0.5 * ((pt2[dim] - pt1[dim]) / \
                            lengthscale)**2)
                
            elif lengthscale == 0 and pt1[dim] != pt2[dim]:
                
                GP_prior_cov[i, j] = 0
 
GP_prior_cov += noise_var * np.eye(num_sample_points)
        
# For plotting GP samples:
Y, X = np.meshgrid(x_vals, y_vals)


#%% Sample several 2D functions, and save information about them.
    
# Folder in which to save results. Added _take2 to the folder name, so that the 
# functions used in the ICRA paper don't get accidentally overwritten.
save_folder = 'Sampled_functions_2D/30_by_30_take2/'

if not os.path.isdir(save_folder):
    os.mkdir(save_folder)
    
# Number of times to sample the GP:
num_GP_samples = 100

for i in range(num_GP_samples):
    
    # Draw a sample from the GP:
    GP_sample = np.random.multivariate_normal(mean, GP_prior_cov)
    GP_sample = np.transpose(GP_sample.reshape((num_pts[0], num_pts[1])))

    # Compute the gradient of the sample:
    incr = np.array([x_vals[1] - x_vals[0], y_vals[1] - y_vals[0]])
    gradient = np.gradient(GP_sample, incr[0], incr[1])
    
    # Save information about the sample and its gradient:
    io.savemat(save_folder + 'Sampled_objective_' + str(i) + '.mat', 
               {'sample': GP_sample, 'gradient': gradient, 
                 'points_to_sample': points_to_sample, 'variance': variance, 
                 'lengthscale': lengthscales, 'noise_var': noise_var})
    
#%% This code can be used to plot one or more of the GP samples, and to also
# plot gradient information. This is useful for getting a visual sense of
# how coactive feedback will be provided in the simulations.

for i in [0]:
    
    # Load objective function and its gradient:
    data = io.loadmat(save_folder + 'Sampled_objective_' + str(i) + '.mat')

    sample = data['sample']
    gradient = data['gradient']

    # Plot the objective function:
    fig = plt.figure()
    ax = fig.gca(projection='3d')
    
    surf = ax.plot_surface(Y, X, sample, cmap=cm.coolwarm, linewidth=0, antialiased=False)
    
    plt.xlabel('x')
    plt.ylabel('y')
    ax.set(zlabel='Objective function')
    plt.title('Objective function')

    # Plot each gradient component over the domain (one plot for x-component,
    # one plot for y-component): 
    for dim, grad in enumerate(gradient):
    
        fig = plt.figure()
        ax = fig.gca(projection='3d')
        
        surf = ax.plot_surface(Y, X, grad, cmap=cm.coolwarm, linewidth=0, antialiased=False)
        
        plt.xlabel('x')    
        plt.ylabel('y')
        ax.set(zlabel = 'Gradient')
        
        plt.title('Objective function gradient, dimension ' + str(dim))
        
    # Make both gradient plots again. This time, overlay a representation of
    # the gradient magnitude thresholds at each point. Use three different 
    # colors to differentiate the following: 1) gradients below 50th
    # percentile threshold, 2) gradients between the 2 thresholds, and 3)
    # gradients above the 75th percentile threshold.
    for dim, grad in enumerate(gradient):
    
        magn = np.abs(grad)
        
        # Percentiles to consider:
        percentiles = np.percentile(magn.flatten(), [50, 75])
        
        fig = plt.figure()
        ax = fig.gca(projection='3d')
        surf = ax.plot_surface(Y, X, sample, cmap=cm.coolwarm, linewidth=0, antialiased=False)

        c = np.zeros(X.shape)    # Initialize colors for scatter plot

        for j, perc in enumerate(percentiles):
 
            c += 0.5 * (magn >= perc).astype(int)

        ax.scatter(Y, X, sample, c = c, cmap = cm.viridis, s = 6)
        
        plt.xlabel('x')    
        plt.ylabel('y')
        ax.set(zlabel = 'Objective function')
        
        plt.title('GP Samples, Color-Coded by Gradient Magnitude Percentile')


                