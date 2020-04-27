# -*- coding: utf-8 -*-
"""
For the ICRA video, we made some animations of how the preference model 
posteriors evolve after each iteration. This script saves the stack of images
to make such an animation for a 2D objective function's model posterior, for
the case with n = 1, b = 1, and coactive feedback. For every iteration, we
save an image of the model posterior from the simulation.
"""

import numpy as np
import scipy.io as io
import os

import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from matplotlib import cm

from matplotlib import rcParams
rcParams.update({'font.size': 18})

from Preference_GP_learning import feedback

 
# SET THE FOLLOWING FLAG TO EITHER TRUE OR FALSE, DEPENDING ON WHETHER THE
# MODEL POSTERIOR INFORMATION FOR ALL RUNS HAS ALREADY BEEN SAVED. If the
# posterior information is already computed and saved, setting this to True
# will save runtime. If you try setting this to True but the information is not
# saved, then you will get an error. If you set this to False, then all of the 
# necessary information will be saved, such that you can set this to True if
# running this script ever again.
posterior_already_computed = True

# Index of objective function (from the set of 100 randomly-generated synthetic
# objective functions) to use for these plots:
obj_number = 1

# Folder for saving plots:
save_plots_folder = 'Plots/2D_obj_' + str(obj_number) + \
                    '_animation_plots/n_1_b_1_coactive/'

if not os.path.isdir(save_plots_folder):
    os.makedirs(save_plots_folder)

# Folder for saving (or loading) posterior information:
save_info_folder = '2D_obj_' + str(obj_number) + '_sim_posteriors/n_1_b_1_coactive/'

if not os.path.isdir(save_info_folder):
    os.makedirs(save_info_folder)

# Load data to use for plotting evolution of the posterior:
sim_folder = 'Buffer_dueling_mixed_initiative/'
buffer_size = 1      # CoSpar parameter (b)
filename = sim_folder + 'Opt_2D_900_buffer_' + str(buffer_size) + \
                 '_vary_obj_run_' + str(obj_number) + '.mat'     
    
data = io.loadmat(filename)

data_pt_idxs = data['data_pt_idxs']  # Data: points alg. selected in simulation
pref_nums = data_pt_idxs.shape[0]

if not posterior_already_computed:

    # Load preference labels and GP model hyperparameters:
    labels = data['labels'][:, 1]
    preference_noise = data['preference_noise'][0][0]
    lengthscales = data['lengthscale'][0][0] * np.ones(2)
    signal_variance = data['signal_variance'][0][0]
    GP_noise_var = data['GP_noise_var'][0][0]
    
    # Load coactive feedback:
    virtual_pt_idxs = data['virtual_pt_idxs']
    virtual_labels = data['virtual_labels']
    virtual_sample_idxs = data['virtual_sample_idxs'][0] + 1

    # Domain over which learning occurred:
    points_to_sample = io.loadmat('Sampled_functions_2D/30_by_30/Sampled_objective_' \
                                + str(obj_number) + '.mat')['points_to_sample']

    # Determine dimensionality of state space:
    if len(points_to_sample.shape) == 1:
        state_dim = 1  
    else:
        state_dim = points_to_sample.shape[1]
    
    # Number of points in input domain:
    num_pts_sample = points_to_sample.shape[0]
    
    # Instantiate the prior covariance matrix, using a squared exponential
    # kernel in each dimension of the input space:
    GP_prior_cov =  signal_variance * np.ones((num_pts_sample, num_pts_sample))   
    
    for i in range(num_pts_sample):
    
        pt1 = points_to_sample[i, :]
        
        for j in range(num_pts_sample):
            
            pt2 = points_to_sample[j, :]
            
            for dim in range(state_dim):
                
                lengthscale = lengthscales[dim]
                
                if lengthscale > 0:
                    GP_prior_cov[i, j] *= np.exp(-0.5 * ((pt2[dim] - pt1[dim]) / \
                                lengthscale)**2)
                    
                elif lengthscale == 0 and pt1[dim] != pt2[dim]:
                    
                    GP_prior_cov[i, j] = 0
     
    GP_prior_cov += GP_noise_var * np.eye(num_pts_sample)
           
    GP_prior_cov_inv = np.linalg.inv(GP_prior_cov)

# For plotting:
num_pts = [30, 30]   # Number of points in each grid dimension

# Points in input domain:
x_vals = np.linspace(0, 1, num_pts[0])
y_vals = np.linspace(0, 1, num_pts[1])
Y, X = np.meshgrid(x_vals, y_vals)

# Make a plot for each iteration of the algorithm.
for pref_num in range(pref_nums + 1):
    
    print('Iter %i of %i' % (pref_num, pref_nums))
    
    # Get model posterior to use for this plot:
    if not posterior_already_computed: 
        
        # Preference data at this iteration:
        X_ = data_pt_idxs[: pref_num, :]
        y_ = labels[: pref_num]
        
        # Add coactive feedback to the preference data:
        num_coactive_points = len(np.where(virtual_sample_idxs < pref_num)[0])
        X_virtual = virtual_pt_idxs[:num_coactive_points, :]
        y_virtual = virtual_labels[:num_coactive_points, 1]
        
        # Update the Gaussian process preference model:
        posterior_model = feedback(np.vstack((X_, X_virtual)), 
                                   np.concatenate((y_, y_virtual)), GP_prior_cov_inv, 
                                   preference_noise)    
    
        # Unpack model posterior:
        post_mean = posterior_model['mean'].reshape(tuple(num_pts))

    else:
        
        posterior_model = io.loadmat(save_info_folder + '2D_obj_1_' + \
                                     str(pref_num) + '_preferences.mat')
    
        # Unpack model posterior:
        post_mean = posterior_model['post_mean']
    
    # Plot posterior mean:
    fig = plt.figure(figsize = (8, 6))
    ax = fig.gca(projection='3d')
    
    surf = ax.plot_surface(Y, X, post_mean, cmap=cm.coolwarm, linewidth=0, 
                           antialiased=False)
    
    plt.xlabel('x', labelpad = 10)
    plt.ylabel('y', labelpad = 10)
    ax.set_zlabel('\nPosterior Utility', labelpad = 19)
    
    plt.xticks([0, 0.5, 1])
    plt.yticks([0, 0.5, 1])
    ax.set_zticks([0, 0.02])
    ax.tick_params(axis='z', which='major', pad=13)
    ax.set_zlim3d(-0.01, 0.03)   
    
    if pref_num == 0:
        trial_num = pref_num
    else:
        trial_num = pref_num + 1
    
    ax.set_title('Number of Trials: ' + str(trial_num), y = 1.08)
    
    if not posterior_already_computed:
        # Save information about posterior:
        io.savemat(save_info_folder + '2D_obj_' + str(obj_number) + '_' + \
                    str(pref_num) + '_preferences.mat', {'post_mean': post_mean})
    
    # Save plot:
    plt.savefig(save_plots_folder + '2D_obj_' + str(obj_number) + '_' + \
                str(pref_num) + '_preferences_titled.png')
    
    plt.close('all')
    
    