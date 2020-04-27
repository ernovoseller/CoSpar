# -*- coding: utf-8 -*-
"""
Plot results from compass-gait biped simulations.
"""

import numpy as np
import scipy.io as io

import matplotlib.pyplot as plt
from matplotlib import rcParams
rcParams.update({'font.size': 24})

fig_num = 0

plt.close('all')

#%% Plot the objective function.

# Domain to use for plotting (also the domain used for learning):
num_pts_sample = 24
x = np.linspace(0.08, 0.18, num_pts_sample)

# Polynomial coefficients of CG biped objective
fit = np.array([-660.1898, 556.5202, -186.4672, 34.1877, -3.4090, 0.1602])

plt.figure(0, figsize = (8, 6))

plt.plot(x, np.polyval(fit, x), linewidth = 3)

plt.xlabel('Step length (m)')
plt.ylabel('Cost of transport (J*s/m)')

plt.xticks([0.08, 0.13, 0.18])
plt.yticks([0.020, 0.025, 0.030])

plt.tight_layout()

#%% Plot posteriors and posterior samples for different amounts of data.
import os

from Preference_GP_learning import feedback    

num_samples_plot = 3    # Number of GP samples to draw on each plot
num_samples = 2         # We used pairwise preferences with Self-Sparring

# Folder in which to save plots:
save_plots_folder = 'Plots/'

if not os.path.isdir(save_plots_folder):
    os.mkdir(save_plots_folder)

# Numbers of preferences at which to make a posterior plot:
pref_nums = [0, 5, 20, 75]

# Load data from the first run:
save_folder = 'Compass_biped_results/'
run_num = 0
data = io.loadmat(save_folder + 'Opt_' + str(num_samples) + '_samples_' \
                     + str(num_pts_sample) + '_pts_run_' + str(run_num) + '.mat')

obj_values = data['objective_values'][0]
data_pt_idxs = data['data_pt_idxs']
labels = data['labels']
preference_noise = data['preference_noise'][0][0]
lengthscales = data['lengthscale'][0]
signal_variance = data['signal_variance'][0][0]
GP_noise_var = data['GP_noise_var'][0][0]

# Domain over which to optimize:
points_to_sample = x

# Determine dimensionality of state space:
if len(points_to_sample.shape) == 1:
    state_dim = 1  
else:
    state_dim = points_to_sample.shape[1]                  
  
points_to_sample = points_to_sample.reshape((num_pts_sample, state_dim))

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

sampled_points = points_to_sample.flatten()

for i, pref_num in enumerate(pref_nums):
    
    print('Iter %i of %i' % (i + 1, len(pref_nums)))
    
    # Preference data to use for this plot:
    X = data_pt_idxs[: pref_num, :]
    y = labels[: pref_num, 1]
    
    # Update the Gaussian process preference model:
    posterior_model = feedback(X, y, GP_prior_cov_inv, preference_noise)    
    
    # Load reward samples:
    """
    NOTE: here is the code for actually generating the reward samples that will
    be plotted. We saved several of them, so that they would not need to
    be constantly resampled.
    
    from Preference_GP_learning import advance
    
    # Sample new points at which to query for a preference:
    _, reward_models = advance(posterior_model, num_samples_plot)
    #    io.savemat('Plotting_data/Reward_samples_' + str(pref_num) + '_preferences.mat',
    #      {'reward_samples': reward_models, 'sampled_points': sampled_points})
    """
    
    reward_models = io.loadmat('Plotting_data/Reward_samples_' + str(pref_num) + \
                               '_preferences.mat')['reward_samples']

    # Unpack model posterior:
    post_mean = posterior_model['mean']
    cov_evecs = np.real(posterior_model['cov_evecs'])
    cov_evals = posterior_model['cov_evals']
    
    # Construct posterior covariance matrix:
    post_cov = cov_evecs @ np.diag(cov_evals) @ np.linalg.inv(cov_evecs)
    
    # Posterior standard deviation at each point:
    post_stdev = np.sqrt(np.diag(post_cov))
    
    plt.figure(figsize = (8, 6))
    
    # Plot posterior mean and standard deviation:
    plt.plot(sampled_points, post_mean, color = 'blue', linewidth = 3)
    plt.fill_between(sampled_points, post_mean - 2*post_stdev, 
                     post_mean + 2*post_stdev, alpha = 0.3, color = 'blue')
    
    # Plot posterior samples:
    for j in range(num_samples_plot):
        
        reward_model = reward_models[j, :]
        
        plt.plot(sampled_points, reward_model, color = 'green',
                 linestyle = '--', linewidth = 3)
        
    plt.ylim([-0.035, 0.043])
    
    plt.xlabel('Step length (m)')
    plt.ylabel('Posterior Utility')
    plt.xticks([0.08, 0.13, 0.18])
    plt.yticks([-0.02, 0, 0.02, 0.04])
    
    if i == 0:
        plt.legend(['Posterior', 'Posterior samples'], loc = 'upper left',
                   fontsize = 23)
    
    plt.tight_layout()
    
#    plt.savefig(save_plots_folder + 'Compass_biped_2STD_' + str(pref_num) + \
#                '_preferences.png')
    
    