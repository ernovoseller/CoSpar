# -*- coding: utf-8 -*-
"""
Plot the first few steps of Self-Sparring in the Compass-Gait biped example.
This code makes some illustrations of Self-Sparring (i.e., sampling functions
from the posterior and selecting actions that maximize this posterior). These
plots were used in the video accompanying the ICRA submission, but not in the 
paper itself.

We will draw samples from the preference model posteriors that have the same
maxima (i.e. the same selected action) as in the simulations; this will give 
examples of posterior samples that are similar to those that would have been
sampled in the simulation.
"""

import numpy as np
import scipy.io as io
import os
import matplotlib.pyplot as plt
from matplotlib import rcParams
rcParams.update({'font.size': 18})

from Preference_GP_learning import advance 


plt.close('all')

# Folder in which to save plots:
save_plots_folder = 'Plots/Illustrate_self_sparring/'

if not os.path.isdir(save_plots_folder):
    os.mkdir(save_plots_folder)

# Input domain over which learning occurs:
num_pts_sample = 24    # Number of points in each posterior sample
points_to_sample = np.linspace(0.08, 0.18, num_pts_sample)
state_dim = 1       # Dimensionality of input space

# Folder into which information about each model posterior is saved. (The
# animation script saves these posteriors, so that they don't have to be
# recalculated multiple times.)
save_info_folder = 'Plotting_data/CG_biped_sim_posteriors/'

# Load data of which actions were selected in a particular CG biped simulation 
# run:
data_pt_idxs = io.loadmat('Compass_biped_results/Opt_2_samples_' + str(num_pts_sample) \
                          + '_pts_run_0.mat')['data_pt_idxs']

# We will plot samples from the posterior for models trained using each of the
# following numbers of preferences:
pref_nums = [0, 1]

# Indices of posterior samples to plot corresponding to each value in pref_nums.
# These were determined by drawing a number of samples from the posterior,
# plotting the ones in which the selected action matches the actual experimental
# run, and then manually selecting the most photogenic among these.

# The idea is to sample functions from the posterior for which the maximum
# (i.e. selected action) matches the experiment--this gives examples of
# posterior samples that have the same maximum points as the functions that
# would have been sampled in the simulation.
sample_idxs_plot = [[2, 25], [14, 43]]

action_num_labels = ['1st', '2nd']

for i, pref_num in enumerate(pref_nums):

    # Gaussian process prior model:
    GP_model = io.loadmat(save_info_folder + 'Compass_biped_' + \
                                 str(pref_num) + '_preferences.mat')
    
    # Unpack model:
    GP_mean = GP_model['post_mean'].flatten()
    cov_evecs = np.real(GP_model['cov_evecs'])
    cov_evals = GP_model['cov_evals'].flatten()
    GP_model['mean'] = GP_mean   # Rename for compatibility with advance function
    GP_model['cov_evals'] = cov_evals
    
    # Construct posterior covariance matrix:
    prior_cov = cov_evecs @ np.diag(cov_evals) @ np.linalg.inv(cov_evecs)
    
    # Posterior standard deviation at each point:
    GP_stdev = np.sqrt(np.diag(prior_cov))
    
    # Check whether we have already drawn some posterior samples for this 
    # number of preferences.
    post_sample_filename = 'Plotting_data/Samples_from_reward_model_' + str(pref_num) + \
                           '_pref.mat'
                           
    if not os.path.isfile(post_sample_filename):
        
        # Draw a number of samples from the posterior model:
        trial_samples, reward_samples = advance(GP_model, 50)
        
        # Save samples and reward models:
        io.savemat(post_sample_filename, {'samples': trial_samples, 
                                          'reward_models': reward_samples})
    
    # Load and unpack saved reward information:
    data = io.loadmat(post_sample_filename)
        
    trial_samples = data['samples'].flatten()
    reward_samples = data['reward_models']
    
    # Actions sampled in this simulation iteration:
    samples = data_pt_idxs[pref_num, :]
    
    # Find the indices among the 50 reward samples in which the actions sampled
    # match the actual simulation.
    for j, sample in enumerate(samples):
        
        # Find all indices among the posterior samples where the action selected
        # in the simulation was sampled:
        sample_idxs = np.where(trial_samples == sample)[0]
        
        if len(sample_idxs) == 0:
            print('WARNING: none of the posterior samples selects the same action as that chosen in the simulation.')
            
        # Make a plot of each of these samples, overlayed upon the model 
        # posterior. From these, I manually decided which one was the more
        # photogenic.
        plt.figure(figsize = (9, 7))
        
        # Plot posterior mean and standard deviation:
        plt.plot(points_to_sample, GP_mean, color = 'blue', linewidth = 3)
        plt.fill_between(points_to_sample, GP_mean - 2*GP_stdev, 
                         GP_mean + 2*GP_stdev, alpha = 0.3, color = 'blue')        
        
        # Plot the posterior samples:
        for idx in sample_idxs:
            
            reward_sample = reward_samples[idx, :]
            
            plt.plot(points_to_sample, reward_sample, color = 'green',
                     linestyle = '--', linewidth = 3)    
 
        plt.ylim([-0.035, 0.043])
        
        plt.xlabel('Step length (m)')
        plt.ylabel('Posterior Utility')
        
        plt.xticks([0.08, 0.13, 0.18])
        plt.yticks([-0.02, 0, 0.02, 0.04])
        
        plt.tight_layout()   
        plt.title(str(pref_num) + ' Preferences, Samples of ' + \
                  action_num_labels[j] + ' Action Selected')
    
    # Select one of each type of sampled function, and add these to another plot:
    plt.figure(figsize = (8, 6))
    
    # Plot posterior mean and standard deviation:
    plt.plot(points_to_sample, GP_mean, color = 'blue', linewidth = 3)
    plt.fill_between(points_to_sample, GP_mean - 2*GP_stdev, 
                     GP_mean + 2*GP_stdev, alpha = 0.3, color = 'blue')        
    
    # Plot the samples from the posterior:
    for idx in sample_idxs_plot[i]:
        
        reward_sample = reward_samples[idx, :]
        
        plt.plot(points_to_sample, reward_sample, color = 'green',
                 linestyle = '--', linewidth = 3)
    
    plt.ylim([-0.035, 0.043])
    
    plt.xlabel('Step length (m)')
    plt.ylabel('Posterior Utility')
    
    plt.xticks([0.08, 0.13, 0.18])
    plt.yticks([-0.02, 0, 0.02, 0.04])
    
    plt.tight_layout()
    
    # Save plot:
    plt.savefig(save_plots_folder + 'Pref_' + str(pref_num) + '_plot_1.png')
    
    # Mark maxima (i.e. selected actions) on the plot:
    for idx in sample_idxs_plot[i]:
        
        sample_x = points_to_sample[trial_samples[idx]]
        sample_y = reward_samples[idx, trial_samples[idx]]
        
        plt.plot(sample_x, sample_y, color = 'green', marker = 'o', 
                 markersize = '25')
      
    # Save plot:
    plt.savefig(save_plots_folder + 'Pref_' + str(pref_num) + '_plot_2.png') 
    