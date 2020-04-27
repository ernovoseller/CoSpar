# -*- coding: utf-8 -*-
"""
For the ICRA video, we made some animations of how the preference model 
posteriors evolve after each iteration. This script saves the stack of images
to make such an animation for the compass-gait biped's model posterior. For 
every iteration, we save an image of the model posterior from one of the CG 
biped simulation runs.
"""

import numpy as np
import scipy.io as io
import os

import matplotlib.pyplot as plt
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

# Folder for saving plots:
save_plots_folder = 'Plots/CG_biped_animation_plots/'

if not os.path.isdir(save_plots_folder):
    os.makedirs(save_plots_folder)

# Folder for saving (or loading) posterior information:
save_info_folder = 'CG_biped_sim_posteriors/'

if not os.path.isdir(save_info_folder):
    os.makedirs(save_info_folder)

# Load data to use for plotting evolution of the posterior:   
CG_sim_folder = 'Compass_biped/'

run_num = 0
num_samples = 2      # CoSpar parameter (n)
num_pts_sample = 24  # Number of points in input domain

data = io.loadmat(CG_sim_folder + 'Opt_' + str(num_samples) + '_samples_' \
                     + str(num_pts_sample) + '_pts_run_' + str(run_num) + \
                     '.mat')

data_pt_idxs = data['data_pt_idxs']  # Data: points alg. selected in simulation
pref_nums = data_pt_idxs.shape[0]

# Domain over which learning occurred:
points_to_sample = np.linspace(0.08, 0.18, num_pts_sample)

# Determine dimensionality of state space:
if len(points_to_sample.shape) == 1:
    state_dim = 1  
else:
    state_dim = points_to_sample.shape[1]                  
 
if not posterior_already_computed:    
    
    points_to_sample = points_to_sample.reshape((num_pts_sample, state_dim))

    # Load preference labels and GP model hyperparameters:
    labels = data['labels']
    preference_noise = data['preference_noise'][0][0]
    lengthscales = data['lengthscale'][0]
    signal_variance = data['signal_variance'][0][0]
    GP_noise_var = data['GP_noise_var'][0][0]

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

    points_to_sample = points_to_sample.flatten()

# Make a plot for each iteration of the algorithm.
for pref_num in range(pref_nums + 1):
    
    print('Iter %i of %i' % (pref_num, pref_nums))
    
    # Get model posterior to use for this plot:
    if not posterior_already_computed: 
        
        # Preference data at this iteration:
        X = data_pt_idxs[: pref_num, :]
        y = labels[: pref_num, 1]
        
        # Update the Gaussian process preference model:
        posterior_model = feedback(X, y, GP_prior_cov_inv, preference_noise)    
        
        # Unpack model posterior:
        post_mean = posterior_model['mean']
        cov_evecs = np.real(posterior_model['cov_evecs'])
        cov_evals = posterior_model['cov_evals']
    
    else:
    
        posterior_model = io.loadmat(save_info_folder + 'Compass_biped_' + \
                                     str(pref_num) + '_preferences.mat')
    
        # Unpack model posterior:
        post_mean = posterior_model['post_mean'].flatten()
        cov_evecs = np.real(posterior_model['cov_evecs'])
        cov_evals = posterior_model['cov_evals'].flatten()
    
    # Construct posterior covariance matrix:
    post_cov = cov_evecs @ np.diag(cov_evals) @ np.linalg.inv(cov_evecs)
    
    # Posterior standard deviation at each point:
    post_stdev = np.sqrt(np.diag(post_cov))
    
    # Without title, used (8, 6). (8, 6.3) keeps the actual plot the same size
    # while adding a title.
    plt.figure(figsize = (8, 6.3))
    
    # Plot posterior mean and standard deviation:
    plt.plot(points_to_sample, post_mean, color = 'blue', linewidth = 3)
    plt.fill_between(points_to_sample, post_mean - 2*post_stdev, 
                     post_mean + 2*post_stdev, alpha = 0.3, color = 'blue')

    plt.ylim([-0.035, 0.043])
    
    plt.xlabel('Step length (m)')
    plt.ylabel('Posterior Utility')
    plt.title('Number of Trials: ' + str(pref_num * 2))
    
    plt.xticks([0.08, 0.13, 0.18])
    plt.yticks([-0.02, 0, 0.02, 0.04])
    plt.tight_layout()
    
    if not posterior_already_computed: 
        # Save information about posterior:
        io.savemat(save_info_folder + 'Compass_biped_' + str(pref_num) + \
                    '_preferences.mat', {'post_mean': post_mean, 
                                         'cov_evecs': cov_evecs,
                                         'cov_evals': cov_evals})
    
    # Save plot:
    plt.savefig(save_plots_folder + 'Compass_biped_2STD_' + str(pref_num) + \
                '_preferences_titled.png')
    
    plt.close('all')
    
    