
"""
Use CoSpar framework to optimize over a set of 2D synthetic objective functions.

This version: no buffer (n > 1, b = 0); no coactive feedback. As this version 
does not use buffers or coactive feedback, it is actually the Self-Sparring 
algorithm, with the GP preference model of Chu and Ghahramani (2005) used to 
model the latent reward function.
"""

import numpy as np
import os
import scipy.io as io
import itertools

from Preference_GP_learning import advance, feedback
from CoSpar_feedback_functions import get_objective_value, get_preference


# Number of times to sample the GP model on each iteration. The algorithm 
# will obtain preferences between all non-overlapping groups of num_samples 
# samples.
# NOTE: for ICRA simulations, we ran this script with both num_samples = 2
# and num_samples = 3.
num_samples = 3

run_nums = np.arange(100)     # Repeat test once for each number in this list.

# GP model hyperparameters. The underlying reward function has a Gaussian
# process prior with a squared exponential kernel.
signal_variance = 0.0001   # Gaussian process amplitude parameter
lengthscales = [0.15, 0.15]          # Larger values = smoother reward function
preference_noise = 0.01    # How noisy are the user's preferences?
GP_noise_var = 1e-5        # GP model noise--need at least a very small
                           # number to ensure that the covariance matrix
                           #  is invertible.

num_trials = 150     # Total number of posterior samples/trials
num_iterations = int(np.ceil(num_trials / num_samples))

# Folder in which to save the results. Added _take2 to the folder name, so that 
# the functions used in the ICRA paper don't get accidentally overwritten.
save_folder = 'Sim_results/GP_preference_multi_dueling_take2/'

if not os.path.isdir(save_folder):
    os.mkdir(save_folder)  

# Load points in the grid over which objective functions were sampled.
    
# Load one of the objective functions:
filename = 'Sampled_functions_2D/30_by_30/Sampled_objective_' + str(0) + '.mat'

# Points over which objective function was sampled. These are the points over
# which we will draw samples.
points_to_sample = io.loadmat(filename)['points_to_sample']

# Get 1) dimension of input space and 2) number of points in objective function
# grid (also the number of points we will jointly sample in each posterior sample)
num_pts_sample, state_dim = points_to_sample.shape               

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
       
GP_prior_cov_inv = np.linalg.inv(GP_prior_cov)  # Inverse of covariance matrix

# List of all pairs of samples between which to obtain pairwise preferences 
# (there are (num_samples choose 2) of these):
preference_queries = list(itertools.combinations(np.arange(num_samples), 2))
num_pref = len(preference_queries)    # Pairwise preferences per iteration

# Each iteration of the outer loop is a new repetition of the experiment.
for i, run_num in enumerate(run_nums):

    # Load objective function to use in this experimental repetition.
        
    # Filename for objective function for this run:
    filename = 'Sampled_functions_2D/30_by_30/Sampled_objective_' + \
                str(run_num) + '.mat'
    
    data = io.loadmat(filename)
    objective_function = data['sample']    
    
    # Initialize data matrix and label vector:
    data_pts = np.empty((num_pref * num_iterations * 2, state_dim))
    data_pt_idxs = np.zeros((num_pref * num_iterations, 2)).astype(int)
    labels = np.empty((num_pref * num_iterations, 2))

    # Also store objective function values (for diagnostic purposes only--the 
    # learning algorithm cannot see this):
    objective_values = np.empty(num_samples * num_iterations)

    pref_count = 0  # Keeps track of how many preferences are in the dataset

    # In each iteration, we sample num_samples points from the model posterior, 
    # and obtain all possible pairwise preferences between these points.   
    for it in range(num_iterations):
       
        # Print status:
        print('Run %i of %i, iteration %i of %i' % (i + 1, len(run_nums), 
            it + 1, num_iterations))
        
        # Preference data observed so far (used to train GP preference model):
        X = data_pt_idxs[: pref_count, :]
        y = labels[: pref_count, 1]
        
        # Update the Gaussian process preference model:
        posterior_model = feedback(X, y, GP_prior_cov_inv, preference_noise)    
        
        # Sample new points at which to query for a preference:
        sampled_point_idxs, _ = advance(posterior_model, num_samples)

        # Obtain coordinate points corresponding to these indices, and 
        # store the objective function values:
        sampled_points = np.empty((num_samples, state_dim))
        
        for j in range(num_samples):
           
            sampled_point_idx = sampled_point_idxs[j]
            # Coordinate point representation:
            sampled_points[j, :] = points_to_sample[sampled_point_idx, :]
            
            sample_idx = it * num_samples + j
            # Objective function value:
            objective_values[sample_idx] = \
                get_objective_value(sampled_point_idx, objective_function) 
        
        # Obtain a preference between each pair of samples, and store all
        # of the new information.
        for j in range(num_pref):
           
            # Sampled points to compare:
            idx1 = sampled_point_idxs[preference_queries[j][0]]
            idx2 = sampled_point_idxs[preference_queries[j][1]]
            
            # Convert samples to coordinate point representation:
            sampled_pts = np.vstack((points_to_sample[idx1, :], 
                                    points_to_sample[idx2, :]))           

            # Query to obtain new preference:
            preference = get_preference(idx1, idx2, objective_function)                

            # Update the data:            
            data_pts[2*pref_count: 2 * pref_count + 2, :] = sampled_pts
            data_pt_idxs[pref_count, :] = [idx1, idx2]
            labels[pref_count, :] = [1 - preference, preference]      

            pref_count += 1

    # Save the results for this experimental run:
    io.savemat(save_folder + 'Opt_2D_900_' + str(num_samples) + '_samples_' + \
               'vary_obj_run_' + \
        str(run_num) + '.mat', {'data_pts': data_pts, 
           'data_pt_idxs': data_pt_idxs, 'labels': labels, 
            'objective_values': objective_values, 
            'signal_variance': signal_variance, 'lengthscale': lengthscale, 
            'GP_noise_var': GP_noise_var, 'preference_noise': preference_noise,
            'filename': filename})
    
    