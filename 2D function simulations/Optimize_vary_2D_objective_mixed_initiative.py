
"""
Use CoSpar framework to optimize over a set of 2D synthetic objective functions.

This version: no buffer (n > 1, b = 0); with coactive feedback.
"""

import numpy as np
import os
import scipy.io as io
import itertools

from Preference_GP_learning import advance, feedback
from CoSpar_feedback_functions import (get_objective_value, get_preference,
get_coactive_feedback)

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

num_trials = 150     # Total number of objective function evaluations/trials
num_iterations = int(np.ceil(num_trials / num_samples))

# Folder in which to save the results. Added _take2 to the folder name, so that 
# the functions used in the ICRA paper don't get accidentally overwritten.
save_folder = 'Sim_results/Multi_dueling_mixed_initiative_take2/'

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

# Coordinate points in each dimension:
points_in_each_dim = []

for i in range(state_dim):
    points_in_each_dim.append(np.unique(points_to_sample[:, i]))

percentile_thresholds = [50, 75]  # Percentile thresholds for coactive feedback
incr_frac = [0.05, 0.10]          # Increments for coactive feedback

# Define increments matrix for coactive feedback. Increments are defined as
# specified fractions of the total range in each dimension.
increments = np.empty((len(percentile_thresholds), state_dim))

for j in range(state_dim):

    rng = points_to_sample[-1, j] - points_to_sample[0, j]
    num_pts = len(points_in_each_dim[j])
    
    for i in range(len(percentile_thresholds)):
    
        increments[i, j] = np.ceil((rng * incr_frac[i]) * num_pts)               

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
    
    # Load objective function and its gradient:
    data = io.loadmat(filename)
    objective_function = data['sample']
    gradient = data['gradient']    

    # Initialize gradient thresholds:
    gradient_thresholds = np.empty((len(percentile_thresholds), state_dim))
    
    for j in range(state_dim):
        
        magn = np.abs(gradient[j, :, :].flatten())    # Gradient magnitudes
        gradient_thresholds[:, j] = np.percentile(magn, percentile_thresholds)
        
    # Initialize data matrix and label vector:
    data_pts = np.empty((num_pref * num_iterations * 2, state_dim))
    data_pt_idxs = np.zeros((num_pref * num_iterations, 2)).astype(int)
    labels = np.empty((num_pref * num_iterations, 2))

    # Also store objective function values (for diagnostic purposes only--the 
    # learning algorithm cannot see this):
    objective_values = np.empty(num_samples * num_iterations)

    # Initialize data matrix and label vector for storing virtual preferences
    # (i.e., those obtained via coactive feedback):
    virtual_pts = np.empty((0, state_dim))
    virtual_pt_idxs = np.zeros((0, 2)).astype(int)
    virtual_labels = np.empty((0, 2))
    virtual_sample_idxs = np.empty(0)

    pref_count = 0  # Keeps track of how many preferences are in the dataset

    # In each iteration, we sample num_samples points from the model posterior, 
    # and obtain all possible pairwise preferences between these points.      
    for it in range(num_iterations):
       
        # Print status:
        print('Run %i of %i, iteration %i of %i' % (i + 1, len(run_nums), 
            it + 1, num_iterations))
        
        # Preference data observed so far (used to train GP preference model):
        X = np.vstack((data_pt_idxs[: pref_count, :], virtual_pt_idxs))
        y = np.concatenate((labels[: pref_count, 1], virtual_labels[:, 1]))
        
        # Update the Gaussian process preference model:
        posterior_model = feedback(X, y, GP_prior_cov_inv, preference_noise)    
        
        # Sample new points at which to query for a preference:
        sampled_point_idxs, _ = advance(posterior_model, num_samples)

        # Obtain coordinate points corresponding to these indices, and 
        # store the objective function values:
        sampled_points = np.empty((num_samples, state_dim))
        
        for j in range(num_samples):
           
            sampled_point_idx = sampled_point_idxs[j]
            # Coordinate representation:
            sampled_points[j, :] = points_to_sample[sampled_point_idx, :]
            
            sample_idx = it * num_samples + j
            # Objective function value:
            objective_values[sample_idx] = \
                get_objective_value(sampled_point_idx, objective_function) 

            # Query for coactive feedback:            
            coactive_feedback = get_coactive_feedback(sampled_point_idx, 
                              gradient, points_to_sample, points_in_each_dim, 
                              gradient_thresholds, increments).astype(int)

            # If there's coactive feedback, then store it:
            if np.size(coactive_feedback) > 0:
                
                virtual_pts = np.vstack((virtual_pts, 
                                         points_to_sample[coactive_feedback, :]))
                virtual_pt_idxs = np.vstack((virtual_pt_idxs, 
                                         [sampled_point_idx, coactive_feedback]))
                virtual_labels = np.vstack((virtual_labels, [0, 1]))
                virtual_sample_idxs = np.concatenate((virtual_sample_idxs, [sample_idx]))
        
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
               'vary_obj_run_' + str(run_num) + '.mat', {'data_pts': data_pts, 
           'data_pt_idxs': data_pt_idxs, 'labels': labels, 
            'objective_values': objective_values, 
            'signal_variance': signal_variance, 'lengthscale': lengthscale, 
            'GP_noise_var': GP_noise_var, 'preference_noise': preference_noise,
            'filename': filename, 'virtual_pts': virtual_pts, 
            'virtual_pt_idxs': virtual_pt_idxs, 'virtual_labels': virtual_labels,
            'virtual_sample_idxs': virtual_sample_idxs})
    
    