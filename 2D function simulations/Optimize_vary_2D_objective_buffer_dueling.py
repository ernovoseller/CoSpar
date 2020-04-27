
"""
Use CoSpar framework to optimize over a set of 2D synthetic objective functions.

This version: preference buffer (n = 1, b > 0); no coactive feedback.
"""

import numpy as np
import os
import scipy.io as io

from Preference_GP_learning import advance, feedback
from CoSpar_feedback_functions import get_objective_value, get_preference


# Number of previous samples to compare with each new data point. A value
# of 1 indicates that all consecutively-taken actions are compared to each
# other. A value of 2 indicates that each new sample is compared against
# the previous 2, etc.
# NOTE: for ICRA simulations, we ran this script with buffer_size = 1.
buffer_size = 1

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

# Folder in which to save the results. Added _take2 to the folder name, so that 
# the functions used in the ICRA paper don't get accidentally overwritten.
save_folder = 'Buffer_dueling_take2/'

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

num_samples = 1   # Number of samples to request at a time from advance function

# Calculate the total number of preferences that will be queried:
num_pref = int(buffer_size * (buffer_size - 1) / 2 + buffer_size * \
    (num_trials - buffer_size))

# Each iteration of the outer loop is a new repetition of the experiment.
for i, run_num in enumerate(run_nums):

    # Load objective function to use in this experimental repetition.
        
    # Filename for objective function for this run:
    filename = 'Sampled_functions_2D/30_by_30/Sampled_objective_' + \
                str(run_num) + '.mat'
    
    data = io.loadmat(filename)
    objective_function = data['sample']    
    
    # Initialize data matrix and label vector:
    data_pts = np.empty((num_pref * 2, state_dim))
    data_pt_idxs = np.zeros((num_pref, 2)).astype(int)
    labels = np.empty((num_pref, 2))

    # Also store objective function values (for diagnostic purposes only--the 
    # learning algorithm cannot see this):
    objective_values = np.empty(num_trials)

    pref_count = 0  # Keeps track of how many preferences are in the dataset
    
    # Initialize buffer for generating preference queries:
    buffer = -np.ones(buffer_size).astype(int)
    buffer_ptr = 0    # Points to next index in the buffer to overwrite

    # In each trial, we sample a new point from the model posterior, and
    # compare it to everything in the preference buffer.    
    for trial in range(num_trials):
       
        # Print status:
        print('Run %i of %i, trial %i of %i' % (i + 1, len(run_nums), 
            trial + 1, num_trials))
        
        # Preference data observed so far (used to train GP preference model):
        X = data_pt_idxs[: pref_count, :]
        y = labels[: pref_count, 1]
        
        # Update the Gaussian process preference model:
        posterior_model = feedback(X, y, GP_prior_cov_inv, preference_noise)    
        
        # Sample new points at which to query for a preference:
        sampled_points, _ = advance(posterior_model, num_samples)
        sampled_point_idx = sampled_points[0]
        # Coordinate point representation:
        sampled_point = points_to_sample[sampled_point_idx, :]
       
        # Get corresponding objective function value:
        objective_values[trial] = get_objective_value(sampled_point_idx, 
                        objective_function)
        
        # Obtain a preference between the newly-sampled point and each
        # sample in the buffer:
        for j in range(buffer_size):
           
            buffer_point_idx = buffer[j]    # Process next point in buffer
            
            # In case buffer is not yet full:
            if buffer_point_idx < 0:
                continue
            
            # Convert buffer point to coordinates:
            buffer_point = points_to_sample[buffer_point_idx, :]          

            # Query to obtain new preference:
            preference = get_preference(buffer_point_idx, sampled_point_idx,
                                             objective_function)               

            # Update the data:            
            data_pts[2*pref_count: 2 * pref_count + 2, :] = \
                np.vstack((buffer_point, sampled_point))
            data_pt_idxs[pref_count, :] = [buffer_point_idx, sampled_point_idx]
            labels[pref_count, :] = [1 - preference, preference]
            
            pref_count += 1
        
        # Update the buffer:
        buffer[buffer_ptr] = sampled_point_idx
        buffer_ptr = np.mod(buffer_ptr + 1, buffer_size)

    # Save the results for this experimental run:
    io.savemat(save_folder + 'Opt_2D_900_buffer_' + str(buffer_size) + \
               '_vary_obj_run_' + str(run_num) + '.mat', 
               {'data_pts': data_pts, 'data_pt_idxs': data_pt_idxs, 
            'labels': labels, 'objective_values': objective_values, 
            'signal_variance': signal_variance, 'lengthscale': lengthscale, 
            'GP_noise_var': GP_noise_var, 'preference_noise': preference_noise,
            'filename': filename})
    
    