
"""
Use multi-dueling bandits framework to optimize the compass-gait biped's cost
of transport over step length (at a fixed velocity).

This version does not use buffers or coactive feedback. It is therefore the 
Self-Sparring algorithm, using the GP preference model of Chu and Ghahramani 
(2005) to model the latent reward function.
"""

import numpy as np
import os
import scipy.io as io
import itertools

from Preference_GP_learning import advance, feedback


def get_gait_objective(x):
    """
    Evaluates the objective function at x. 
    """

    # Polynomial coefficients of CG biped objective
    fit = np.array([-660.1898, 556.5202, -186.4672, 34.1877, -3.4090, 0.1602])
    
    return np.polyval(fit, x)
           
def get_gait_preference(pt1, pt2):
    """
    Obtain a gait preference between two points; prefer whichever point has a 
    lower objective function value; break ties randomly.
    
    Inputs: pt1 and pt2 are the two points to be compared. Both points should 
    be arrays with length equal to the dimensionality of the state space.    
    
    Output: 0 = pt1 preferred; 1 = pt2 preferred.
    """
    
    obj1 = get_gait_objective(pt1)
    obj2 = get_gait_objective(pt2)
    
    if obj2 < obj1:
        return 1
    elif obj1 < obj2:
        return 0
    else:
        return np.random.choice(2)


# Domain over which to optimize:
points_to_sample = np.linspace(0.08, 0.18, 24)

# Number of times to sample the GP model on each iteration. The algorithm 
# will obtain preferences between all pairs of samples.
num_samples = 2

run_nums = np.arange(100)     # Repeat test once for each number in this list.

# GP model hyperparameters. The underlying reward function has a Gaussian
# process prior with a squared exponential kernel.
signal_variance = 0.0001   # Gaussian process amplitude parameter
lengthscales = [0.025]          # Larger values = smoother reward function
preference_noise = 0.01    # How noisy are the user's preferences?
GP_noise_var = 1e-8        # GP model noise--need at least a very small
                            # number to ensure that the covariance matrix
                            #  is invertible.

num_trials = 150     # Total number of posterior samples/trials
num_iterations = int(np.ceil(num_trials / num_samples))

# Folder in which to save the results. Added _take2 to the folder name, so that 
# the functions used in the ICRA paper don't get accidentally overwritten.
save_folder = 'Compass_biped_take2/'

if not os.path.isdir(save_folder):
    os.mkdir(save_folder)                           
                            
# Define points over which to draw samples:
num_pts_sample = points_to_sample.shape[0]   # Number of points in a sample

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

# List of all pairs of samples at which to obtain preferences:
preference_queries = list(itertools.combinations(np.arange(num_samples), 2))
num_pref = len(preference_queries)    # Preferences per iteration

# Each iteration of the outer loop is a new repetition of the experiment.
for i, run_num in enumerate(run_nums):

    # Initialize data matrix and label vector:
    data_pts = np.empty((num_pref * num_iterations * 2, state_dim))
    data_pt_idxs = np.zeros((num_pref * num_iterations, 2)).astype(int)
    labels = np.empty((num_pref * num_iterations, 2))

    # Also store objective function values (for diagnostic purposes only--the 
    # learning algorithm cannot see this):
    objective_values = np.empty(num_samples * num_iterations)

    pref_count = 0  # Keeps track of how many preferences are in the dataset
    
    for it in range(num_iterations):  # Iterations of the experiment
       
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
            sampled_point = points_to_sample[sampled_point_idx, :]
            sampled_points[j, :] = sampled_point
            
            objective_values[it * num_samples + j] = \
                get_gait_objective(sampled_point) 
        
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
            preference = get_gait_preference(sampled_pts[0, :], 
                                             sampled_pts[1, :])                

            # Update the data:            
            data_pts[2*pref_count: 2 * pref_count + 2, :] = sampled_pts
            data_pt_idxs[pref_count, :] = [idx1, idx2]
            labels[pref_count, :] = [1 - preference, preference]

            pref_count += 1

    # Save the results for this experimental run:
    io.savemat(save_folder + 'Opt_' + str(num_samples) + '_samples_' + \
               str(num_pts_sample) + '_pts_run_' + \
        str(run_num) + '.mat', {'data_pts': data_pts, 
           'data_pt_idxs': data_pt_idxs, 'labels': labels, 
            'objective_values': objective_values, 
            'signal_variance': signal_variance, 'lengthscale': lengthscale, 
            'GP_noise_var': GP_noise_var, 'preference_noise': preference_noise})
    