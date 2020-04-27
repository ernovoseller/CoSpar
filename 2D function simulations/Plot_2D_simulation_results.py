# -*- coding: utf-8 -*-
"""
Plot results from simulations optimizing 2D randomly-generated synthetic 
objective functions.
"""

import numpy as np
import scipy.io as io

import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from matplotlib import cm

from matplotlib import rcParams
rcParams.update({'font.size': 18})

plt.close('all')

def plot_avg_objective_vals(filename_part1, filename_part2, num_trials, num_runs, 
                           fig_num, plot_mean_SD = True, line_plot = False,
                           color = 'blue', alpha = 0.5, norm = False, 
                           plot_SD = True, mean_linestyle = '-', 
                           mean_linewidth = 1.5, plot_SE = True):
    """
    Function to calculate means and standard deviations of objective function 
    values over the different runs, and add them to the given figure. Also, 
    includes an option for just plotting each sequence separately.
    
    Options:
        1) filenames of data files are assumed to be in the form 
        filename_part1_x_filename_part2, where x is the number corresponding to a
        particular simulation run.
        2) num_trials: number of trials to plot from each simulation
        3) num_runs: number of repetitions of the experiment
        4) fig_num: index of new figure
        5) plot_mean_SD: whether to plot mean of trials and a measure of the 
        deviation from the mean
        6) line_plot: if this is set to true, then plot trajectory of each 
        individual run
        7) color: color of lines and shaded area
        8) alpha: for setting transparency of shaded area (if any)
        9) norm: if true, then normalize each objective function to lie between 
        0 and 1
        10) plot_SD: if false, do not plot shaded area corresponding to standard 
        deviation or standard error. This is useful for just plotting the mean
        of all the trials.
        11) mean_linestyle and mean_linewidth: arguments for plotting the mean,
        in case you want to change them from the defaults.
        12) plot_SE: if True, then plot standard error instead of standard deviation.
    """
    
    plt.figure(fig_num)
    
    # Obtain the objective values over the runs.
    obj_vals = np.empty((num_trials, num_runs))
    
    for run in range(num_runs):
        
        # Load and unpack results:
        results = io.loadmat(filename_part1 + str(run) + filename_part2)
        obj = results['objective_values'].flatten()[: num_trials]
        
        if norm:   # Normalize objective function values
            obj_function = io.loadmat('Sampled_functions_2D/30_by_30/Sampled_objective_' + \
                str(run) + '.mat')
            obj_function = obj_function['sample'].flatten()
            
            obj = (obj - np.min(obj_function)) /   \
                    (np.max(obj_function) - np.min(obj_function))
        
        obj_vals[:, run] = obj
        
        if line_plot:
            plt.plot(np.arange(1, num_trials + 1), obj_vals[:, run],
                     color = color)
    
    if plot_mean_SD:  # If plotting mean and deviation
        mean = np.mean(obj_vals, axis = 1)
        stdev = np.std(obj_vals, axis = 1)
        
        if plot_SE: # If plotting standard error rather than standard dev.
            stdev /= np.sqrt(num_runs)
        
        # Plot the mean over the trials:    
        plt.plot(np.arange(1, num_trials + 1), mean, color = color,
                 linestyle = mean_linestyle, linewidth = mean_linewidth)
        
        # Add deviation to plot
        if plot_SD:
            plt.fill_between(np.arange(1, num_trials + 1), mean - stdev, 
                             mean + stdev, alpha = alpha, color = color)

#%% Plot an example objective function.

num_pts = [30, 30]

x_vals = np.linspace(0, 1, num_pts[0])
y_vals = np.linspace(0, 1, num_pts[1])
Y, X = np.meshgrid(x_vals, y_vals)

# Folder in which samples were saved:
save_folder = 'Sampled_functions_2D/30_by_30/'

obj_number = 1   # Objective function to plot

data = io.loadmat(save_folder + 'Sampled_objective_' + str(obj_number) + '.mat')

sample = data['sample']

# Normalize the sample:
sample = (sample - np.min(sample)) / (np.max(sample) - np.min(sample))

points_to_sample = data['points_to_sample']

fig = plt.figure(figsize = (7.2, 4.76))
ax = fig.gca(projection='3d')

surf = ax.plot_surface(Y, X, sample, cmap=cm.coolwarm, linewidth=0, antialiased=False)

plt.xlabel('x', labelpad = 10)
plt.ylabel('y', labelpad = 10)
ax.set_zlabel('\nObjective value', labelpad = 19)
plt.colorbar(surf, pad = 0.15, ticks = [0, 0.2, 0.4, 0.6, 0.8, 1])

plt.xticks([0, 0.5, 1])
plt.yticks([0, 0.5, 1])
ax.set_zticks([0, 0.5, 1])
ax.tick_params(axis='z', which='major', pad=13)      


##%% Calculates and save the posterior mean that we will plot in the next cell,
## so that it can be loaded without needing to be recalculated each time.
#   
#from Preference_GP_learning import feedback    
#    
## Load data from experiment:
#
#buffer_size = 1
#save_folder = 'Buffer_dueling_mixed_initiative/'
#filename = save_folder + 'Opt_2D_900_buffer_' + str(buffer_size) + \
#                 '_vary_obj_run_' + str(obj_number) + '.mat' 
#
#data = io.loadmat(filename)
#
## Load preference feedback:
#data_pt_idxs = data['data_pt_idxs']
#labels = data['labels'][:, 1]
#
## Load coactive feedback:
#virtual_pt_idxs = data['virtual_pt_idxs']
#virtual_labels = data['virtual_labels'][:, 1]
#
#preference_noise = data['preference_noise'][0][0]
#lengthscales = data['lengthscale'][0][0] * np.ones(2)
#signal_variance = data['signal_variance'][0][0]
#GP_noise_var = data['GP_noise_var'][0][0]
#
## Determine dimensionality of state space:
#if len(points_to_sample.shape) == 1:
#    state_dim = 1  
#else:
#    state_dim = points_to_sample.shape[1]                  
#  
#num_pts_sample = points_to_sample.shape[0]
#
## Instantiate the prior covariance matrix, using a squared exponential
## kernel in each dimension of the input space:
#GP_prior_cov =  signal_variance * np.ones((num_pts_sample, num_pts_sample))   
#
#for i in range(num_pts_sample):
#
#    pt1 = points_to_sample[i, :]
#    
#    for j in range(num_pts_sample):
#        
#        pt2 = points_to_sample[j, :]
#        
#        for dim in range(state_dim):
#            
#            lengthscale = lengthscales[dim]
#            
#            if lengthscale > 0:
#                GP_prior_cov[i, j] *= np.exp(-0.5 * ((pt2[dim] - pt1[dim]) / \
#                            lengthscale)**2)
#                
#            elif lengthscale == 0 and pt1[dim] != pt2[dim]:
#                
#                GP_prior_cov[i, j] = 0
# 
#GP_prior_cov += GP_noise_var * np.eye(num_pts_sample)
#       
#GP_prior_cov_inv = np.linalg.inv(GP_prior_cov)
#
## Update the Gaussian process preference model:
#posterior_model = feedback(np.vstack((data_pt_idxs, virtual_pt_idxs)), 
#                           np.concatenate((labels, virtual_labels)), GP_prior_cov_inv, 
#                           preference_noise)   
#
## Posterior mean:
#post_mean = posterior_model['mean'].reshape(tuple(num_pts))
#
#io.savemat('Post_mean_for_plot.mat', {'post_mean': post_mean})

#%% Plot the posterior mean by loading a saved file, rather than re-fitting the model:
rcParams.update({'font.size': 18})

post_mean = io.loadmat('Post_mean_for_plot.mat')['post_mean']

# Plot posterior mean:
fig = plt.figure(figsize = (7.2, 4.76))
ax = fig.gca(projection='3d')

surf = ax.plot_surface(Y, X, post_mean, cmap=cm.coolwarm, linewidth=0, 
                       antialiased=False)

plt.xlabel('x', labelpad = 10)
plt.ylabel('y', labelpad = 10)
ax.set_zlabel('\nPosterior Utility', labelpad = 19)
plt.colorbar(surf, pad = 0.15)

plt.xticks([0, 0.5, 1])
plt.yticks([0, 0.5, 1])
ax.set_zticks([0, 0.03])
ax.tick_params(axis='z', which='major', pad=13)


#%% Make a plot with all learning curves on one plot (mean +/- standard error).

# Plot multi-dueling bandits cases.
rcParams.update({'font.size': 12})

# Color-blind friendly palette: https://gist.github.com/thriveth/8560036
CB_colors = ['#377eb8', '#4daf4a', '#ff7f00', 
                  '#f781bf', '#a65628', '#984ea3',
                  '#999999', '#e41a1c', '#dede00']                  

colors = CB_colors[:3]
   
fig_num = 3

num_runs = 100     # Times experiment was repeated
filename_part2 = '.mat'
num_trials = 150     # Total number of posterior samples/trials

# Plot multi-dueling cases:
num_samples_values = [2, 3]

alpha = 0.4

for i, num_samples in enumerate(num_samples_values):
    
    # Folder into which results are saved:
    save_folder = 'GP_preference_multi_dueling/'
    
    filename_part1 = save_folder + 'Opt_2D_900_' + str(num_samples) + '_samples_' \
                     + 'vary_obj_run_'
    
    # Plot mean +/- stdev:
    plot_avg_objective_vals(filename_part1, filename_part2, num_trials, num_runs, 
                               fig_num, plot_mean_SD = True, line_plot = False,
                               color = colors[i], norm = True, alpha = alpha,
                               mean_linestyle = 'dotted', mean_linewidth = 2)

    # Folder into which results are saved:
    save_folder = 'Multi_dueling_mixed_initiative/'

    filename_part1 = save_folder + 'Opt_2D_900_' + str(num_samples) + '_samples_' \
                     + 'vary_obj_run_'

    # Plot mean +/- stdev:
    plot_avg_objective_vals(filename_part1, filename_part2, num_trials, num_runs, 
                               fig_num, plot_mean_SD = True, line_plot = False,
                               color = colors[i], norm = True, alpha = alpha,
                               mean_linewidth = 2)

# Plot preference buffer trials, multi-dueling:
buffer_size = 1

# Folder into which results are saved:
save_folder = 'Buffer_dueling/'

filename_part1 = save_folder + 'Opt_2D_900_buffer_' + str(buffer_size) + \
               '_vary_obj_run_'

# Plot mean +/- stdev:
plot_avg_objective_vals(filename_part1, filename_part2, num_trials, num_runs, 
                           fig_num, plot_mean_SD = True, line_plot = False,
                           color = colors[2], norm = True, alpha = alpha,
                           mean_linestyle = 'dotted', mean_linewidth = 2)

# Plot preference buffer trials, mixed-initiative:

# Folder into which results are saved:
save_folder = 'Buffer_dueling_mixed_initiative/'

filename_part1 = save_folder + 'Opt_2D_900_buffer_' + str(buffer_size) + \
               '_vary_obj_run_'

# Plot mean +/- stdev:
plot_avg_objective_vals(filename_part1, filename_part2, num_trials, num_runs, 
                           fig_num, plot_mean_SD = True, line_plot = False,
                           color = colors[2], norm = True, alpha = alpha,
                           mean_linewidth = 2)

plt.xlabel('Number of objective function evaluations')
plt.ylabel('Objective function value')
plt.ylim([0.4, 1])

plt.legend(['n = 2, b = 0', 'n = 2, b = 0, coactive', 
            'n = 3, b = 0', 'n = 3, b = 0, coactive',
            'n = 1, b = 1', 'n = 1, b = 1, coactive'])
                 
   
#%% Plot color-blind-friendly palette:    
    
#CB_color_cycle = ['#377eb8', '#ff7f00', '#4daf4a',
#                  '#f781bf', '#a65628', '#984ea3',
#                  '#999999', '#e41a1c', '#dede00']        
    
#plt.figure()
#    
#for i, color in enumerate(CB_color_cycle):
#
#    plt.plot([0, 1], [i, i], c = color)
    