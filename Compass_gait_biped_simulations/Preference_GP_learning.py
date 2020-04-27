
"""
These are functions used in preference GP learning, for learning a Bayesian 
preference model given preference data and drawing samples from this model.
"""

import numpy as np
from scipy.stats import norm
from scipy.optimize import minimize


def advance(posterior_model, num_samples):
    """
    Draw a specified number of samples from the preference GP Bayesian model 
    posterior.
    
    Inputs:
        1) posterior_model: this is the model posterior, represented as a 
           dictionary of the form {'mean': post_mean, 'cov_evecs': evecs, 
           'cov_evals': evals}; post_mean is the posterior mean, a length-n 
           NumPy array in which n is the number of points over which the 
           posterior is to be sampled. cov_evecs is an n-by-n NumPy array in 
           which each column is an eigenvector of the posterior covariance,
           and evals is a length-n array of the eigenvalues of the posterior 
           covariance.
           
        2) num_samples: the number of samples to draw from the posterior; a 
           positive integer.
           
    Outputs:
        1) A num_samples-length NumPy array, in which each element is the index
           of a sample.
        2) A num_samples-length list, in which each entry is a sampled reward
           function. Each reward function sample is a length-n vector (see 
           above for definition of n).
    
    """
    
    samples = np.empty(num_samples)    # To store the sampled actions
    
    # Unpack model posterior:
    mean = posterior_model['mean']
    cov_evecs = posterior_model['cov_evecs']
    cov_evals = posterior_model['cov_evals']
    
    num_features = len(mean)
    
    R_models = []       # To store the sampled reward functions
    
    # Draw the samples:
    for i in range(num_samples):
    
        # Sample reward function from GP model posterior:
        X = np.random.normal(size = num_features)
        R = mean + cov_evecs @ np.diag(np.sqrt(cov_evals)) @ X
        
        R = np.real(R)
        
        samples[i] = np.argmax(R) # Find where reward function is maximized
        
        R_models.append(R)        # Store sampled reward function
        
    return samples.astype(int), R_models
    

def feedback(data, labels, GP_prior_cov_inv, preference_noise, 
             r_init = []):
    """
    Function for updating the GP preference model given data.

    Inputs (m = number of preferences):
        1) data: m x 2 NumPy array in which each row contains the indices
           of the two points compared in the m-th preference.
        2) labels: length-m NumPy array, in which each element is 0 (1st 
           queried point preferred), 1 (2nd point preferred), or 0.5 (no 
           preference).
        3) GP_prior_cov_inv: n-by-n NumPy array, where n is the number of 
           points over which the posterior is to be sampled 
        4) preference_noise: positive scalar parameter. Higher values indicate
           larger amounts of noise in the expert preferences.
        5) (Optional) initial guess for convex optimization; length-n NumPy
           array when specified.
               
    Output: the updated model posterior, represented as a dictionary of the 
           form {'mean': post_mean, 'cov_evecs': evecs, 'cov_evals': evals};
           post_mean is the posterior mean, a length-n NumPy array in which n
           is the number of points over which the posterior is to be sampled.
           cov_evecs is an n-by-n NumPy array in which each column is an
           eigenvector of the posterior covariance, and evals is a length-n 
           array of the eigenvalues of the posterior covariance.
    
    """   
    num_features = GP_prior_cov_inv.shape[0]

    # Remove any preference data recording no preference:
    labels = labels.flatten()
    pref_indices = np.where((data[:, 0] != data[:, 1]) & (labels != 0.5))[0]
    data = data[pref_indices, :].astype(int)
    labels = labels[pref_indices].astype(int)

    # Solve convex optimization problem to obtain the posterior mean reward 
    # vector via Laplace approximation:    
    if r_init == []:
        r_init = np.zeros(num_features)    # Initial guess

    res = minimize(preference_GP_objective, r_init, args = (data, labels, 
                   GP_prior_cov_inv, preference_noise), method='L-BFGS-B', 
                   jac=preference_GP_gradient)
    
    # The posterior mean is the solution to the optimization problem:
    post_mean = res.x

    # Obtain inverse of posterior covariance approximation by evaluating the
    # objective function's Hessian at the posterior mean estimate:
    post_cov_inverse = preference_GP_hessian(post_mean, data, labels, 
                   GP_prior_cov_inv, preference_noise) 

    # Calculate the eigenvectors and eigenvalues of the inverse posterior 
    # covariance matrix:
    evals, evecs = np.linalg.eigh(post_cov_inverse)

    # Invert the eigenvalues to get the eigenvalues corresponding to the 
    # covariance matrix:
    evals = 1 / evals 
    
    # Return the model posterior:
    return {'mean': post_mean, 'cov_evecs': evecs, 'cov_evals': evals}

    
def preference_GP_objective(f, data, labels, GP_prior_cov_inv, preference_noise):
    """
    Evaluate the optimization objective function for finding the posterior 
    mean of the GP preference model (at a given point); the posterior mean is 
    the minimum of this (convex) objective function.
    
    Inputs:
        1) f: the "point" at which to evaluate the objective function. This is
           a length-n vector, where n is the number of points over which the 
           posterior is to be sampled.
        2)-5): same as the descriptions in the feedback function. 
        
    Output: the objective function evaluated at the given point (f).
    """
    
    
    obj = 0.5 * f @ GP_prior_cov_inv @ f
    
    num_samples = data.shape[0]
    
    for i in range(num_samples):   # Go through each pair of data points
        
        data_pts = data[i, :]    # Data points queried in this sample
        label = labels[i]
        
        z = (f[data_pts[label]] - f[data_pts[1 - label]]) / (np.sqrt(2) * preference_noise)
        obj -= np.log(norm.cdf(z))
        
    return obj


def preference_GP_gradient(f, data, labels, GP_prior_cov_inv, preference_noise):
    """
    Evaluate the gradient of the optimization objective function for finding 
    the posterior mean of the GP preference model (at a given point).
    
    Inputs:
        1) f: the "point" at which to evaluate the gradient. This is a length-n
           vector, where n is the number of points over which the posterior 
           is to be sampled.
        2)-5): same as the descriptions in the feedback function. 
        
    Output: the objective function's gradient evaluated at the given point (f).
    """
    
    grad = GP_prior_cov_inv @ f    # Initialize to 1st term of gradient
    
    num_samples = data.shape[0]
    
    for i in range(num_samples):   # Go through each pair of data points
        
        data_pts = data[i, :]    # Data points queried in this sample
        label = labels[i]
        
        s_pos = data_pts[label]
        s_neg = data_pts[1 - label]
        
        z = (f[s_pos] - f[s_neg]) / (np.sqrt(2) * preference_noise)
        
        value = (norm.pdf(z) / norm.cdf(z)) / (np.sqrt(2) * preference_noise)
        
        grad[s_pos] -= value
        grad[s_neg] += value
        
    return grad

def preference_GP_hessian(f, data, labels, GP_prior_cov_inv, preference_noise):
    """
    Evaluate the Hessian matrix of the optimization objective function for 
    finding the posterior mean of the GP preference model (at a given point).
    
    Inputs:
        1) f: the "point" at which to evaluate the Hessian. This is
           a length-n vector, where n is the number of points over which the 
           posterior is to be sampled.
        2)-5): same as the descriptions in the feedback function. 
        
    Output: the objective function's Hessian matrix evaluated at the given 
            point (f).
    """
    
    num_samples = data.shape[0]
    
    Lambda = np.zeros(GP_prior_cov_inv.shape)
    
    for i in range(num_samples):   # Go through each pair of data points
        
        data_pts = data[i, :]    # Data points queried in this sample
        label = labels[i]
        
        s_pos = data_pts[label]
        s_neg = data_pts[1 - label]
        
        z = (f[s_pos] - f[s_neg]) / (np.sqrt(2) * preference_noise)   
        
        ratio = norm.pdf(z) / norm.cdf(z)
        value = ratio * (z + ratio) / (2 * preference_noise**2)
        
        Lambda[s_pos, s_pos] += value
        Lambda[s_neg, s_neg] += value
        Lambda[s_pos, s_neg] -= value
        Lambda[s_neg, s_pos] -= value
    
    return GP_prior_cov_inv + Lambda


