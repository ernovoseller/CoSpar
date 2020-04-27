# -*- coding: utf-8 -*-
"""
Functions for obtaining preference and coactive feedback. They are used by the
2D objective function simulations.
"""

import numpy as np

def get_objective_value(point_idx, objective):
    """
    Evaluates the objective function at x.
    
    Inputs: 1) point_idx is the index of the point at which to evaluate the
               objective. (This is the point's index in the prior covariance 
               matrix, as well as the row index in the points_to_sample array.)
            2) n-by-m NumPy array of the objective function value in each
               grid cell. (n = m = 30 for ICRA simulations.)
    
    Output: scalar value of the objective function at the queried point.
    """
    
    return objective.flatten()[point_idx]

           
def get_preference(pt1, pt2, objective):
    """
    Obtain a preference between two points by preferring whichever point has a 
    higher objective function value; break ties randomly.
    
    NOTE: this differs from the get_gait_preference function used by the 
    compass-gait biped simulations, in that higher objective function values 
    are preferred, rather than lower ones.
    
    Inputs: pt1 and pt2 are the two points to be compared. Both points should 
    be arrays with length equal to the dimensionality of the input space (i.e.,
    2D for ICRA simulations).    
    
    Output: 0 = pt1 preferred; 1 = pt2 preferred.
    """
    
    obj1 = get_objective_value(pt1, objective)
    obj2 = get_objective_value(pt2, objective)
    
    if obj2 > obj1:
        return 1
    elif obj1 > obj2:
        return 0
    else:
        return np.random.choice(2)

def get_coactive_feedback(point_idx, gradient, points_to_sample, 
                          points_in_each_dim, gradient_thresholds, increments):
    """
    This function returns coactive feedback (i.e., a suggested improvement) for
    the specified point. To do this, we use information about the objective 
    function's gradient at the queried point. The gradient component with the
    largest magnitude is compared to two thresholds. If its magnitude is lower
    than both thresholds, then no coactive feedback is given. If its magnitude
    lies between the two thresholds, then the coactive feedback updates the
    queried point by a smaller amount. If its magnitude lies above both 
    thresholds, then the coactive feedback updates the queried point by a 
    larger amount.
    
    Inputs:
        1) point_idx: index of the queried point.
        2) gradient: d x n x m NumPy array, where d = dimensionality of the
           input space, and n x m are the dimensions of the objective function.
           (n = m = 30 in ICRA simulations. d = 2, since functions are 2D.)
        3) points_to_sample: nm x d array, in which each row specifies the
           coordinates of a point in the input space (over which we are
           optimizing).
        4) points_in_each_dim: length-d list, in which each element is a 
           1D array of the (discretized) values in the input space in the
           corresponding dimension.
           Note: points_in_each_dim[i] = np.unique(points_to_sample[:, i]).
        5) gradient_thresholds: thresholds to use for gradient magnitudes. 
           2-by-d NumPy array, in which row 0 corresponds to the lower 
           threshold, and row 1 corresponds to the higher threshold. Thresholds
           are defined separately for each input space dimension.
        6) increments: amount (in terms of grid cells) to increment point_idx
           when determining the coactive feedback. 2-by-d NumPy array, in which 
           row 0 corresponds to the lower threshold, and row 1 corresponds to 
           the higher threshold. Increments are defined separately for each 
           input space dimension.
           
    Output: if no coactive feedback is given, return np.empty(0). Otherwise,
            return the scalar index of the coactive feedback point.
    """
        
    state_dim = increments.shape[1]    # Dimension of the input space
    
    point = points_to_sample[point_idx, :]   # Coordinates of queried point
    
    # Get point index within each dimension:
    point_idx_dims = np.zeros(state_dim).astype(int)
    
    for i in range(state_dim):
        point_idx_dims[i] = np.where(points_in_each_dim[i] == point[i])[0][0]
    
    # Get values of the gradient's components at the given point:
    if state_dim == 1:
        gradient_vals = np.array([gradient[point_idx]])
        
    else:
        gradient_vals = np.empty(state_dim)
    
        for i in range(state_dim):
            gradient_vals[i] = gradient[i][tuple(point_idx_dims)]
            
    # Find the component of the gradient with the largest magnitude:
    abs_gradient_vals = np.abs(gradient_vals)
    largest_magn = np.max(abs_gradient_vals)
    
    # Find which element of the gradient has the largest magnitude:
    largest_magn_idx = np.where(abs_gradient_vals == largest_magn)[0]
    largest_magn_idx = np.random.choice(largest_magn_idx)   # In case of tie    
    
    # Determine which threshold bin this falls into:
    gradient_thresholds = np.append([0], gradient_thresholds[:, largest_magn_idx])
    perc_bin = np.max(np.where(largest_magn >= gradient_thresholds)[0])
    
    if perc_bin == 0:  # Gradient not steep enough to suggest coactive feedback
        
        return np.empty(0)
    
    else:
        perc_bin -= 1
    
    # Sign of gradient at this point
    sign = 2 * int(gradient_vals[largest_magn_idx] > 0) - 1
    
    # Increment point according to percentile bin and gradient sign:
    prev_idx = point_idx_dims[largest_magn_idx]
    
    new_idx = prev_idx + \
                    sign * increments[perc_bin, largest_magn_idx]
    
    # Make sure the value is still within the allowable bounds:
    if new_idx < 0:
        new_idx = 0
    elif new_idx > len(points_in_each_dim[largest_magn_idx]) - 1:
        new_idx = len(points_in_each_dim[largest_magn_idx]) - 1
    
    # If the value hasn't changed (i.e. it was already at the edge), do not
    # return coactive feedback:
    if prev_idx == new_idx:
        return np.empty(0)
    
    # Update and return the new point as coactive feedack:
    point_idx_dims[largest_magn_idx] = new_idx
    
    # Convert to scalar index:
    point_idx = 0
    prod = 1
    
    for i in range(state_dim - 1, -1, -1):
        
        point_idx += point_idx_dims[i] * prod
        prod *= len(points_in_each_dim[i])
        
    return point_idx
