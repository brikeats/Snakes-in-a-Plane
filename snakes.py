from functools import partial
from scipy import optimize, ndimage
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt


    
def snake_energy(flattened_pts, edge_dist, alpha, beta):
    """
    Compute the energy associated with a proposed contour. The contour is defined
    by N 2-dimensional points. The energy is comprised of external energy, which is
    derived from the supplied distance images; and internal energy, which is computed
    based only on the characteristics of the contour. Note that the image 
    interpolation uses only 1st-order splines, which increases speed at the 
    expense of accuracy.
    
    The current implementation was created for a closed contour. An open contour 
    formulation should replace the periodic 'np.roll' calls by non-periodic end-off 
    shifts.

    Args:
        flattened_pts ((2*N,)-shaped numpy array): A flattened list of the contour 
            points, ordered so that adjacent points are consecutive in the list. 
            Can be created by calling arr_2d.ravel() on an ordered (N,2)-shaped array 
            of points.
        edge_dist (2D numpy array): Distance transform of binary edge detector.
        alpha (float): The relative weight given to unevenly spaced points. A higher
            value encourages evenly-spaced points. Should be > 0.
        beta (float): The weight given to local curvature. A higher value encourages
            flat contours.
        
    Returns:
        float: Image energy. (lower is better)
    """

    pts = np.reshape(flattened_pts, (int(len(flattened_pts)/2), 2))
    
    # external energy (favors low values of distance image)
    dist_vals = ndimage.interpolation.map_coordinates(edge_dist, [pts[:,0], pts[:,1]], order=1)
    edge_energy = np.sum(dist_vals)
    external_energy = edge_energy

    # spacing energy (favors equi-distant points)
    prev_pts = np.roll(pts, 1, axis=0)
    next_pts = np.roll(pts, -1, axis=0)
    displacements = pts - prev_pts
    point_distances = np.sqrt(displacements[:,0]**2 + displacements[:,1]**2)
    mean_dist = np.mean(point_distances)
    spacing_energy = np.sum((point_distances - mean_dist)**2)

    # curvature energy (favors smooth curves)
    curvature_1d = prev_pts - 2*pts + next_pts
    curvature = (curvature_1d[:,0]**2 + curvature_1d[:,1]**2)
    curvature_energy = np.sum(curvature)
    
    return external_energy + alpha*spacing_energy + beta*curvature_energy

    
def fit_snake(pts, edge_dist, alpha=0.5, beta=0.25, nits=100, point_plot=None):
    """
    Fit an active contour model (aka snakes) based on some initial points and a 
    feature image. Given a list of points as a starting point, it evolves the points
    until they sit at a minimum of the energy function 'snake_energy'. This function
    is not especially good at avoiding local minima, and it does not adapt the number 
    of points in the contour. Therefore, it is most useful for "polishing up" and 
    already good initial guess.

    Args:
        pts ((N,2)-shaped numpy array): A list of the contour points, ordered so that
            adjacent points are consecutive in the list (ie, in clockwise or counter-
            clockwise order).
        edge_dist (2D numpy array): Distance transform of binary edge detector.
        alpha (float): The weight given to unevenly spaced points. A higher value encourages
            evenly-spaced points. Should be > 0.
        beta (float): The weight given to local curvature. A higher value encourages
            flat contours.
        point_plot (matplotlib.lines.Line2D, optional): A matplotlib line object for
            the given points. The Line2D data will be updated on each iteration to 
            provide an animation of the optimization.
    
    Returns:
        (N,2)-shaped numpy array: The points after minimization.
    """
    
    if point_plot:
        def callback_function(new_pts):
            callback_function.nits += 1
            y = new_pts[0::2]
            x = new_pts[1::2]
            point_plot.set_data(x,y)
            plt.title('%i iterations' % callback_function.nits)
            point_plot.figure.canvas.draw()
            plt.pause(0.02)
        callback_function.nits = 0
    else:
        callback_function = None
    
    # optimize
    cost_function = partial(snake_energy, alpha=alpha, beta=beta, edge_dist=edge_dist)
    options = {'disp':False}
    options['maxiter'] = nits  # FIXME: check convergence
    method = 'BFGS'  # 'BFGS', 'CG', or 'Powell'. 'Nelder-Mead' has very slow convergence
    res = optimize.minimize(cost_function, pts.ravel(), method=method, options=options, callback=callback_function)
    optimal_pts = np.reshape(res.x, (int(len(res.x)/2), 2))

    return optimal_pts

