import sys
import os
import numpy as np
from skimage import measure
from functools import partial
from scipy.interpolate import splprep, splev
from scipy.integrate import simps
from scipy.ndimage.filters import uniform_filter
from scipy import ndimage
from skimage import filters, feature, morphology
import matplotlib
import matplotlib.pyplot as plt
import warnings
from snakes import fit_snake

warnings.filterwarnings("ignore",category=matplotlib.cbook.mplDeprecation)


def enhance_ridges(frame, mask=None):
    """Detect ridges (larger hessian eigenvalue)"""
    blurred = filters.gaussian(frame, 2)
    Hxx, Hxy, Hyy = feature.hessian_matrix(blurred, sigma=4.5, mode='nearest', order="xy")
    ridges = feature.hessian_matrix_eigvals(Hxx, Hxy, Hyy)[1]
    return np.abs(ridges)



def mask_to_boundary_pts(mask, pt_spacing=10):
    """
    Convert a binary image containing a single object to a set
    of 2D points that are equally spaced along the object's contour.
    """
    # interpolate boundary
    boundary_pts = measure.find_contours(mask, 0)[0]
    tck, u = splprep(boundary_pts.T, u=None, s=0.0, per=1)
    u_new = np.linspace(u.min(), u.max(), 1000)
    x_new, y_new = splev(u_new, tck, der=0)

    # get equi-spaced points along spline-interpolated boundary
    x_diff, y_diff = np.diff(x_new), np.diff(y_new)
    S = simps(np.sqrt(x_diff**2 + y_diff**2))
    N = int(round(S/pt_spacing))

    u_equidist = np.linspace(0, 1, N+1)
    x_equidist, y_equidist = splev(u_equidist, tck, der=0)
    return np.array(list(zip(x_equidist, y_equidist)))



# load data: the raw image an a binary region-of-interest image
im = np.load('cropped_frame.npy')
mask = np.load('enlarged_mask.npy')
# mask = np.load('shifted_mask.npy')
# mask = np.load('shrunken_mask.npy')
# mask = np.load('target_mask.npy')

# get boundary points of mask
boundary_pts = mask_to_boundary_pts(mask, pt_spacing=3)
x, y = boundary_pts[:,1], boundary_pts[:,0]


# distance from ridge midlines
ridges = enhance_ridges(im)
thresh = filters.threshold_otsu(ridges)
prominent_ridges = ridges > 0.8*thresh
skeleton = morphology.skeletonize(prominent_ridges)
edge_dist = ndimage.distance_transform_edt(~skeleton)
edge_dist = filters.gaussian(edge_dist, sigma=2)


# distance from skeleton branch points
blurred_skeleton = uniform_filter(skeleton.astype(float), size=3)
corner_im = blurred_skeleton > 4./9
corners_labels = measure.label(corner_im)
corners = np.array([region.centroid for region in measure.regionprops(corners_labels)])



# show the intermediate images
plt.gray()
plt.ion()
plt.subplot(221)
plt.imshow(im)
plt.title('original image')
plt.axis('off')
plt.subplot(222)
plt.imshow(ridges)
plt.title('ridge filter')
plt.axis('off')
plt.subplot(223)
plt.imshow(skeleton)
plt.plot(corners[:,1], corners[:,0], 'ro')
plt.title('ridge skeleton w/ branch points')
plt.axis('off')
plt.subplot(223)
plt.imshow(skeleton)
plt.autoscale(False)
plt.plot(corners[:,1], corners[:,0], 'ro')
plt.title('ridge skeleton w/ branch points')
plt.subplot(224)
plt.imshow(edge_dist)
plt.title('distance transform of skeleton')
plt.axis('off')
plt.ioff()
plt.show()


# show an animation of the fitting procedure
fig = plt.figure()
plt.imshow(im, cmap='gray')
plt.plot(x, y, 'bo')
line_obj, = plt.plot(x, y, 'ro')
plt.axis('off')
    
plt.ion()
plt.pause(0.01)
snake_pts = fit_snake(boundary_pts, edge_dist, nits=60, alpha=0.5, beta=0.2, point_plot=line_obj)
plt.ioff()
plt.pause(0.01)
plt.show()
    