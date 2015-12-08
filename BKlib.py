from functools import partial
from scipy import optimize, ndimage
import itertools
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
import subprocess
import cv2
from pykalman import KalmanFilter
import platform
import pims
from numpy import ma
from matplotlib.colors import BoundaryNorm, ListedColormap
import time


"""
Some functions that I've found helpful. I'm sure this is reinventing the wheel,
but whatever.

"""

def isplit(iterable, splitters):
    """
    Splits a list about a particular element into a list-of-lists
    thanks to this guy: http://stackoverflow.com/questions/4322705/split-a-list-into-nested-lists-on-a-value
    """
    return [list(g) for k,g in itertools.groupby(iterable,lambda x:x in splitters) if not k]
    

def partition(list_, num):
    # Partition list as evenly as possible.
    part_sizes = [len(list_) / int(num) for _ in range(num)]
    remainder = len(list_) % num    
    for part_num in range(remainder):
        part_sizes[part_num] += 1    
    end_inds = np.cumsum(part_sizes)
    sta_inds = [end - size for end, size in zip(end_inds, part_sizes)]
    return [list_[sta_ind:end_ind] for sta_ind, end_ind in zip(sta_inds, end_inds)]


def partition_indices(list_, num):
    # Partition list as evenly as possible.
    part_sizes = [len(list_) / int(num) for _ in range(num)]
    remainder = len(list_) % num    
    for part_num in range(remainder):
        part_sizes[part_num] += 1    
    end_inds = np.cumsum(part_sizes)
    sta_inds = [end - size for end, size in zip(end_inds, part_sizes)]
    return zip(sta_inds, end_inds)


def read_config_file(fn):
    """
    Read a simple config file. More complex configs should be in xml or yaml.
    Values should be in format "key=value" or "key value". Values are converted
    to int's or float's if possible; if not, it's a string.
    """
    config = dict()
    with open(fn) as f:
        for line in f.readlines():
            line = line.rstrip()  # remove newline character
            
            # skip blank lines and comment lines
            if not line or line[0]=='#':
                continue
            
            # remove inline comments
            ind = line.find('#')
            if ind != -1:
                line = line[:ind].strip()
            
            # parse
            if '=' in line:
                line_parts = line.split('=')
            else:
                line_parts = line.split(' ')
            if len(line_parts) != 2:
                print 'Could not parse line', line, ', skipping...'
                continue
            
            # cast to appropriate type
            key, val_str = line_parts[0].strip(), line_parts[1].strip()
            try:
                config[key] = int(val_str)
            except ValueError:
                try:
                    config[key] = float(val_str)
                except ValueError:
                    config[key] = val_str.replace('"','')

    return config            
    

def print_image_properties(im):
    if not isinstance(im, (np.ndarray)):
        raise TypeError('print_image_properties only handles 2D or 3D numpy arrays')
    try:
        nchan = im.shape[3]
    except IndexError:
        nchan = 1
    print
    print 'image size:    %i x %i' % (im.shape[0], im.shape[1])
    print 'num. channels: %i' % nchan
    print 'dtype: %s' % im.dtype
    print 'min, max: %.1f, %.1f' % (np.min(im), np.max(im))
    print 'mean, stdev: %.1f, %.1f' % (np.mean(im), np.std(im))
    

def vidshow(frames, start_frame=0, end_frame=-1, fps=10, **kwargs):
    # similar to imshow, but for arrays with a time dimension
    if not isinstance(frames, np.ndarray):
        raise TypeError('vidshow requires a 3D or 4D numpy array')
    if len(frames.shape) == 3:
        is_color = False
    elif len(frames.shape) == 4:
        is_color = True
        if frames.shape[3] != 3:
            raise IndexError('vidshow only knows how to display 3-channel frames')
    else:
        raise IndexError('vidshow requires a 3D or 4D numpy array')
    
    frames = frames[start_frame:end_frame]
    
    plt.gray()
    im = plt.imshow(frames[0], **kwargs)
    for frame_num, frame in enumerate(frames):
        im.set_data(frame)        
        plt.pause(1./fps)
    plt.show()
    
    
def flip_dim(a, axis=0): 
    # like numpy.fliplr or numpy.flipud but works on arbitrary dimension
    idx = [slice(None)]*len(a.shape)
    idx[axis] = slice(None, None, -1)
    return a[idx]


def tiff_to_ndarray(fn):
    """
    Load a tiff stack as 3D numpy array.
    You must have enough RAM to hold the whole movie in memory.
    """
    frames = pims.TiffStack(fn)
    num_frames = len(frames)
    sz = frames.frame_shape
    arr = np.empty((num_frames, sz[0], sz[1]), dtype=frames.pixel_type)
    for frame_num, frame in enumerate(frames):
        arr[frame_num, :, :] = np.fliplr(np.swapaxes(frame, 0, 1))
    return arr


def imshow_overlay(im, mask, alpha=0.5, color='red', **kwargs):
    """Show semi-transparent red mask over an image"""
    mask = mask > 0
    mask = ma.masked_where(~mask, mask)        
    plt.imshow(im, **kwargs)
    plt.imshow(mask, alpha=alpha, cmap=ListedColormap([color]))


class AviReader:

    """Read a file as an immutable, iterable, sliceable  sequence of frames."""

    def __init__(self, fn):
        self.cap = cv2.VideoCapture(fn)
        self.first_frame = 0
        self.last_frame = int(self.cap.get(cv2.CAP_PROP_FRAME_COUNT))
        self.fn = fn
        self.num_frames = len(self)
        self.frame_rate = int(self.cap.get(cv2.CAP_PROP_FPS))
        self.frame_size = (self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT),
                           self.cap.get(cv2.CAP_PROP_FRAME_WIDTH))

    def __len__(self):
        return self.last_frame - self.first_frame

    def __iter__(self):
        self.cap.set(cv2.CAP_PROP_POS_FRAMES, self.first_frame)
        return self

    def next(self):
        current_frame = int(self.cap.get(cv2.CAP_PROP_POS_FRAMES))
        if current_frame >= self.last_frame:
            raise StopIteration
        else:
            _, frame = self.cap.read()
            return frame

    def __str__(self):
        repr_str = 'AviReader instance from '+self.fn+': '
        repr_str += str(len(self))+' frames of shape '+str(self.frame_size())
        repr_str += ', ' +str(self.frame_rate())+' fps'
        return repr_str

    def __getitem__(self, index):
        # FIXME: doesn't handle step (stride), nor negative slice indices
        if isinstance(index, int):  # single frame
            if index < 0:
                index = len(self) + index
            if index + self.first_frame > self.last_frame:
                raise IndexError
            self.cap.set(cv2.CAP_PROP_POS_FRAMES, index + self.first_frame)
            ok, frame = self.cap.read()
            if ok:
                return frame
            else:
                raise IndexError
        elif isinstance(index, slice):  # slice
            self.first_frame = index.start
            # if index.stop is not None:  # FIXME: why doesn't this work with [sta:] indexing?
            if index.stop <= self.last_frame:
                self.last_frame = index.stop
            return self
        else:
            raise TypeError('Avi indices should be integer or slices')



class TifReader:

    """An immutable, iterable sequence of frames."""

    def __init__(self, fn):
        self.fn = fn
        self.im = Image.open(fn)
        self.first_frame = 0
        idx = 0
        while True:        
            try:
                self.im.seek(idx)
            except EOFError:
                self.last_frame = idx
                break
            idx += 1
        self.num_frames = self.last_frame
        self._total_frames = self.num_frames
        self.frame_size = self.im.size

    def __len__(self):
        return self.last_frame - self.first_frame

    def __iter__(self):
        self.iter_frame = 0
        return self

    def next(self):
        if self.iter_frame + self.first_frame >= self.last_frame:
            raise StopIteration
        else:
            self.im.seek(self.first_frame + self.iter_frame)
            self.iter_frame += 1        
        return np.array(self.im)

    def __str__(self):
        repr_str = self.__class__.__name__+' instance from '+self.fn+': '
        repr_str += str(len(self))+' frames of shape '+str(self.frame_size)
        return repr_str

    def __getitem__(self, index):
        # FIXME: doesn't handle step (stride), nor slicing w negative indices
        if isinstance(index, int):  # single frame
            if index < 0:
                index = len(self) + index
            if index + self.first_frame > self.last_frame:
                raise IndexError
            self.im.seek(index + self.first_frame)
            return np.array(self.im)
        elif isinstance(index, slice):  # slice
            self.first_frame = index.start
            # if index.stop is not None:  # FIXME: why doesn't this work with [sta:] indexing?
            if index.stop <= self.last_frame:
                self.last_frame = index.stop
            return self
        else:
            raise TypeError('Avi indices should be integer or slices')
    
    @property
    def shape(self):
        sz = self[0].shape
        return (sz[0], sz[1], len(self))
    


def write_video(frames, filename, fps=20):
    """ 
    Uses avconv to write a 3D numpy array to a video file. 
    Currently only supports grayscale arrays.    
    """
    
    # On Mac systems, copy ffmeg binaries to your PATH (http://ffmpegmac.net/)
    
    if platform.system() == 'Windows':
        err_str = 'Don\'t know how to write a movie for %s platform' % platform.system()
        raise NotImplementedError(err_str)

    
    if len(frames.shape) == 4:
        pix_fmt = 'rgb24'
    else:
        pix_fmt = 'gray'
    
    # normalize
    max_pix_val = np.percentile(frames, 99.9)
    if frames.dtype in (np.bool, bool):
        frames = frames.astype(np.uint8)
    frames -= frames.min()
    frames[frames>max_pix_val] = max_pix_val
    if max_pix_val > 0:
            frames *= 255. / max_pix_val
    frames = frames.astype(np.uint8)
    
    # figure out which av program is installed
    program_name = ''
    try:
        subprocess.check_call(['avconv', '-h'])
        program_name = 'avconv'
    except OSError:
        try:
            subprocess.check_call(['ffmpeg', '-h'])
            program_name = 'ffmpeg'
        except OSError:
            pass
    if not program_name:
        raise OSError('Can\'t find avconv or ffmpeg')
    
    # prepare pipe to av converter program
    size_str = '%ix%i' % (frames.shape[1], frames.shape[2])
    cmd = [program_name,
            '-y', # (optional) overwrite output file if it exists
            '-f', 'rawvideo',
            '-vcodec','rawvideo',
            '-s', size_str, # size of one frame
            '-pix_fmt', pix_fmt,
            '-r', str(fps), # frames per second
            '-i', '-', # The imput comes from a pipe
            '-an', # Tells FFMPEG not to expect any audio
            '-qscale', '4',
            '-vcodec','mjpeg',
            filename]
    pipe = subprocess.Popen(cmd, stdin=subprocess.PIPE)

    # write frames            
    for frame in frames:
        frame = np.fliplr(frame)
        pipe.stdin.write(frame.tostring())
    pipe.stdin.close()
    pipe.wait()


def label_im_to_color(im, cmap='jet'):
    im = im.astype(float)
    im -= np.min(im)
    im /= np.max(im)
    cmap = plt.cm.get_cmap(cmap)
    return cmap(im)



class KalmanSmoother2D:
    
    def __init__(self, x_noise, y_noise, smoothness_x=1, smoothness_y=1):
        
        dt = 1
        
        # model
        F = np.eye(4)
        F[0, 2] = dt
        F[1, 3] = dt

        H = np.zeros((2, 4))
        H[0, 0] = 1
        H[1, 1] = 1

        R = np.zeros((2, 2))
        R[0, 0] = x_noise * x_noise
        R[1, 1] = y_noise * y_noise

        sigma_ax, sigma_ay = 1, 1
        G = np.zeros((4, 1))
        G[2] = sigma_ax*dt
        G[3] = sigma_ay*dt

        Q = np.transpose(G)*G
        Q[0, 1] = 0; Q[1, 0] = 0
        Q[0, 3] = 0; Q[3, 0] = 0
        Q[1, 2] = 0; Q[2, 1] = 0
        Q[2, 3] = 0; Q[3, 2] = 0

        # initialize filter
        self.kf = KalmanFilter()
        self.kf.transition_matrices = F
        self.kf.observation_matrices = H
        self.kf.transition_covariance = Q
        self.kf.observation_covariance = R

        # default initial state
        # TODO: maybe use first measurement as default?
        self.kf.initial_state_mean = np.zeros((4,))
        self.kf.initial_state_covariance = np.zeros((4, 4))
        
    # TODO: get innovations?
        
    def set_initial_state(self, initial_mean, initial_covariance=np.zeros((4,4))):
        if initial_mean.shape[0] == 2:
            print 'initial velocity unspecified, assuming v0 = 0'
            initial_mean = np.array([initial_mean[0], initial_mean[1], 0, 0])
        self.kf.initial_state_mean = initial_mean
        self.kf.initial_state_covariance = initial_covariance
        
    def set_measurements(self, measurements):
        self.smooth_means, self.smooth_covs = self.kf.smooth(measurements)
        
    def get_smoothed_measurements(self):
        return self.smooth_means[:,0:2]
    
    def get_velocities(self):
        return self.smooth_means[:,2:]
        
    def get_covariances(self):
        return self.smooth_covs



def snake_energy(flattened_pts, edge_dist, corner_dist, alpha, beta, gamma):
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
        corner_dist (2D numpy array): Distance transform of binary interest point detector.
        alpha (float): The relative weight given to unevenly spaced points. A higher
            value encourages evenly-spaced points. Should be > 0.
        beta (float): The weight given to local curvature. A higher value encourages
            flat contours.
        gamma (float): The relative weights given to the distance from edges or corners.
            A gamma of 0 means to consider only the edges; gamma=1 means we use only
            the corner distance image.
    
    Returns:
        float: Image energy. (lower is better)
    """

    pts = np.reshape(flattened_pts, (len(flattened_pts)/2, 2))
    
    # external energy (favors low values of distance image)
    dist_vals = ndimage.interpolation.map_coordinates(edge_dist, [pts[:,0], pts[:,1]], order=1)
    edge_energy = np.sum(dist_vals)
    dist_vals = ndimage.interpolation.map_coordinates(corner_dist, [pts[:,0], pts[:,1]], order=1)
    corner_energy = np.sum(dist_vals)
    external_energy = (1-gamma)*edge_energy + gamma*corner_energy

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

    
def fit_snake(pts, corner_dist, edge_dist, 
              alpha=0.5, beta=0.25, gamma=0.8, 
              point_plot=None, nits=100):
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
        corner_dist (2D numpy array): Distance transform of binary interest point detector.
        alpha (float): The weight given to unevenly spaced points. A higher value encourages
            evenly-spaced points. Should be > 0.
        beta (float): The weight given to local curvature. A higher value encourages
            flat contours.
        gamma (float): The relative weights given to the distance from edges or corners.
            A gamma of 0 means to consider only the edges; gamma=1 means we use only
            the corner distance image.
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
            time.sleep(0.1)
        callback_function.nits = 0
    else:
        callback_function = None
    
    # optimize
    cost_function = partial(snake_energy, alpha=alpha, beta=beta, gamma=gamma, 
                            edge_dist=edge_dist, corner_dist=corner_dist)
    options = {'disp':False}
    options['maxiter'] = nits  # FIXME: check convergence
    method = 'BFGS'  # 'BFGS', 'CG', or 'Powell'. 'Nelder-Mead' has very slow convergence
    res = optimize.minimize(cost_function, pts.ravel(), method=method, options=options, callback=callback_function)
    optimal_pts = np.reshape(res.x, (len(res.x)/2, 2))

    return optimal_pts
















