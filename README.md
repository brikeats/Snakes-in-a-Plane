# Snakes in a Plane

Active contour model (aka snakes) implemented in python. 

An active contour model is an image analysis technique which evolves a curve to find boundaries in images. I couldn't find a python implementation of active contour model online, so I implemented it myself. 

This code requires only the standard python scientific stack: numpy, scipy, matplotlib, and skimage. The main function is `fit_snake` in `snakes.py`; this function will evolve a closed contour to fit an image. Starting with an initial guess for the boundary points, `fit_snake` shifts the points around until they sit at a local minimum of the function `snake_energy`. The snake energy is pretty much as described in [the wikipedia article](https://en.wikipedia.org/wiki/Active_contour_model): internal energy terms penalizing stretching, and encouraging smoothness; and external energy terms that push the curve towards edges and "corners."
