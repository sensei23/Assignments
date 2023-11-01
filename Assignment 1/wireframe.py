import numpy as np
import matplotlib.pyplot as plt
import cv2

def plot_wireframe(projections, image):
    '''
    projections: n_samples X 2 ((height * width) X 2D point coordinate)
    image: on which wireframe has to be plotted
    '''
    # Plotting image points on the image
    plt.scatter(projections[:, 0], projections[:, 1], color='r', s=13, lw=1)

    # This is plotting vertical lines
    # We are finding two points across which we want to join a line. For this we are finding the nth and n+6th point for every line
    # This can be changes with you use case
    for (u, v) in zip(range(0, 48, 6), range(5, 48, 6)):
        plt.plot([projections[u, 0], projections[v, 0]], [projections[u, 1], projections[v, 1]], color='r')

    # Here the points are plotting horizontal lines
    for (u, v) in zip(range(6), range(42, 48)):
        plt.plot([projections[u, 0], projections[v, 0]], [projections[u, 1], projections[v, 1]], color='r')
    
    plt.imshow(image)