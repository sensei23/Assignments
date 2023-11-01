import numpy as np
import scipy.signal as sig

def LucasKanade(prev_gray, next_gray, points, win_size=(15, 15)):
    # Get the height and width of the images
    h, w = prev_gray.shape[:2]
    
    # Get the window size for the Lucas-Kanade method
    win_h, win_w = win_size
    
    # Get the number of points to track
    n_points = points.shape[0]
    
    # Get the gradient of the previous and next images
    Ix, Iy = np.gradient(prev_gray)
    It = next_gray - prev_gray
    
    # Initialize an array for the optical flow
    optical_flow = np.zeros((n_points, 2))
    
    # Loop over each point to track
    for i, (x, y) in enumerate(points):
        # Get the window around the current point in the gradient images
        Ix_win = Ix[y-win_h//2:y+win_h//2+1, x-win_w//2:x+win_w//2+1].flatten()
        Iy_win = Iy[y-win_h//2:y+win_h//2+1, x-win_w//2:x+win_w//2+1].flatten()
        It_win = It[y-win_h//2:y+win_h//2+1, x-win_w//2:x+win_w//2+1].flatten()
        
        # Create the gradient matrix
        gradient_matrix = np.vstack((Ix_win, Iy_win)).T
        
        # Solve for the optical flow using the least-squares method
        flow, _, _, _ = np.linalg.lstsq(gradient_matrix, -It_win, rcond=None)
        
        # Add the optical flow for the current point to the overall optical flow
        optical_flow[i] = flow
        
    # Return the optical flow
    return optical_flow

# You can then call the LucasKanade function in your code by passing in the previous 
# and next grayscale images and the points to track, and it will return the optical flow as the difference between the new and old points. 
# The win_size parameter can be adjusted to control the size of the window for the Lucas-Kanade method.