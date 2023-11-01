import numpy as np

def OpticalFlowLK(prev_gray, next_gray, points, win_size=15, max_iterations=30, epsilon=0.03):
    # Define the window size as half of the specified size
    half_win_size = win_size // 2
    
    # Create an empty array for the optical flow
    flow = np.zeros((points.shape[0], 2), dtype=np.float32)
    
    # Compute the gradient of the next gray image
    next_gradient_x, next_gradient_y = np.gradient(next_gray)
    
    # Loop over each point to track
    for i, point in enumerate(points):
        # Get the x and y coordinate of the point
        x, y = point.astype(np.int32)
        
        # Get the window of the next gray image around the current point
        window = next_gray[y-half_win_size:y+half_win_size+1, x-half_win_size:x+half_win_size+1].flatten()
        
        # Get the window of the gradient of the next gray image along the x and y axis around the current point
        gradient_x = next_gradient_x[y-half_win_size:y+half_win_size+1, x-half_win_size:x+half_win_size+1].flatten()
        gradient_y = next_gradient_y[y-half_win_size:y+half_win_size+1, x-half_win_size:x+half_win_size+1].flatten()
        
        # Compute the Jacobian matrix J
        J = np.column_stack((gradient_x, gradient_y))
        
        # Compute the gradient of the previous gray image at the current point
        prev_gradient = np.array([prev_gray[y, x+1] - prev_gray[y, x-1], prev_gray[y+1, x] - prev_gray[y-1, x]])
        
        # Initialize the optical flow for the current point
        u, v = np.zeros((2,))
        
        # Iterate to refine the optical flow for the current point
        for iteration in range(max_iterations):
            # Compute the error between the current window and the previous window
            error = window - prev_gray[y-half_win_size:y+half_win_size+1, x-half_win_size-int(u):x+half_win_size+1-int(u)] - prev_gradient[0] * u - prev_gradient[1] * v
            
            # Compute the update to the optical flow
            update = np.linalg.lstsq(J, error, rcond=None)[0]
            
            # Update the optical flow
            u += update[0]
            v += update[1]
            
            # Check if the update is small enough to stop the iteration
            if np.abs(update).max() < epsilon:
                break
                
        # Save the





# Computing optical flow from scratch using the Lucas-Kanade method involves finding the best displacement vector for a small window 
# around each pixel in one image that would align it with the corresponding window in the other image. This is done by finding the parameters of a 
# local affine flow model that minimize the difference between the windows in the two images.


import numpy as np

def OpticalFlowLK(prev_gray, next_gray, points, win_size=(15, 15), max_iterations=30, epsilon=0.01):
    # Get the height and width of the images
    h, w = prev_gray.shape[:2]
    
    # Create an empty array for the optical flow
    flow = np.zeros((h, w, 2), dtype=np.float32)
    
    # Define the half window size
    half_win_size = (win_size[0]//2, win_size[1]//2)
    
    # For each point
    for point in points:
        # Get the x and y coordinates of the point
        x, y = point
        
        # Get the window around the point in both images
        prev_window = prev_gray[y-half_win_size[1]:y+half_win_size[1]+1, x-half_win_size[0]:x+half_win_size[0]+1].flatten()
        next_window = next_gray[y-half_win_size[1]:y+half_win_size[1]+1, x-half_win_size[0]:x+half_win_size[0]+1].flatten()
        
        # Create a grid of x and y coordinates for the window
        x_grid, y_grid = np.meshgrid(np.arange(win_size[0]), np.arange(win_size[1]), indexing='ij')
        x_grid = x_grid.flatten() - half_win_size[0]
        y_grid = y_grid.flatten() - half_win_size[1]
        
        # Initialize the displacement vector for the point
        u, v = 0, 0
        
        # Iterate until the maximum number of iterations is reached or the displacement vector changes by less than epsilon
        for iteration in range(max_iterations):
            # Compute the warped window using the current displacement vector
            warped_next_window = next_gray[y+v-half_win_size[1]:y+v+half_win_size[1]+1, x+u-half_win_size[0]:x+u+half_win_size[0]+1].flatten()
            
            # Compute the error between the warped window and the reference window
            error = prev_window - warped_next_window
            
            # Compute the gradient of the warped window in the x and y directions
            gradient_x = cv2.Sobel(next_gray[y+v-half_win_size[1]:y+v+half_win_size[1]+1, x+u-half_win_size[0]:x+u+half_win_size[0]+1], cv2
