import cv2
import numpy as np

def OpticalFlowRefine(prev_gray, next_gray, flow):
    # Get the height and width of the images
    h, w = prev_gray.shape[:2]
    
    # Create an empty array for the refined flow
    refined_flow = np.zeros((h, w, 2), dtype=np.float32)
    
    # Define the window size for the Lucas-Kanade method
    win_size = (15, 15)
    
    # Set the termination criteria for the Lucas-Kanade method
    criteria = (cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 10, 0.03)
    
    # Use the Lucas-Kanade method to refine the optical flow
    for y in range(0, h, win_size[1]):
        for x in range(0, w, win_size[0]):
            # Get the current window of the flow
            window_flow = flow[y:y + win_size[1], x:x + win_size[0]]
            
            # Get the x and y values of the flow in the current window
            u = window_flow[..., 0]
            v = window_flow[..., 1]
            
            # Use the Lucas-Kanade method to refine the flow in the current window
            u, v, _, _ = cv2.calcOpticalFlowPyrLK(prev_gray, next_gray, u, v, None, None, win_size, 3, criteria, 0, 1e-4)
            
            # Add the refined flow to the overall refined flow
            refined_flow[y:y + win_size[1], x:x + win_size[0], 0] = u
            refined_flow[y:y + win_size[1], x:x + win_size[0], 1] = v
            
    # Return the refined optical flow
    return refined_flow

# Load the two images
img1 = cv2.imread("img1.png")
img2 = cv2.imread("img2.png")

# Number of levels
numLevels = 5

# Step 01: Gaussian smooth and scale Img1 and Img2
img1_scaled = cv2.GaussianBlur(img1, (5, 5), 0)
img2_scaled = cv2.GaussianBlur(img2, (5, 5), 0)
img1_scaled = cv2.resize(img1_scaled, None, fx=0.5, fy=0.5)
img2_scaled = cv2.resize(img2_scaled, None, fx=0.5, fy=0.5)

# Step 02: Compute the optical flow at this resolution
prev_gray = cv2.cvtColor(img1_scaled, cv2.COLOR_BGR2GRAY)
next_gray = cv2.cvtColor(img2_scaled, cv2.COLOR_BGR2GRAY)
flow = cv2.calcOpticalFlowFarneback(prev_gray, next_gray, None, 0.5, 3, 15, 3, 5, 1.2, 0)

# Step 03: For each level
for level in range(numLevels - 1, 0, -1):
    # a. Scale Img1 and Img2 by a factor of 2^(1 - level)
    scale = np.power(2, 1 - level)
    img1_scaled = cv2.resize(img1, None, fx=1 / scale, fy=1 / scale)
    img2_scaled = cv2.resize(img2, None, fx=1 / scale, fy=1 / scale)
    prev_gray = cv2.cvtColor(img1_scaled, cv2.COLOR_BGR2GRAY)
    next_gray = cv2.cvtColor(img2_scaled, cv2.COLOR_BGR2GRAY)
    
    # b. Upscale the previous layer‘s optical flow by a factor of 2
    flow = cv2.resize(flow, None, fx=2, fy=2, interpolation=cv2.INTER_CUBIC)
    
    # c. Compute u and v by calling OpticalFlowRefine with the previous level‘s optical flow
    u, v = OpticalFlowRefine(prev_gray, next_gray, flow)

# u and v are the final optical flow values

# ______________________________________________________________________________________________________________

import cv2
import numpy as np

def imsmooth2d(im):
    return cv2.GaussianBlur(im, (3,3), 0)

def imgradient(im):
    return cv2.Sobel(im, cv2.CV_32F,1,0,ksize=3), cv2.Sobel(im, cv2.CV_32F,0,1,ksize=3)

def get_window(im, point, window_size=5):
    x, y = point
    half_size = window_size // 2
    return im[x-half_size:x+half_size+1, y-half_size:y+half_size+1]

def OpticalFlowRefine(Img1, Img2, window_size, u0, v0):
    Ix, Iy = imgradient(Img1)
    It = np.abs(imsmooth2d(Img2) - imsmooth2d(Img1))
    u = np.zeros(Img1.shape)
    v = np.zeros(Img1.shape)
    
    half_size = window_size // 2
    for x in range(half_size, Img1.shape[0] - half_size):
        for y in range(half_size, Img1.shape[1] - half_size):
            window_Ix = Ix[x-half_size:x+half_size+1, y-half_size:y+half_size+1]
            window_Iy = Iy[x-half_size:x+half_size+1, y-half_size:y+half_size+1]
            window_It = It[x-half_size:x+half_size+1, y-half_size:y+half_size+1]
            
            A = np.array([[np.sum(window_Ix**2), np.sum(window_Ix * window_Iy)],
                         [np.sum(window_Ix * window_Iy), np.sum(window_Iy**2)]])
            b = np.array([-np.sum(window_Ix * window_It), -np.sum(window_Iy * window_It)])
            
            delta = np.linalg.inv(A).dot(b)
            u[x, y] = u0 + delta[0]
            v[x, y] = v0 + delta[1]
    return u, v

def MultiScaleLucasKanade(Img1, Img2, window_size, num_levels):
    u = np.zeros(Img1.shape)
    v = np.zeros(Img1.shape)
    scale = 2 ** (1 - num_levels)
    Img1 = cv2.GaussianBlur(Img1, (3,3), 0)
    Img2 = cv2.GaussianBlur(Img2, (3,3), 0)
    Img1

# ____________________________________________________________________________________________________________

import numpy as np
import cv2 as cv

def OpticalFlowRefine(Img1, Img2, windowSize, u0, v0):
    Ix, Iy, It = getGradients(Img1, Img2)
    height, width = Img1.shape[:2]
    sz = windowSize//2
    u = np.zeros((height, width))
    v = np.zeros((height, width))
    for i in range(sz, height-sz):
        for j in range(sz, width-sz):
            Ix_window = Ix[i-sz:i+sz+1, j-sz:j+sz+1]
            Iy_window = Iy[i-sz:i+sz+1, j-sz:j+sz+1]
            It_window = It[i-sz:i+sz+1, j-sz:j+sz+1]
            Ix_window = Ix_window.flatten()
            Iy_window = Iy_window.flatten()
            b = -It_window.flatten()
            A = np.vstack([Ix_window, Iy_window]).T
            delta = np.linalg.lstsq(A, b, rcond=None)[0]
            u[i, j] = u0 + delta[0]
            v[i, j] = v0 + delta[1]
    return u, v

def MultiScaleLucasKanade(Img1, Img2, windowSize, numLevels):
    u_final = np.zeros(Img1.shape[:2])
    v_final = np.zeros(Img1.shape[:2])
    Img1 = cv.GaussianBlur(Img1, (5, 5), 0)
    Img2 = cv.GaussianBlur(Img2, (5, 5), 0)
    scale_factor = 2**(1-numLevels)
    Img1 = cv.resize(Img1, None, fx=scale_factor, fy=scale_factor, interpolation=cv.INTER_LINEAR)
    Img2 = cv.resize(Img2, None, fx=scale_factor, fy=scale_factor, interpolation=cv.INTER_LINEAR)
    u0, v0 = LucasKanade(Img1, Img2, windowSize)
    for level in range(2, numLevels+1):
        scale_factor = 2**(1-level)
        Img1 = cv.resize(Img1, None, fx=scale_factor, fy=scale_factor, interpolation=cv.INTER_LINEAR)
        Img2 = cv.resize(Img2, None, fx=scale_factor, fy=scale_factor, interpolation=cv.INTER_LINEAR)
        u0 = cv.resize(u0, None, fx=2, fy=2,

# ______________________________________________________________________________________________________________

def MultiScaleLucasKanade(Img1, Img2, windowSize, numLevels):
    # Compute the Gaussian pyramid for Img1 and Img2
    pyramid1 = [Img1]
    pyramid2 = [Img2]
    for i in range(numLevels-1):
        pyramid1.append(cv.pyrDown(pyramid1[-1]))
        pyramid2.append(cv.pyrDown(pyramid2[-1]))
    
    # Compute the optical flow for the coarsest level
    u_prev = np.zeros_like(pyramid1[-1])
    v_prev = np.zeros_like(pyramid1[-1])
    for i in range(windowSize//2, pyramid1[-1].shape[0]-windowSize//2):
        for j in range(windowSize//2, pyramid1[-1].shape[1]-windowSize//2):
            u,v = LucasKanade(pyramid1[-1], pyramid2[-1], windowSize, i, j)
            u_prev[i,j] = u
            v_prev[i,j] = v
    
    # Refine the optical flow at each level of the pyramid
    for level in range(numLevels-2, -1, -1):
        # Upscale the previous level's optical flow
        u_prev = cv.pyrUp(u_prev)
        v_prev = cv.pyrUp(v_prev)
        u_prev = 2*u_prev
        v_prev = 2*v_prev
        
        # Scale the images for the current level
        cur_img1 = pyramid1[level]
        cur_img2 = pyramid2[level]
        u_cur = np.zeros_like(cur_img1)
        v_cur = np.zeros_like(cur_img1)
        
        # Refine the optical flow using the previous level's flow as the initial estimate
        for i in range(windowSize//2, cur_img1.shape[0]-windowSize//2):
            for j in range(windowSize//2, cur_img1.shape[1]-windowSize//2):
                i_up = int(round(i/2))
                j_up = int(round(j/2))
                u0 = u_prev[i_up, j_up]
                v0 = v_prev[i_up, j_up]
                u,v = OpticalFlowRefine(cur_img1, cur_img2, windowSize, u0, v0, i, j)
                u_cur[i,j] = u
                v_cur[i,j] = v
        
        # Set the refined optical flow for the current level as the previous level
        u_prev = u_cur
        v_prev = v_cur
    
    return u_prev, v_prev

# _______________________________________________________________________________________________________

import cv2 as cv
import numpy as np

def lucaskanade_optflow(frame1, frame2, window_sz, num_features, u0, v0, t=0.02):
    sz = window_sz // 2
    Ix, Iy = cv.Sobel(frame1, cv.CV_32F, 1, 0, ksize=3), cv.Sobel(frame1, cv.CV_32F, 0, 1, ksize=3)
    It = np.abs(cv.GaussianBlur(frame2, (3,3), 0) - cv.GaussianBlur(frame1, (3,3), 0))
    u, v = np.zeros(frame1.shape), np.zeros(frame1.shape)
    for i in range(num_features):
        x, y = np.random.randint(sz, frame1.shape[0]-sz-1), np.random.randint(sz, frame1.shape[1]-sz-1)
        A = np.stack((Ix[y-sz:y+sz+1, x-sz:x+sz+1].flatten(), Iy[y-sz:y+sz+1, x-sz:x+sz+1].flatten()), axis=0)
        b = -It[y-sz:y+sz+1, x-sz:x+sz+1].flatten()
        delta = np.linalg.inv(A.dot(A.T)).dot(A).dot(b)
        u[y, x] = delta[0] + u0[y, x]
        v[y, x] = delta[1] + v0[y, x]
    return u, v

def optical_flow_refine(frame1, frame2, window_sz, u0, v0):
    return lucaskanade_optflow(frame1, frame2, window_sz, 1, u0, v0)

# def MultiScaleLucasKanade(Img1, Img2, window_sz, num_levels):
#     u, v = np.zeros(Img1.shape), np.zeros(Img1.shape)
#     for level in range(num_levels, 0, -1):
#         scale = 2**(1-level)
#         Img1_scaled = cv.resize(Img1, (0,0), fx=scale, fy=scale, interpolation=cv.INTER_AREA)
#         Img2_scaled = cv.resize(Img2, (0,0), fx=scale, fy=scale, interpolation=cv.INTER_AREA)
#         u_scaled, v_scaled = lucaskanade_optflow(Img1_scaled, Img2_scaled, window_sz, 1, u, v, t=0.02)
#         u, v = cv.resize(u_scaled, (Img1.shape[

def MultiScaleLucasKanade(Img1, Img2, window_sz, num_levels):
    u, v = np.zeros(Img1.shape), np.zeros(Img1.shape)
    for level in range(num_levels, 0, -1):
        scale = 2**(1-level)
        Img1_scaled = cv.resize(Img1, (0,0), fx=scale, fy=scale, interpolation=cv.INTER_AREA)
        Img2_scaled = cv.resize(Img2, (0,0), fx=scale, fy=scale, interpolation=cv.INTER_AREA)
        u_scaled, v_scaled = lucaskanade_optflow(Img1_scaled, Img2_scaled, window_sz, 1, u/scale, v/scale, t=0.02)
        u, v = cv.resize(u_scaled*scale, (Img1.shape[1], Img1.shape[0])), cv.resize(v_scaled*scale, (Img1.shape[1], Img1.shape[0]))
    return u, v

# The function first initializes the optical flow vectors u and v with zeros of the same shape as the input images. 
# Then, it iterates over the number of levels specified in num_levels in descending order using a for loop. For each level,
#  the images are downscaled to the appropriate level using cv.resize function with interpolation method set to cv.INTER_AREA. 
# The optical flow at the current level is then computed using the lucaskanade_optflow function with initial estimates of u and v scaled down by the appropriate factor. 
# Finally, the estimated optical flow is upscaled to the original image size using cv.resize and stored in u and v for the next level of the pyramid. The function returns the final estimates of the optical flow u and v.

# _______________________________________________________________________________________________________

def MultiScaleLucasKanade(Img1, Img2, windowSize, numLevels):
    # Define a list of scales to process the images at
    scales = [2**(1-level) for level in range(numLevels)]
    
    # Compute the Gaussian pyramid for each image
    Img1_pyramid = [Img1]
    Img2_pyramid = [Img2]
    for scale in scales[1:]:
        Img1_pyramid.append(cv.pyrDown(Img1_pyramid[-1], dstsize=(int(scale*Img1.shape[1]), int(scale*Img1.shape[0]))))
        Img2_pyramid.append(cv.pyrDown(Img2_pyramid[-1], dstsize=(int(scale*Img2.shape[1]), int(scale*Img2.shape[0]))))
    
    # Compute optical flow at the coarsest level
    u0 = np.zeros_like(Img1_pyramid[-1])
    v0 = np.zeros_like(Img1_pyramid[-1])
    u, v = OpticalFlowRefine(Img1_pyramid[-1], Img2_pyramid[-1], windowSize, u0, v0)
    for level in range(numLevels-1):
        # Upscale optical flow from previous level
        u = cv.pyrUp(u, dstsize=(Img1_pyramid[level].shape[1], Img1_pyramid[level].shape[0]))
        v = cv.pyrUp(v, dstsize=(Img1_pyramid[level].shape[1], Img1_pyramid[level].shape[0]))
        
        # Refine optical flow at this level
        u, v = OpticalFlowRefine(Img1_pyramid[level], Img2_pyramid[level], windowSize, u, v)
        
    return u, v

# This version of the function also uses Gaussian pyramid to compute optical flow at multiple scales, 
# but it does it in a slightly different way. It first computes the pyramid for both images 
# and then processes them level-by-level, using cv.pyrUp to upsample the optical flow from the previous level 
# and OpticalFlowRefine to refine the optical flow at the current level. The function then returns the final optical flow u and v.