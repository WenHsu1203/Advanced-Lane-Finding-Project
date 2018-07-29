import numpy as np 
import cv2


# Calibrate the undistorted image
def cal_undistort(img, mtx,dist):
	# Convert RGB to grayscale
    img_copy = np.copy(img)
    gray = cv2.cvtColor(img_copy,cv2.COLOR_RGB2GRAY)
    # Use the cv2 undistort function to get the undistorted image
    undist = cv2.undistort(img_copy,mtx,dist,None,mtx)
    return undist

# Directional Threshold
def dir_threshold(img, sobel_kernel=3, thresh=(0, np.pi/2)):
    img_copy = np.copy(img)
	# Convert RGB to grayscale
    gray = cv2.cvtColor(img_copy,cv2.COLOR_RGB2GRAY)
    # Calculate the gradient of x and y direction
    sobelx = cv2.Sobel(gray,cv2.CV_64F,1,0,ksize = sobel_kernel)
    sobely = cv2.Sobel(gray,cv2.CV_64F,0,1,ksize = sobel_kernel)
    # Calculate the absolute value of gradients
    abs_sobelx = np.absolute(sobelx)
    abs_sobely = np.absolute(sobely)
    # Use the arctan2 function to get the directions
    dir_grad = np.arctan2(abs_sobely, abs_sobelx)
    # Apply the threshold
    binary_output = np.zeros_like(dir_grad)
    binary_output[(dir_grad >= thresh[0]) & (dir_grad <= thresh[1])] = 1
    return binary_output

# Magnitude Threshold
def mag_thresh(img, sobel_kernel=3, mag_thresh=(0, 255)):
    img_copy = np.copy(img)
	# Convert RGB to grayscale
    gray = cv2.cvtColor(img_copy,cv2.COLOR_RGB2GRAY)
    # Calculate the gradient of x and y direction
    sobelx = cv2.Sobel(gray,cv2.CV_64F,1,0,ksize=sobel_kernel)
    sobely = cv2.Sobel(gray,cv2.CV_64F,0,1,ksize=sobel_kernel)
    # Calculate the sum of absolute gradients
    abs_sobel =  np.sqrt(sobelx**2+sobely**2)
    # Scaled the abs value by dividing the max value and 
    # multiply 255, also turn it into uint8
    scaled_sobel = np.uint8(255*abs_sobel/np.max(abs_sobel))
    # Apply the threshold
    binary_output = np.zeros_like(scaled_sobel)
    binary_output[(scaled_sobel >= mag_thresh[0]) & (scaled_sobel <= mag_thresh[1])] = 1
    return binary_output

# HLS Threshold
def hls_select(img, thresh=(0, 255)):
    img_copy = np.copy(img)
	# Convert RGB to HLS
    hls = cv2.cvtColor(img_copy,cv2.COLOR_RGB2HLS)
    # Grab the S channel from HLS
    s_channel = hls[:,:,2]
    # Apply the threshold
    binary_output = np.zeros_like(s_channel) 
    binary_output[(s_channel>thresh[0]) & (s_channel<=thresh[1])] = 1
    return binary_output

def hsv_thresh(img,thres):
    img_copy = np.copy(img)
    hsv = cv2.cvtColor(img_copy,cv2.COLOR_RGB2HSV)
    h = hsv[:,:,0]
    binary_output = np.zeros_like(h)
    binary_output[(h>thres[0]) & (h<=thres[1])] = 1
    return binary_output

def luv_thresh (img, thres = (0,255)):
    img_copy = np.copy(img)
    LUV = cv2.cvtColor(img_copy, cv2.COLOR_RGB2LUV)
    L = LUV[:,:,0]
    binary_output = np.zeros_like(L)
    binary_output[(L>=thres[0])&(L<=thres[1])] = 1
    return binary_output

# Combined Threshold
def combined_select(img,kernel_size = 3, mag_thres =(0, 255), hls_thres=(0, 255),luv_thres=(0,255),hsv_thres = (0,180)):
	# Apply different kinds of threshold and combine together
    img_copy = np.copy(img) 
    mag_binary = mag_thresh(img_copy,kernel_size,mag_thres)
    hls_binary = hls_select(img_copy, hls_thres)
    luv_binary = luv_thresh(img_copy, luv_thres)
    hsv_binary = hsv_thresh(img_copy, hsv_thres)
    combined_binary = np.zeros_like(hls_binary)
    combined_binary[( ((mag_binary == 1) | (luv_binary==1) | (hsv_binary==1)) & (hls_binary == 1) )] =1 
    return combined_binary

def warp(img,M):
    img_copy = np.copy(img)
	# Prepare an image for drawing lines
    image_size = (img_copy.shape[1],img_copy.shape[0])
    # use the cv2 warpPerspective to get the warpped imag
    warpped_img = cv2.warpPerspective(img_copy,M,image_size,flags = cv2.INTER_LINEAR)
    return warpped_img

def measure_curvature_real(ploty, left_fit_cr, right_fit_cr,ym_per_pix):
    '''
    Calculates the curvature of polynomial functions in meters.
    '''
    # Define y-value where we want radius of curvature
    y_eval = np.max(ploty)
    
    # Calculation of R_curve (radius of curvature)
    left_curverad = ((1 + (2*left_fit_cr[0]*y_eval*ym_per_pix + left_fit_cr[1])**2)**1.5) / np.absolute(2*left_fit_cr[0])
    right_curverad = ((1 + (2*right_fit_cr[0]*y_eval*ym_per_pix + right_fit_cr[1])**2)**1.5) / np.absolute(2*right_fit_cr[0])
    return left_curverad, right_curverad
    
def find_lane_pixels(binary_warped):
    binary_warped_copy = np.copy(binary_warped)
    # Take a histogram of the bottom half of the image
    histogram = np.sum(binary_warped_copy[binary_warped_copy.shape[0]//2:,:], axis=0)
    # Create an output image to draw on and visualize the result
    out_img = np.dstack((binary_warped_copy, binary_warped_copy, binary_warped_copy))
    # Find the peak of the left and right halves of the histogram
    # These will be the starting point for the left and right lines
    midpoint = np.int(histogram.shape[0]//2)
    leftx_base = np.argmax(histogram[:midpoint])
    rightx_base = np.argmax(histogram[midpoint:]) + midpoint

    # Choose the number of sliding windows
    nwindows = 9
    # Set the width of the windows +/- margin
    margin = 100
    # Set minimum number of pixels found to recenter window
    minpix = 50

    # Set height of windows - based on nwindows above and image shape
    window_height = np.int(binary_warped_copy.shape[0]//nwindows)
    # Identify the x and y positions of all nonzero pixels in the image
    nonzero = binary_warped_copy.nonzero()
    nonzeroy = np.array(nonzero[0])
    nonzerox = np.array(nonzero[1])
    # Current positions to be updated later for each window in nwindows
    leftx_current = leftx_base
    rightx_current = rightx_base

    # Create empty lists to receive left and right lane pixel indices
    left_lane_inds = []
    right_lane_inds = []

    # Step through the windows one by one
    for window in range(nwindows):
        # Identify window boundaries in x and y (and right and left)
        win_y_low = binary_warped_copy.shape[0] - (window+1)*window_height
        win_y_high = binary_warped_copy.shape[0] - window*window_height
        # Find the four below boundaries of the window ###
        win_xleft_low = leftx_current - margin # Update this
        win_xleft_high = leftx_current + margin  # Update this
        win_xright_low = rightx_current - margin  # Update this
        win_xright_high = rightx_current + margin  # Update this
        
        # Identify the nonzero pixels in x and y within the window ###
        good_left_inds = ((nonzeroy >= win_y_low) & (nonzeroy < win_y_high) & 
        (nonzerox >= win_xleft_low) &  (nonzerox < win_xleft_high)).nonzero()[0]
        good_right_inds = ((nonzeroy >= win_y_low) & (nonzeroy < win_y_high) & 
        (nonzerox >= win_xright_low) &  (nonzerox < win_xright_high)).nonzero()[0]
        
        # Append these indices to the lists
        left_lane_inds.append(good_left_inds)
        right_lane_inds.append(good_right_inds)
        
        if len(good_left_inds) > minpix:
            leftx_current = np.int(np.mean(nonzerox[good_left_inds]))
        if len(good_right_inds) > minpix:        
            rightx_current = np.int(np.mean(nonzerox[good_right_inds]))


    # Concatenate the arrays of indices (previously was a list of lists of pixels)
    try:
        left_lane_inds = np.concatenate(left_lane_inds)
        right_lane_inds = np.concatenate(right_lane_inds)
    except ValueError:
        # Avoids an error if the above is not implemented fully
        pass

    # Extract left and right line pixel positions
    leftx = nonzerox[left_lane_inds]
    lefty = nonzeroy[left_lane_inds] 
    rightx = nonzerox[right_lane_inds]
    righty = nonzeroy[right_lane_inds]

    return leftx, lefty, rightx, righty, out_img


def fit_polynomial(binary_warped):
    binary_warped_copy = np.copy(binary_warped)
    
    margin = 100
    # Find our lane pixels first
    leftx, lefty, rightx, righty, out_img = find_lane_pixels(binary_warped_copy)
    # Fit a second order polynomial to each using `np.polyfit` ###
    left_fit = np.polyfit(lefty, leftx, 2)
    right_fit = np.polyfit(righty, rightx, 2)

    # Generate x and y values for plotting
    ploty = np.linspace(0, binary_warped_copy.shape[0]-1, binary_warped_copy.shape[0] )
    try:
        left_fitx = left_fit[0]*ploty**2 + left_fit[1]*ploty + left_fit[2]
        right_fitx = right_fit[0]*ploty**2 + right_fit[1]*ploty + right_fit[2]
    except TypeError:
        # Avoids an error if `left` and `right_fit` are still none or incorrect
        print('The function failed to fit a line!')
        left_fitx = 1*ploty**2 + 1*ploty
        right_fitx = 1*ploty**2 + 1*ploty
    # Calculate the offset

    ## Visualization ##
    # Colors in the left and right lane regions
    out_img[lefty, leftx] = [255, 0, 0]
    out_img[righty, rightx] = [0, 0, 255]
        
    # Create an image to draw the lines on
    warp_zero = np.zeros_like(binary_warped_copy).astype(np.uint8)
    color_warp = np.dstack((warp_zero, warp_zero, warp_zero))

    # Recast the x and y points into usable format for cv2.fillPoly()
    pts_left = np.array([np.transpose(np.vstack([left_fitx, ploty]))])
    pts_right = np.array([np.flipud(np.transpose(np.vstack([right_fitx, ploty])))])
    pts = np.hstack((pts_left, pts_right))

    # Draw the lane onto the warped blank image
    cv2.fillPoly(color_warp, np.int_([pts]), (0,255, 0))
    # Combine the result with the original image
    
    result = cv2.addWeighted(out_img, 1, color_warp, 0.3, 0)
    
    ym_per_pix = 30/720 # meters per pixel in y dimension
    xm_per_pix = 3.7/700 # meters per pixel in x dimension
    left_fit_cr = np.polyfit(ploty*ym_per_pix, left_fitx*xm_per_pix, 2)
    right_fit_cr = np.polyfit(ploty*ym_per_pix, right_fitx*xm_per_pix, 2)
    left_curverad, right_curverad = measure_curvature_real(ploty,left_fit_cr,right_fit_cr,ym_per_pix)
    curve_rad = int((left_curverad + right_curverad)/2)
    
    return result,curve_rad

# Unwarp
def unwarp(img,invM):
    img_copy = np.copy(img)
	# Calculate the image size
    image_size = (img_copy.shape[1],img_copy.shape[0])
	# Just the warpping function, but change the order of dst and src
    unwarpped_img = cv2.warpPerspective(img_copy,invM,image_size,flags = cv2.INTER_LINEAR)
    return unwarpped_img