# Introduction
Develop a software in Python to identify the lane boundaries in a video from a front-facing camera on a car. The video is filmed on a highway in California.

# The goals / steps of this project are the following:
1. Compute the camera calibration matrix and distortion coefficients given a set of chessboard images
2. Apply a distortion correction to raw images
3. Use color transforms, gradients, etc., to create a thresholded binary image
4. Apply a perspective transform to rectify binary image ("birds-eye view")
5. Detect lane pixels and fit to find the lane boundary
6. Determine the curvature of the lane and vehicle position with respect to center
7. Warp the detected lane boundaries back onto the original image
8. Output visual display of the lane boundaries and numerical estimation of lane curvature and vehicle position

### Camera Calibration and Image Undistortion
The code for this step is contained in the first code cell of the IPython notebook located in "Advanced Line Finding.ipynb" block 1.
First I grab all the object points and image points from the provided chessboard images using `findChessboardCorners` from cv2. With object and image points, I can use `calibrateCamera` & `undistort` function to get the camera matrix and undistorted image. 
###### Example of a distortion-corrected image☟
![Undistort Output](https://github.com/WenHsu1203/Advanced-Lane-Finding-Project/blob/master/output_images/undistort_output.png?raw=true)

### Color Transformation/ Gradient Threshold Binary Image
I used a combination of color and gradient thresholds to generate a binary image (thresholding steps at lines 15 through 88 in `util.py`).  
#### Example of a binary threshold imgage☟

![Binary Threshold](https://github.com/WenHsu1203/Advanced-Lane-Finding-Project/blob/master/output_images/binary_combo.png?raw=true)

### Perspective Transform
The code for my perspective transform includes a function called `warp(img)`, which appears in lines 71 through 95 in the file `util.py`.  The `warp(img)` function takes as inputs an image (`img`), as well as the camera matrix M computed before.  I chose to hardcode the source and destination points in the following manner:

```python
src = np.float32([[585,461],
[200,717],
[1088,704],
[708,459]])
dst = np.float32([[320,0],
[320,720],
[980,720],
[980,0]])
```

This resulted in the following source and destination points:

| Source        | Destination   | 
|:-------------:|:-------------:| 
| 585, 461      | 320, 0        | 
| 200, 717      | 320, 720      |
| 1088, 704     | 980, 720      |
| 708, 459      | 980, 0        |

I verified that my perspective transform was working as expected by drawing the `src` and `dst` points onto a test image and its warped counterpart to verify that the lines appear parallel in the warped image.

###### Example of a perspective-transformed image☟

![Perspective Transformed](https://github.com/WenHsu1203/Advanced-Lane-Finding-Project/blob/master/output_images/warped_straight_lings.png?raw=true)

### Identifying Lane-Line Pixels and Fitting with a Polynomial
The code for my identified lane-line pixels and fit their positions with a polynomial include a function called `find_lane_pixels(binary_warped)`, which appears in lines 110 through 184 and `fit_polynomial(binary_warped)`, which appears in lines 187 through 236 in the file `util.py`. In short, it divided the images into many layers and calculate the histogram. With the histogram, we can get the highest value which indicated the lane position for each layer. After combining all the points, we can draw the line using the polyline function. 
###### Example of a lane-line identification image☟

![Lane Line](https://github.com/WenHsu1203/Advanced-Lane-Finding-Project/blob/master/output_images/curve_image.png?raw=true)

### Calculate the Radius of Curvature of the Lane and the Position of the Vehicle Respect to the Center
I did this in lines 97 through 109 in my code in `util.py`. It uses the data from the funtion `find_lane_pixels` and calculate the radius formula, and it outputs both the left and right lane radius, and I print out the average of them on the final video.

### Image of the Final Result
I implemented this step in my code in block 8 in jupyter notebook file. 
#### Example of the final image☟

![final](https://github.com/WenHsu1203/Advanced-Lane-Finding-Project/blob/master/output_images/final.png?raw=true)


# Result
Check out the video ☟

[![Video](http://img.youtube.com/vi/2HLe7jOSUZs/0.jpg)](http://www.youtube.com/watch?v=2HLe7jOSUZs "Lane-Finding ")




