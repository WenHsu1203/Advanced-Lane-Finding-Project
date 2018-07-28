## Writeup Template

---

**Advanced Lane Finding Project**

The goals / steps of this project are the following:

* Compute the camera calibration matrix and distortion coefficients given a set of chessboard images.
* Apply a distortion correction to raw images.
* Use color transforms, gradients, etc., to create a thresholded binary image.
* Apply a perspective transform to rectify binary image ("birds-eye view").
* Detect lane pixels and fit to find the lane boundary.
* Determine the curvature of the lane and vehicle position with respect to center.
* Warp the detected lane boundaries back onto the original image.
* Output visual display of the lane boundaries and numerical estimation of lane curvature and vehicle position.

[//]: # (Image References)

[image1]: ./examples/undistort_output.png "Undistorted"
[image2]: ./test_images/test1.jpg "Road Transformed"
[image3]: ./examples/binary_combo_example.jpg "Binary Example"
[image4]: ./examples/warped_straight_lines.jpg "Warp Example"
[image5]: ./examples/color_fit_lines.jpg "Fit Visual"
[image6]: ./examples/example_output.jpg "Output"
[video1]: ./project_video.mp4 "Video"

## [Rubric](https://review.udacity.com/#!/rubrics/571/view) Points

### Here I will consider the rubric points individually and describe how I addressed each point in my implementation.  

---

### Writeup / README

#### 1. Provide a Writeup / README that includes all the rubric points and how you addressed each one.  You can submit your writeup as markdown or pdf.  [Here](https://github.com/udacity/CarND-Advanced-Lane-Lines/blob/master/writeup_template.md) is a template writeup for this project you can use as a guide and a starting point.  

Here you go!

### Camera Calibration

#### 1. Briefly state how you computed the camera matrix and distortion coefficients. Provide an example of a distortion corrected calibration image.

The code for this step is contained in the first code cell of the IPython notebook located in "./examples/example.ipynb" (or in lines # through # of the file called `some_file.py`).

First I grab all the object points and image points from the provided chessboard images using `findChessboardCorners` from cv2. With object and image points, I can use `calibrateCamera` & `undistort` function to get the camera matrix and undistorted image.

![alt text](https://goo.gl/zfkLPQ)

### Pipeline (single images)

#### 1. Provide an example of a distortion-corrected image.

To demonstrate this step, I will describe how I apply the distortion correction to one of the test images like this one:
![alt text](https://goo.gl/zfkLPQ)
I used the object points and image points to calibrate the camera matrix and distance to undistort the test images
#### 2. Describe how (and identify where in your code) you used color transforms, gradients or other methods to create a thresholded binary image.  Provide an example of a binary image result.

I used a combination of color and gradient thresholds to generate a binary image (thresholding steps at lines # through # in `util.py`).  Here's an example of my output for this step.  
![alt text](https://goo.gl/MSmkPN)

#### 3. Describe how (and identify where in your code) you performed a perspective transform and provide an example of a transformed image.

The code for my perspective transform includes a function called `warp(img)`, which appears in lines 71 through 95 in the file `util.py`.  The `warp(img)` function takes as inputs an image (`img`), as well as source (`src`) and destination (`dst`) points.  I chose the hardcode the source and destination points in the following manner:

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

![alt text](https://goo.gl/Yg5NFV)

#### 4. Describe how (and identify where in your code) you identified lane-line pixels and fit their positions with a polynomial?

The code for my identified lane-line pixels and fit their positions with a polynomial include a function called `find_lane_pixels(binary_warped)`, which appears in lines 110 through 184 and `fit_polynomial(binary_warped)`, which appears in lines 187 through 232 in the file `util.py`. In short, it divided the images into many layers and calculate the histogram. With the histogram, we can get the highest value which indicated the lane position for each layer. After combining all the points, we can draw the line using the polyline function. 

![alt text](https://goo.gl/xyqDUP)

#### 5. Describe how (and identify where in your code) you calculated the radius of curvature of the lane and the position of the vehicle with respect to center.

I did this in lines 97 through 109 in my code in `util.py`. It uses the data from the funtion `find_lane_pixels` and calculate the radius formula.

#### 6. Provide an example image of your result plotted back down onto the road such that the lane area is identified clearly.

I implemented this step in my code in block 29 in jupyter notebook file.  Here is an example of my result on a test image:

![alt text](https://goo.gl/X6EfRU)

---

### Pipeline (video)

#### 1. Provide a link to your final video output.  Your pipeline should perform reasonably well on the entire project video (wobbly lines are ok but no catastrophic failures that would cause the car to drive off the road!).

Here's a [link to my video result](https://youtu.be/toQ99iWcBC0)

---

### Discussion

#### 1. Briefly discuss any problems / issues you faced in your implementation of this project.  Where will your pipeline likely fail?  What could you do to make it more robust?
When I am processing the video, it really takes too much time since I did not use the `Search from Prior method`. Also, I apply my threshold on many different test_images and not all of the noises can be filtered out or the outputs are ideal. I think I need to spend more time on tweaking the parameters or even try more methods such as HSV or RGB to make it more robust.


