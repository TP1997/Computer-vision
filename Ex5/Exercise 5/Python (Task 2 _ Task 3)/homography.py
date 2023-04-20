import cv2
import numpy as np
import os

os.chdir('/home/tuomas/Python/DATA.ML.300/Ex5/Exercise 5/Python (Task 2 & Task 3)')
img = cv2.imread("reference.jpg", cv2.IMREAD_GRAYSCALE)  # query image
test_img = cv2.imread("image.jpg") # test image

# Features
sift = cv2.SIFT_create()
keyp_image, descrip_image = sift.detectAndCompute(img, None)

# Feature matching
index_params = dict(algorithm=0, trees=5)
search_params = dict()
flann = cv2.FlannBasedMatcher(index_params, search_params)

#%%
# Convert the  test image to grayscale using proper cv2-function.
# After that calculate the keypoints and descriptors with SIFT.
# Then calculate the matches between both query and test image descriptors
# with already declared flann using knnMatch-function (k = 2). 
# Store the matches to "matches"-variable.

##--your-code-starts-here--##
grayframe = cv2.cvtColor(test_img, cv2.COLOR_BGR2GRAY)
keyp_grayframe, descrip_keyframe = sift.detectAndCompute(grayframe,None)
matches = flann.knnMatch(descrip_image, descrip_keyframe, k=2)
##--your-code-ends-here--##

#%%
good_points = []
thresh = 0.6

for m, n in matches:
    if m.distance < thresh * n.distance:
        good_points.append(m)
        
cv2.imshow("Query image", img)
cv2.waitKey()

#%%
if len(good_points) > 20:
    query_pts = np.float32([keyp_image[m.queryIdx].pt for m in good_points]).reshape(-1, 1, 2)
    test_pts = np.float32([keyp_grayframe[m.trainIdx].pt for m in good_points]).reshape(-1, 1, 2)
    
    # Calculate the homography using cv2.findHomography, look up the documentation
    # (https://docs.opencv.org/master/d9/d0c/group__calib3d.html)
    # for the function to see what values it takes in. Store this homography matrix to 
    # variable "matrix". Note that the function returns the mask as well and 
    # the code will throw an error if you don't store it anywhere. 
    
    ##--your-code-starts-here--##
    matrix,_ = cv2.findHomography(query_pts, test_pts)  # replace me
    ##--your-code-ends-here--##
    
    # Perspective transform
    h, w = img.shape
    pts = np.float32([[0, 0], [0, h], [w, h], [w, 0]]).reshape(-1, 1, 2)
    dst = cv2.perspectiveTransform(pts, matrix)
    homography = cv2.polylines(test_img, [np.int32(dst)], True, (255, 0, 0), 3)
    cv2.imshow("Homography", homography)    
    cv2.waitKey()
    # Warp the image using cv2.warpPerspective and the homography matrix 
    # so the target is in one to one correspondence to query image
    # in terms of perspective.
    # Use dsize = (720, 540)
    # HINT: In order to produce the opposite of what the homography does what 
    # should you do with the homography matrix?
    
    ##--your-code-starts-here--##
    im_warped = cv2.warpPerspective(img, matrix, dsize=(720,540)) # replace me
    ##--your-code-ends-here--##
    cv2.imshow("Warped image", im_warped)
    cv2.waitKey()
else:
    cv2.imshow("Homography", grayframe)
    cv2.waitKey()
