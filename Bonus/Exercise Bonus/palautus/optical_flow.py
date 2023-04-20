import numpy as np
import matplotlib.pyplot as plt
import cv2  
import os

#%%
os.chdir('/home/tuomas/Python/DATA.ML.300/Bonus/Exercise Bonus/Task 1/')
# Read in the image frames
frame_1 = cv2.imread('images/pacman1.png')
frame_2 = cv2.imread('images/pacman2.png')
frame_3 = cv2.imread('images/pacman3.png')

# Convert to RGB
frame_1 = cv2.cvtColor(frame_1, cv2.COLOR_BGR2RGB)
frame_2 = cv2.cvtColor(frame_2, cv2.COLOR_BGR2RGB)
frame_3 = cv2.cvtColor(frame_3, cv2.COLOR_BGR2RGB)


# Visualize the individual color channels
f, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(20,10))
ax1.set_title('frame 1')
ax1.imshow(frame_1)
ax2.set_title('frame 2')
ax2.imshow(frame_2)
ax3.set_title('frame 3')
ax3.imshow(frame_3)

#%%
# Find out good set of parameters for Shi-Tomasi for this specific set of images.
# The idea is the found out the values so that it detects corners of the Pac-Man,
# not some other, unrobust points.

##-your-code-starts-here-##
maxCorners = 14  # Maximum amount of corners to be detected, value between 2-20
qualityLevel = 0.1  # Parameter characterizing the minimal accepted quality of image corners. 0.1
                   # Value typically between 0.01-0.5.
minDistance = 35.0   # Minimum possible Euclidean distance between the returned corners. 35
blockSize = 3  # Size of an average block for computing a derivative covariation matrix 
                   # over each pixel neighborhood. 
##-your-code-ends-here-##

# Parameters for ShiTomasi corner detection
feature_params = dict( maxCorners = maxCorners,
                       qualityLevel = qualityLevel,
                       minDistance = minDistance,
                       blockSize = blockSize)


# Convert all frames to grayscale
gray_1 = cv2.cvtColor(frame_1, cv2.COLOR_RGB2GRAY)
gray_2 = cv2.cvtColor(frame_2, cv2.COLOR_RGB2GRAY)
gray_3 = cv2.cvtColor(frame_3, cv2.COLOR_RGB2GRAY)


# Take first frame and find corner points in it
pts_1 = cv2.goodFeaturesToTrack(gray_1, mask = None, **feature_params)

plt.figure()
# display the detected points
plt.imshow(frame_1)
for p in pts_1:
    # plot x and y detected points
    plt.plot(p[0][0], p[0][1], 'r.', markersize=15)
    plt.title("Detected points in the first image")
    
#%%
# Parameters for Lucas-Kanade optical flow
# Your task is to find out good parameter values
# The output should have lines going to the direction of a vector created by the 
# corresponding detected points in images i and i+1. You'll know it when you see it.
    
##-your-code-starts-here-##
winSize = (5, 5) # Size of the search window at each pyramid level
maxLevel = 5    # Maximal pyramid level 
##-your-code-ends-here-##

lk_params = dict( winSize  = winSize,
                  maxLevel = maxLevel,
                  criteria = (cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 10, 0.03))

# Calculate optical flow between first and second frame
pts_2, isFound, err = cv2.calcOpticalFlowPyrLK(gray_1, gray_2, pts_1, None, **lk_params)

# Select good matching points between the two image frames
good_new = pts_2[isFound==1]
good_old = pts_1[isFound==1]

# Create a mask image for drawing (u,v) vectors on top of the second frame
mask = np.zeros_like(frame_2)

# Draw the lines between the matching points (these lines indicate motion vectors)
for i,(new,old) in enumerate(zip(good_new,good_old)):
    a,b = new.ravel()
    c,d = old.ravel()
    # Draw points on the mask image
    mask = cv2.circle(mask,(a,b),8,(200),-1)
    # Draw motion vector as lines on the mask image
    mask = cv2.line(mask, (a,b),(c,d), (200), 3)
    # Add the line image and second frame together

composite_im = np.copy(frame_2)
composite_im[mask!=0] = [0]

plt.figure()
plt.imshow(composite_im)
plt.title("Optical flow between the first and the second image")

pts_1 = cv2.goodFeaturesToTrack(gray_2, mask = None, **feature_params)

plt.figure()
# Display the detected points
plt.imshow(frame_2)
for p in pts_1:
    # plot x and y detected points
    plt.plot(p[0][0], p[0][1], 'r.', markersize=15)
    plt.title("Detected points in the second image")

# Calculate optical flow between first and second frame
pts_2, match, err = cv2.calcOpticalFlowPyrLK(gray_2, gray_3, pts_1, None, **lk_params)

plt.figure()
# Select good matching points between the two image frames
good_new = pts_2[match==1]
good_old = pts_1[match==1]

# Create a mask image for drawing (u,v) vectors on top of the second frame
mask = np.zeros_like(frame_3)

# Draw the lines between the matching points (these lines indicate motion vectors)
for i,(new,old) in enumerate(zip(good_new,good_old)):
    a,b = new.ravel()
    c,d = old.ravel()
    # Draw points on the mask image
    mask = cv2.circle(mask,(a,b),8,(200),-1)
    # Draw motion vector as lines on the mask image
    mask = cv2.line(mask, (a,b),(c,d), (200), 3)
    # Add the line image and second frame together

composite_im = np.copy(frame_3)
composite_im[mask!=0] = [0]

plt.imshow(composite_im)
plt.title("Optical flow between the second and the third image")