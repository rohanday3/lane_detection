"""

 Title: Lane Detection Script
 Author: Anton Elmiger
 Created: 2020-05-27

 Information: Lane Detection for different scenarios input scenarios (see constants.py)

"""

import cv2
import numpy as np
import socket
from sys import platform
import time
from signal import signal, SIGINT
from sys import exit


import lane_detection
import videoIO
import visualization
import constants as const
import kalman_filter
import optical_flow
import transform_points

# Init all classes
vid = videoIO.VideoStream()
visual = visualization.Visualization()
ld = lane_detection.Lane_Detection()
tp = transform_points.Transform_Points()




# Necessary for clean exit of script
def handler(signal_received, frame):
    # Release Gstreamer Hardware
    vid.release()
    print('CTRL+C detected. Cleaning up threads and exit')
    exit(0)
# Run handler befor aborting program with (CTRL+C)
signal(SIGINT, handler)

#of = optical_flow.OpticalFlow()
#kal = kalman_filter.KalmanFilter()

#######################################################

vid.start()                                 # start the video input

# function for edge detection
# TODO remove from main into a class
def edge_detection(img):
    gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY).astype(np.float)       # convert to gray 
    blurredGray = cv2.blur(gray, (3, 3))                                # blur img for robustness
    blurredSobelImg = cv2.Sobel(blurredGray, cv2.CV_8U, 1, 0, ksize=1)  # calculate sobel gradient
    ret, threshSobel = cv2.threshold(                                   # soebel img to binary by threshold
        blurredSobelImg, 7, 255, cv2.THRESH_BINARY)
    kernel = np.ones((3, 3), np.uint8)                                  # erode image to remove noise
    erodedSobel = cv2.erode(threshSobel, kernel, iterations=1)          # erode image to remove noise
    return erodedSobel

img, retrieved = vid.read()                 # Get the first img from video stream
img = vid.undistort(img)                # Undistort the camera img


edge_image = edge_detection(img)            # Detect Edges in image

ld.get_initial_lane_points(edge_image)      # Apply Lane initialization on first image
ld.solve_lane()                             # Solve lane model parameters from lane points
ld.lane_sanity_checks(edge_image)           # Apply corrections to lane model
tp.transform_lane_to_poly(ld)               # Calculate polynomial coeff from lane model

while True:
    if const.PRINT_TIME_PER_FRAME:
        t0 = time.time()
    ################# IMAGE LOADING ################
    if const.MODE == "camera":              
        vid.new_frame.wait()                # Wait until new frame event is set
    img, retrieved = vid.read()             # Load image from video input
    if not retrieved:
        print("End of Video")
        break

    img = vid.undistort(img)            # Undistort the camera img
    edge_image = edge_detection(img)        # Detect Edges in image


    for i in range(2):                      # Solve for lane model multiple times for convergence
        ld.lane_points(edge_image)      
        ld.solve_lane()                     # Solve lane model parameters from lane points
        ld.lane_sanity_checks(edge_image)   # Apply corrections to lane model
    tp.transform_lane_to_poly(ld)           # Calculate polynomial coeff from lane model

    if const.PRINT_TIME_PER_FRAME:
        print("Time per loop: ", np.round((time.time()-t0)*1000,1),"ms")

    ################# Visualize Results #################
    if const.VISUALIZE:
        visual.clear_imgs()                                         # Clear Images to show every iteration
        visual.draw_lane_lines(img, ld)                             # Draw lane model lines on image
        edge_image = cv2.cvtColor(edge_image, cv2.COLOR_GRAY2RGB)   # Generate Edge Image
        visual.draw_lane_points(edge_image, ld)                     # Draw lane edge points on image
        visual.append_img(img)                                      # Append img for drawing
        visual.append_img(edge_image)                               # Append img for drawing
 #       img = vid.bev(img)                                          # Birds Eye View transformation
 #       visual.append_img(img)                                      # Append img for drawing
        visual.show_imgs()                                          # Show images that were appended for drawing
    ####################################################

if const.WRITE_VIDEO:
    visual.write_video()                                             # Write video to file
