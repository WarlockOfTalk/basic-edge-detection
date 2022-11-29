#Module imports
from cv2 import cv2 as cv
import numpy as np
import os
import re 
import logging
import time
import cv2 as cv
from PIL import Image
import math
from numpy import testing

# #Edit out if not on pi
# from picamera.array import PiRGBArray
# from picamera import PiCamera

#Get frames images
# frames_dir = 'Photos/frames/'
# frames = os.listdir(frames_dir)

#Sort images in numerical order 
# frames.sort(key=lambda f: int(re.sub('\D', '', f)))

# #Load images 
# images = []
# for i in tqdm(frames):
#     img = cv.imread(frames_dir + i)
#     images.append(img)

#Test image
test_image = cv.imread('lanedetect_test.jpg')
test_image_edited = cv.imread('lanedetect_test_edited.jpg')
# def getTestFrame():
#     camera = PiCamera()
#     camera.resolution = (1280 , 720)
#     rawImage = PiRGBArray(camera)
#     time.sleep(1)
#     camera.capture(rawImage, format="bgr")
#     camera.close()
#     frame = rawImage.array
#     print("Image size:" + str(np.shape(frame)))
#     return frame

#Convert image colorspace and detect lane color
def edgeDetection(frame):
    #HSV colorspace
    frame_hsv = cv.cvtColor(frame, cv.COLOR_BGR2HSV)
    cv.imwrite('/home/pi/laneDetection/hsv_image.jpg', frame_hsv)
    cv.imwrite('/home/pi/laneDetection/rgb_image.jpg', frame)  
    # cv.imshow('frame hsv', frame_hsv)
    # #Required colour values
    # lower_color = np.array([60, 40, 40])
    # upper_color = np.array([150, 255, 255])

    #Testing colour values
    lower_color = np.array([80//2, 25//100*255, (30//100)*255])
    upper_color = np.array([180//2, 100//100*255, (100//100)*255])


    #Mask colour and detect edges
    mask = cv.inRange(frame_hsv, lower_color, upper_color)
    # cv.imshow('mask', mask)
    edges = cv.Canny(mask, 200, 400)
    return edges 

#Crop to region of interest
def roi(edges):
    height, width = edges.shape
    mask = np.zeros_like(edges)

    #Polygon of roi
    polygon = np.array([[(0, height * 1/2),(width, height * 1/2),(width, height),(0, height)]], np.int32)

    #Create cropped image
    cv.fillPoly(mask, polygon, 255)
    image_roi = cv.bitwise_and(edges, mask)
    return image_roi

#Detect Lines with Hough transform
def detectLineSegments(image_roi):
    rho = 1  #Precision
    theta = np.pi / 180  #Angular precision
    minThreshold = 25 #Number of votes to be considered line

    #Create lines
    line_segments = cv.HoughLinesP(image_roi, rho, theta, minThreshold, np.array([]), minLineLength=8, maxLineGap=4)
    return line_segments
    
def averageSlopeIntercept(frame, line_segments):
    """
    This function combines line segments into one or two lane lines
    If all line slopes are < 0: then we only have detected left lane
    If all line slopes are > 0: then we only have detected right lane
    """
    lane_lines = []
    if line_segments is None:
        logging.info('No line_segment segments detected')
        return lane_lines

    height, width, _ = frame.shape
    left_fit = []
    right_fit = []

    boundary = 1/3
    left_region_boundary = width * (1 - boundary)  # left lane line segment should be on left 2/3 of the screen
    right_region_boundary = width * boundary # right lane line segment should be on left 2/3 of the screen

    for line_segment in line_segments:
        for x1, y1, x2, y2 in line_segment:
            if x1 == x2:
                # logging.info('skipping vertical line segment (slope=inf): %s' % line_segment)
                continue
            fit = np.polyfit((x1, x2), (y1, y2), 1)
            slope = fit[0]
            intercept = fit[1]
            if slope < 0:
                if x1 < left_region_boundary and x2 < left_region_boundary:
                    left_fit.append((slope, intercept))
            else:
                if x1 > right_region_boundary and x2 > right_region_boundary:
                    right_fit.append((slope, intercept))

    left_fit_average = np.average(left_fit, axis=0)
    if len(left_fit) > 0:
        lane_lines.append(makePoints(frame, left_fit_average))

    right_fit_average = np.average(right_fit, axis=0)
    if len(right_fit) > 0:
        lane_lines.append(makePoints(frame, right_fit_average))

    # logging.debug('lane lines: %s' % lane_lines)  # [[[316, 720, 484, 432]], [[1009, 720, 718, 432]]]

    return lane_lines

def makePoints(frame, line):
    height, width, _ = frame.shape
    slope, intercept = line
    y1 = height  # bottom of the frame
    y2 = int(y1 * 1 / 2)  # make points from middle of the frame down

    # bound the coordinates within the frame
    x1 = max(-width, min(2 * width, int((y1 - intercept) / slope)))
    x2 = max(-width, min(2 * width, int((y2 - intercept) / slope)))
    return [[x1, y1, x2, y2]]

#Main fuction runner 
def detectLane(frame):
    edges = edgeDetection(frame)
    cropped_edges = roi(edges)
    line_segments = detectLineSegments(cropped_edges)
    lane_lines = averageSlopeIntercept(frame, line_segments)
    
    return lane_lines

#Display lines onto frame
def displayLines(frame, lines, line_color=(0, 0, 250), line_width=10):
    line_image = np.zeros_like(frame)
    if lines is not None:
        for line in lines:
            for x1, y1, x2, y2 in line:
                cv.line(line_image, (x1, y1), (x2, y2), line_color, line_width)
        line_image = cv.addWeighted(frame, 0.7, line_image, 1, 1)
    return line_image

def displayMiddleLine(frame, steering_angle, line_color=(0, 255, 0), line_width=10 ):
    heading_image = np.zeros_like(frame)
    height, width, _ = frame.shape

    steering_angle_radian = steering_angle / 180.0 * math.pi
    x1 = int(width / 2)
    y1 = height
    x2 = int(x1 - height / 2 / math.tan(steering_angle_radian))
    y2 = int(height / 2)

    cv.line(heading_image, (x1, y1), (x2, y2), line_color, line_width)
    heading_image = cv.addWeighted(frame, 0.7, heading_image, 1, 1)

    return heading_image

def runOnTestImage():
    # frame = getTestFrame()
    frame = test_image_edited
    cv.imshow('frame', frame)
    lane_lines = detectLane(frame)
    print(lane_lines)
    lane_lines_image = displayLines(test_image, lane_lines)
    lane_lines_image = displayMiddleLine(lane_lines_image, 70)
    cv.imshow('lane lines', lane_lines_image)
    return   

# runOnTestImage()

# Video capture objects
cap = cv.VideoCapture(0)

#Define the codec and create VideoWriter object
fourcc = cv.VideoWriter_fourcc(*'XVID')
out = cv.VideoWriter('presentation_video_take_1.avi',fourcc, 20.0, (640,480))

while(cap.isOpened()):
    #Grab frame from capture object
    ret, frame = cap.read()
        

    # Run Lane Detection Functions
    lane_lines = detectLane(frame)
    lane_lines_image = displayLines(frame, lane_lines)
    lane_lines_image = displayMiddleLine(lane_lines_image, 90)

    #Display the lane dtected frame
    # cv.imshow('blank frame', frame)
    cv.imshow('lane lines', lane_lines_image)

    # # Write frame to video file for recording
    out.write(lane_lines_image)

    #Quit
    if cv.waitKey(1) & 0xFF == ord('q'):
        break

#Release Capture object and close video stream
cap.release()
cv.destroyAllWindows()

# cv.waitKey(0)
