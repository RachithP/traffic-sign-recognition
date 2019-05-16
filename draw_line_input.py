# !/usr/bin/env python

'''
ENPM 673 Spring 2019: Robot Perception
Project 3 GMM Bouy Detection

Author:
Ashwin Varghese Kuruttukulam(ashwinvk94@gmail.com)
Rachith Prakash
Graduate Students in Robotics,
University of Maryland, College Park
'''

import glob
import random
import cv2

def showImage(image):
	cv2.namedWindow("Display frame",cv2.WINDOW_NORMAL)
	cv2.imshow("Display frame",image)
	keyPress = cv2.waitKey(0)

	return keyPress

def showImage(image):
	cv2.namedWindow("Display frame",cv2.WINDOW_NORMAL)
	cv2.imshow("Display frame",image)
	cv2.waitKey(0)

TRAINING_IMAGE_SIZE_X = 80
TRAINING_IMAGE_SIZE_Y = 80

# Iterate through each image, convert to BGR, undistort(function takes all channels in input)
pathToImages = 'input'
files = sorted(glob.glob(pathToImages+"/*.jpg"))
counter = 0

# Iterate through iamges in each fodlers to compile on list of traffic sign images\
for imagePath in files:
	image = cv2.imread(imagePath)
	cv2.line(image,(0,700),(1628,700),(255,0,0),5)
	cv2.imwrite('input_line/'+str(counter)+'.jpg',image)
	counter +=1
	print counter
