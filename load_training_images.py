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
from skimage.feature import hog
import numpy as np
import cv2

def showImage(image):
	cv2.namedWindow("Display frame",cv2.WINDOW_NORMAL)
	cv2.imshow("Display frame",image)
	cv2.waitKey(0)

TRAINING_IMAGE_SIZE_X = 80
TRAINING_IMAGE_SIZE_Y = 80

# Iterate through each image, convert to BGR, undistort(function takes all channels in input)
pathToImages = 'Training'
folders = glob.glob(pathToImages+"/*")

# Removing readme file
try:
	folders.remove(pathToImages+'/Readme.txt')
except:
	print 'Readme file not present'

trainingFeaturesList = []
trainingLabelsList = []

nImageCounter = 0
# Iterate through images in each fodlers to compile on list of traffic sign images
for folder in folders:
	images = glob.glob(folder+"/*.ppm")
	for imagePath in images:
		nImageCounter += 1

		# # load training image
		# image = cv2.imread(imagePath)

		# # convert to grayscale
		# trainImage = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

		# # resize
		# trainImage = cv2.resize(trainImage, (TRAINING_IMAGE_SIZE_X, TRAINING_IMAGE_SIZE_Y), interpolation=cv2.INTER_AREA)

		# # We have to tune these
		# fd = hog(trainImage, orientations=8, pixels_per_cell=(TRAINING_IMAGE_SIZE_X, TRAINING_IMAGE_SIZE_Y),cells_per_block=(1, 1))
		# print fd
		# trainingFeaturesList.append(fd)
		# trainingLabelsList.append(1)
		# quit()

print 'Number of training images'
print nImageCounter
