# !/usr/bin/env python

'''
ENPM 673 Spring 2019: Robot Perception
Project 3 GMM Bouy Detection

Author:
Ashwin Varghese Kuruttukulam(ashwinvk94@gmail.com)
Rachith Prakash
Graduate Students in Robotics,
University of Maryland, College Park



This file is a user interface to create negative images for training
Note: It currently only creates images of one size. We should actua;ly create iamges in the
size range we using in the moving windows scaling and create the negative images
'''

import glob
import random
import cv2

def showImage(image):
	cv2.namedWindow("Display frame", cv2.WINDOW_NORMAL)
	cv2.imshow("Display frame", image)
	keyPress = cv2.waitKey(0)

	return keyPress

TRAINING_IMAGE_SIZE_X = 80
TRAINING_IMAGE_SIZE_Y = 80

# Iterate through each image
pathToImages = 'input'
files = glob.glob(pathToImages+"/*.jpg")

counter = 0

print 'If the image is fine press enter, else press spacebar'
# Iterate through iamges in each fodlers to compile on list of traffic sign images\
for imagePath in files:
	image = cv2.imread(imagePath)

	# convert to grayscale
	trainImage = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
	
	# Craet random integer between and including both these numbers
	randY = random.randint(0, 1236-TRAINING_IMAGE_SIZE_Y-700-1)
	randX = random.randint(0, 1628-TRAINING_IMAGE_SIZE_X-1)

	croppedNegativeImage = trainImage[randY:randY+TRAINING_IMAGE_SIZE_Y,randX:randX+TRAINING_IMAGE_SIZE_X]
	key = showImage(croppedNegativeImage)
	
	# Check if that image name already exists. If not create a new name
	Flag = 0
	while Flag==0:
		try:
			fh = open('negative_images/'+str(counter)+'.jpg', 'r')
			counter += 1
		except:
			Flag = 1
	if key==13:
		cv2.imwrite('negative_images/'+str(counter)+'.jpg',croppedNegativeImage)
