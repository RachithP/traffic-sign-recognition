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
from sklearn import svm
import pickle
import random
from sklearn.metrics import classification_report,accuracy_score

'''
Pass 1 for classification between whether the image is a sign or not, Output: 1 = sign, 2 = Not sign
Pass 2 for classification between whether the image is one of the signs we need or not, Output: 1 = yes, 2 = no
Pass 3 for classification between whether the image is a sign or not, Output: label of sign

'''

def classifier(image,hog,model):
	TRAINING_IMAGE_SIZE_X = 64
	TRAINING_IMAGE_SIZE_Y = 64

	# convert to grayscale
	trainImage = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

	# resize
	trainImage = cv2.resize(trainImage, (TRAINING_IMAGE_SIZE_X, TRAINING_IMAGE_SIZE_Y), interpolation=cv2.INTER_AREA)

	# We have to tune these

	# fd = hog.compute(trainImage,winStride,padding,locations)
	fd = hog.compute(trainImage)
	fd = fd.T
	y_pred = model.predict(fd)
	# print 'Class is :',y_pred

	return y_pred[0]


