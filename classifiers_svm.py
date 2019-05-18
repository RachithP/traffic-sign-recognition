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

def classifier(image,classifierNumber):
	# load the model from disk
	if classifierNumber==1:
		filename = 'eight_class_svm.sav'
	elif classifierNumber==2:
		filename = 'correct_sign_2_class_svm.sav'
	elif classifierNumber==3:
		filename = 'two_class_svm.sav'
	loaded_model = pickle.load(open(filename, 'rb'))
	
	winSize = (64,64)
	blockSize = (16,16)
	blockStride = (8,8)
	cellSize = (8,8)
	nbins = 16
	derivAperture = 1
	winSigma = -1.
	histogramNormType = 0
	L2HysThreshold = 2.0000000000000001e-01
	gammaCorrection = 2
	nlevels = 64
	SignedGradients = True
	hog = cv2.HOGDescriptor(winSize,blockSize,blockStride,cellSize,nbins,derivAperture,winSigma,histogramNormType,L2HysThreshold,gammaCorrection,nlevels,SignedGradients)

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
	y_pred = loaded_model.predict(fd)
	# print 'Class is :',y_pred

	return y_pred[0]


