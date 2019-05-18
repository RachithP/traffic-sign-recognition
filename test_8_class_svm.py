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
from sklearn.metrics import classification_report,accuracy_score

def showImage(image):
	cv2.namedWindow("Display frame",cv2.WINDOW_NORMAL)
	cv2.imshow("Display frame",image)
	cv2.waitKey(0)

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

nImagesCutoff = 1000

# load the model from disk
filename = 'model1.sav'
loaded_model = pickle.load(open(filename, 'rb'))

# Iterate through each image, convert to BGR, undistort(function takes all channels in input)
pathToImages = 'Testing'
folders = glob.glob(pathToImages+"/*")

# Removing readme file
try:
	folders.remove(pathToImages+'/Readme.txt')
except:
	print 'Readme file not present'
folders.sort()

testingFeaturesArr = []
testingLabelsArr= []
trainingLabelShortened = [45, 21, 38, 35, 17, 1, 14, 19]

nImageCounter = 0
# Iterate through images in each fodlers to compile on list of traffic sign images
for label,folder in enumerate(folders):
	if label not in trainingLabelShortened:
		continue
	PositiveImages = glob.glob(folder+"/*.ppm")
	for imagePath in PositiveImages:
		print nImageCounter
		nImageCounter += 1

		# load training image
		image = cv2.imread(imagePath)

		# convert to grayscale
		trainImage = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

		# resize
		trainImage = cv2.resize(trainImage, (TRAINING_IMAGE_SIZE_X, TRAINING_IMAGE_SIZE_Y), interpolation=cv2.INTER_AREA)
		# showImage(trainImage)

		# We have to tune these

		# fd = hog.compute(trainImage,winStride,padding,locations)
		fd = hog.compute(trainImage)
		fd = fd.T
		if nImageCounter==1:
			testingFeaturesArr = fd
			testingLabelsArr = np.array(1)
		else:
			testingFeaturesArr = np.vstack((testingFeaturesArr,fd))
			testingLabelsArr = np.append(testingLabelsArr,label)

#Initializing svm classifier
testingLabelsArr = testingLabelsArr.reshape(-1,1)
data_frame = np.hstack((testingFeaturesArr,testingLabelsArr))
np.random.shuffle(data_frame)

x_test = data_frame[:,:-1]
y_test = data_frame[:,-1:].ravel()


print x_test.shape
print y_test.shape
print 'predicting.......'
y_pred = loaded_model.predict(x_test)

# y_pred = model.predict(x_test)
print("Accuracy: "+str(accuracy_score(y_test, y_pred)))
print('\n')
print(classification_report(y_test, y_pred))

# print("Accuracy: "+str(accuracy_score(y_test, y_pred)))
# print('\n')
# print(classification_report(y_test, y_pred))

# # save the model to disk
# filename = 'model1.sav'
# pickle.dump(model, open(filename, 'wb'))


# print 'Number of training images'
# print nImageCounter