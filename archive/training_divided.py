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
winSigma = 4.
histogramNormType = 0
L2HysThreshold = 2.0000000000000001e-01
gammaCorrection = 0
nlevels = 64
hog = cv2.HOGDescriptor(winSize,blockSize,blockStride,cellSize,nbins,derivAperture,winSigma,histogramNormType,L2HysThreshold,gammaCorrection,nlevels)

TRAINING_IMAGE_SIZE_X = 64
TRAINING_IMAGE_SIZE_Y = 64

nImagesCutoff = 1000

# Iterate through each image, convert to BGR, undistort(function takes all channels in input)
pathToImages = 'Training'
folders = glob.glob(pathToImages+"/*")

trainingLabelShortened = [45, 21, 38, 35, 17, 1, 14, 19]

# Removing readme file
try:
	folders.remove(pathToImages+'/Readme.txt')
except:
	print 'Readme file not present'

trainingFeaturesArr = []
trainingLabelsArr= []

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
			trainingFeaturesArr = fd
			trainingLabelsArr = np.array(1)
		else:
			trainingFeaturesArr = np.vstack((trainingFeaturesArr,fd))
			trainingLabelsArr = np.append(trainingLabelsArr,label)

nImageCounter = 0
NegativeImages = glob.glob("negative_images/*.jpg")
for imagePath in NegativeImages:
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
	trainingFeaturesArr = np.vstack((trainingFeaturesArr,fd))
	trainingLabelsArr = np.append(trainingLabelsArr,label+1)
	if nImageCounter>=nImagesCutoff:
		break

#Initializing svm classifier
# model = svm.SVC(C=1.0, kernel='rbf', degree=3, gamma='auto_deprecated', coef0=0.0, shrinking=True, probability=False, tol=0.001, cache_size=200, class_weight='balanced')
model = svm.SVC(gamma=0.01,)
trainingLabelsArr = trainingLabelsArr.reshape(-1,1)
data_frame = np.hstack((trainingFeaturesArr,trainingLabelsArr))
np.random.shuffle(data_frame)

#Spilting into training and testing data
percentage = 80
partitionIndex = int(trainingLabelsArr.shape[0]*percentage/100)
print partitionIndex

x_train, x_test = data_frame[:partitionIndex,:-1],  data_frame[partitionIndex:,:-1]
y_train, y_test = data_frame[:partitionIndex,-1:].ravel() , data_frame[partitionIndex:,-1:].ravel()

print x_train.shape
print y_train.shape
print y_train
model.fit(x_train,y_train)

print 'predicting.......'

y_pred = model.predict(x_test)

print("Accuracy: "+str(accuracy_score(y_test, y_pred)))
print('\n')
print(classification_report(y_test, y_pred))

# print 'Number of training images'
# print nImageCounter