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
import pickle
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
winSigma = -1.
histogramNormType = 0
L2HysThreshold = 2.0000000000000001e-01
gammaCorrection = 2
nlevels = 64
SignedGradients = True
hog = cv2.HOGDescriptor(winSize,blockSize,blockStride,cellSize,nbins,derivAperture,winSigma,histogramNormType,L2HysThreshold,gammaCorrection,nlevels,SignedGradients)

TRAINING_IMAGE_SIZE_X = 64
TRAINING_IMAGE_SIZE_Y = 64

nImagesCutoff = 2200

# Iterate through each image, convert to BGR, undistort(function takes all channels in input)
pathToImages = 'Training'
trainingFolders = glob.glob(pathToImages+"/*")

# Removing readme file
try:
	trainingFolders.remove(pathToImages+'/Readme.txt')
except:
	print 'Readme file not present'

pathToImages = 'Training'
testingFolders = glob.glob(pathToImages+"/*")

# Removing readme file
try:
	testingFolders.remove(pathToImages+'/Readme.txt')
except:
	print 'Readme file not present'
folders = trainingFolders+testingFolders
trainingFeaturesArr = []
trainingLabelsArr= []

nImageCounter = 0
# Iterate through images in each fodlers to compile on list of traffic sign images
for folder in folders:
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
			trainingLabelsArr = np.append(trainingLabelsArr,1)
		if nImageCounter>=nImagesCutoff:
			break
	if nImageCounter>=nImagesCutoff:
			break

nPostiveImages = nImageCounter

nImageCounter = 0
NegativeImages1 = glob.glob("negative_images_1/*.jpg")
NegativeImages2 = glob.glob("negative_images_2/*.jpg")
NegativeImages = NegativeImages1+NegativeImages2
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
	trainingLabelsArr = np.append(trainingLabelsArr,2)
	if nImageCounter>=nImagesCutoff:
		break
print 'Number of positive',nPostiveImages
print 'Number of negative_images',nImageCounter

#Initializing svm classifier
clf = svm.SVC(gamma=0.1)
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
clf.fit(x_train,y_train)

print 'predicting.......'

y_pred = clf.predict(x_test)

print("Accuracy: "+str(accuracy_score(y_test, y_pred)))
print('\n')
print(classification_report(y_test, y_pred))

filename = 'two_class_svm.sav'
pickle.dump(clf, open(filename, 'wb'))