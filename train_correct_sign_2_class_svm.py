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
import pickle

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

trainingLabelShortened = [45, 21, 38, 35, 17, 1, 14, 19]
trainingFeaturesArr = []
trainingLabelsArr= []
nImageCounter = 0

# Getting images in training
pathToImages = 'Training'
foldersTraining = glob.glob(pathToImages+"/*")
# Removing readme file
try:
	foldersTraining.remove(pathToImages+'/Readme.txt')
except:
	print 'Readme file not present'
foldersTraining.sort()

# Getting images in training
pathToImages = 'Testing'
foldersTesting = glob.glob(pathToImages+"/*")
# Removing readme file
try:
	foldersTesting.remove(pathToImages+'/Readme.txt')
except:
	print 'Readme file not present'
foldersTesting.sort()
correctN = 1 
incorrectN = 1 
trainingLabelN = 1 
# Iterate through images in each fodlers to compile on list of traffic sign images
for label,folder in enumerate(foldersTraining):
	if label not in trainingLabelShortened:
		trainingLabel = 2
		if incorrectN>1800:
			continue
	else:
		trainingLabel = 1
	folderTesting = foldersTesting[label]
	print 'label is:',label
	nImageCounter1 = 0
	PositiveImages = glob.glob(folder+"/*.ppm") + glob.glob(folderTesting+"/*.ppm")
	print 'Training and testing folders are'
	print folder
	print folderTesting
	print 'Total number of images in this label is:'
	print len(PositiveImages)
	for imagePath in PositiveImages:
		if trainingLabel == 2:
			incorrectN += 1 
		else:
			correctN += 1

		nImageCounter += 1
		nImageCounter1 += 1

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
			trainingLabelsArr = np.array(trainingLabel)
		else:
			trainingFeaturesArr = np.vstack((trainingFeaturesArr,fd))
			trainingLabelsArr = np.append(trainingLabelsArr,trainingLabel)
	print 'Number of images is:',nImageCounter1

print incorrectN,correctN
#Initializing svm classifier
# model = svm.SVC(C=1.0, kernel='rbf', degree=3, gamma='auto_deprecated', coef0=0.0, shrinking=True, probability=False, tol=0.001, cache_size=200, class_weight='balanced')
model = svm.SVC(gamma=0.01)
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

y_pred = model.predict(x_test)

# y_pred = model.predict(x_test)
print("Accuracy: "+str(accuracy_score(y_test, y_pred)))
print('\n')
print(classification_report(y_test, y_pred))

filename = 'correct_sign_2_class_svm.sav'
pickle.dump(model, open(filename, 'wb'))