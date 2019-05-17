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
	PositiveImages = glob.glob(folder+"/*.ppm")
	for imagePath in PositiveImages:
		nImageCounter += 1

		# load training image
		image = cv2.imread(imagePath)

		# convert to grayscale
		trainImage = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

		# resize
		trainImage = cv2.resize(trainImage, (TRAINING_IMAGE_SIZE_X, TRAINING_IMAGE_SIZE_Y), interpolation=cv2.INTER_AREA)
		# showImage(trainImage)

		# We have to tune these
		fd = hog(trainImage, orientations=8, pixels_per_cell=(TRAINING_IMAGE_SIZE_X, TRAINING_IMAGE_SIZE_Y),cells_per_block=(1, 1))
		trainingFeaturesList.append(fd)
		trainingLabelsList.append(1)
	# if nImageCounter>500:
	# 	break

nImageCounter = 0
NegativeImages = glob.glob("negative_images/*.jpg")
for imagePath in NegativeImages:
	nImageCounter += 1

	# load training image
	image = cv2.imread(imagePath)

	# convert to grayscale
	trainImage = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

	# resize
	trainImage = cv2.resize(trainImage, (TRAINING_IMAGE_SIZE_X, TRAINING_IMAGE_SIZE_Y), interpolation=cv2.INTER_AREA)
	# showImage(trainImage)

	# We have to tune these
	fd = hog(trainImage, orientations=8, pixels_per_cell=(TRAINING_IMAGE_SIZE_X, TRAINING_IMAGE_SIZE_Y),cells_per_block=(1, 1))
	print fd
	trainingFeaturesList.append(fd)
	trainingLabelsList.append(2)
	# if nImageCounter>500:
	# 	break

#Initializing svm classifier
clf = svm.SVC()
trainingFeatures = np.array(trainingFeaturesList)
trainingLabels = np.array(trainingLabelsList)
trainingLabels = trainingLabels.reshape(-1,1)
data_frame = np.hstack((trainingFeatures,trainingLabels))
np.random.shuffle(data_frame)
print trainingFeatures.shape
print data_frame.shape

#Spilting into training and testing data
percentage = 80
partitionIndex = int(len(trainingFeaturesList)*percentage/100)
print partitionIndex

x_train, x_test = data_frame[:partitionIndex,:-1],  data_frame[partitionIndex:,:-1]
y_train, y_test = data_frame[:partitionIndex,-1:].ravel() , data_frame[partitionIndex:,-1:].ravel()

# y_train = y_train.reshape(-1,1)


print x_train.shape
print y_train.shape

clf.fit(x_train,y_train)

y_pred = clf.predict(x_test)

print("Accuracy: "+str(accuracy_score(y_test, y_pred)))
print('\n')
print(classification_report(y_test, y_pred))

# print 'Number of training images'
# print nImageCounter
