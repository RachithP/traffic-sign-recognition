
import glob
import cv2
import imutils
import pickle

def showImage(image):
	cv2.namedWindow("Display frame",cv2.WINDOW_NORMAL)
	cv2.imshow("Display frame",image)
	k = cv2.waitKey(0)
	return k

def pyramid(image, scale=1.5, minSize=(640, 640)):
	# yield the original image
	yield image
 
	# keep looping over the pyramid
	while True:
		print 'k'
		# compute the new dimensions of the image and resize it
		w = int(image.shape[1] / scale)
		image = imutils.resize(image, width=w)
 
		# if the resized image does not meet the supplied minimum
		# size, then stop constructing the pyramid
		if image.shape[0] < minSize[1] or image.shape[1] < minSize[0]:
			break
 
		# yield the next image in the pyramid
		yield image
	
def sliding_window(image, stepSize, windowSize):
	# slide a window across the image
	for y in range(0, image.shape[0]-windowSize[0], stepSize):
		for x in range(0, image.shape[1]-windowSize[0], stepSize):
			# yield the current window
			yield (x,y,image[y:y + windowSize[1], x:x + windowSize[0]])

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
filename = 'eight_class_svm.sav'
loaded_model = pickle.load(open(filename, 'rb'))
# load training image
trainImage = cv2.imread('P.png')

# resize
trainImage = cv2.resize(trainImage, (64, 64), interpolation=cv2.INTER_AREA)
# convert to grayscale
trainImage = cv2.cvtColor(trainImage, cv2.COLOR_BGR2GRAY)
fd = hog.compute(trainImage)
fd = fd.T
y_pred = loaded_model.predict(fd)

print y_pred
# TRAINING_IMAGE_SIZE_X = 64
# TRAINING_IMAGE_SIZE_Y = 64



# # Iterate through each image, convert to BGR, undistort(function takes all channels in input)
# pathToImages = 'input'
# imagePaths = glob.glob(pathToImages+"/*.jpg")
# imagePaths.sort()
# for imageIndex in range(80,len(imagePaths)):
# 	print imageIndex
# 	imagePath = imagePaths[imageIndex]
# 	image = cv2.imread(imagePath)
# 	scaledImages = pyramid(image)
# 	for scaledImage in scaledImages:
# 		showImage(scaledImage)
# 		windows = sliding_window(scaledImage,25,(64,64))
# 		for windowinfo in windows:
# 			window = windowinfo[2]
# 			print window.shape
# 			# resize
# 			trainImage = cv2.resize(trainImage, (TRAINING_IMAGE_SIZE_X, TRAINING_IMAGE_SIZE_Y), interpolation=cv2.INTER_AREA)
# 			# convert to grayscale
# 			trainImage = cv2.cvtColor(window, cv2.COLOR_BGR2GRAY)
# 			fd = hog.compute(trainImage)
# 			fd = fd.T
# 			y_pred = loaded_model.predict(fd)
# 			print y_pred[0]
# 			if y_pred[0]==1:
# 				showImage(window)
# 	quit()