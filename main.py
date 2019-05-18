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
import cv2
import numpy as np
import bisect
from classifiers_svm import classifier
from scipy.stats import multivariate_normal
import pickle

def showImage(image):
	cv2.namedWindow("Display frame", cv2.WINDOW_NORMAL)
	cv2.imshow("Display frame", image)
	cv2.waitKey(100)


# def denoiseMultiColor():
#
# 	cv2.fastNlMeansDenoisingColoredMulti()


def denoiseColor(img):
	return cv2.fastNlMeansDenoisingColored(img, None, 10, 10, 7, 21)


def contractStretching(img):
	clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
	return clahe.apply(img)


def imadjust(src, tol=1, vin=[0, 255], vout=(0, 255)):
	tol = tol / 2
	src = src.astype(float)
	assert len(src.shape) == 2, 'Input image should be 2-dims'
	tol = max(0, min(100, tol))

	if tol > 0:
		# Compute in and out limits
		# Histogram
		hist = np.histogram(src, bins=list(range(256)), range=(0, 255))[0]

		# Cumulative histogram
		cum = hist.copy()
		cum = np.cumsum(cum)

		# Compute bounds
		total = src.shape[0] * src.shape[1]
		low_bound = total * tol / 100
		upp_bound = total * (100 - tol) / 100
		vin[0] = bisect.bisect_left(cum, low_bound)
		vin[1] = bisect.bisect_left(cum, upp_bound)

	# Stretching
	scale = (vout[1] - vout[0]) / (vin[1] - vin[0])
	vs = src - vin[0]
	vs[vs < 0] = 0
	vs = (vs * scale) + vout[0]
	vs[vs > vout[1]] = vout[1]

	return vs.astype(np.uint8)

def classifier_init():
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

	# load the model from disk
	filename = 'correct_sign_2_class_svm.sav'
	correct_sign_model = pickle.load(open(filename, 'rb'))
	filename = 'eight_class_svm.sav'
	eight_class_model = pickle.load(open(filename, 'rb'))
	filename = 'two_class_svm.sav'
	two_class_model = pickle.load(open(filename, 'rb'))
	filename = 'nine_class_svm.sav'
	nine_class_model = pickle.load(open(filename, 'rb'))

	return hog,two_class_model,correct_sign_model,eight_class_model,nine_class_model

def main():

	hog,two_class_model,correct_sign_model,eight_class_model,nine_class_model = classifier_init()

	blue_lower = np.array([80, 90, 0], np.uint8)
	blue_upper = np.array([200, 200, 255], np.uint8)
	# red_lower = np.array([20, 20, 100], np.uint8)
	# red_upper = np.array([40, 200, 255], np.uint8)

	path_to_images = 'denoised_images'
	files = glob.glob(path_to_images + "/*.jpg")
	files.sort()
	path_to_images_hsv = 'denoised_images_hsv'
	files_hsv = glob.glob(path_to_images_hsv + "/*.jpg")
	files_hsv.sort()

	for index in range(len(files)):

		image_denoise = cv2.imread(path_to_images+"/frame"+str(index)+".jpg", -1)  # read image

		image_denoise_hsv = cv2.imread(path_to_images_hsv+"/frame"+str(index)+".jpg", -1)

		blue_hsv = cv2.inRange(image_denoise_hsv, blue_lower, blue_upper)

		# showImage(np.hstack((image_denoise, image_denoise_hsv)))
		# continue

		b_, g_, r_ = cv2.split(image_denoise)  # split into channels

		# b = contractStretching(b_).astype(float)  # adaptive histogram equivalent on each channel
		# g = contractStretching(g_).astype(float)
		# r = contractStretching(r_).astype(float)

		b = imadjust(b_, tol=60).astype(float)
		g = imadjust(g_, tol=60).astype(float)
		r = imadjust(r_, tol=60).astype(float)

		# showImage(np.hstack((b.astype(np.uint8), g.astype(np.uint8), r.astype(np.uint8))))
		# continue
		den = r + g + b  # calculate the denominator

		red_image = np.maximum(0.0, np.minimum(r - b, r - g) / den)
		red_max = np.nanmax(red_image)
		red_image = 255 * red_image / red_max
		red_image = red_image.astype(np.uint8)

		blue_image = np.maximum(0.0, (b - r) / den)
		blue_max = np.nanmax(blue_image)
		blue_image = 255 * blue_image / blue_max
		blue_image = blue_image.astype(np.uint8)

		blue_image = contractStretching(blue_image)
		blue_thresh = cv2.threshold(blue_image, 160, 255, cv2.THRESH_BINARY)[1]
		red_image = contractStretching(red_image)
		red_thresh = cv2.threshold(red_image, 80, 255, cv2.THRESH_BINARY)[1]

		# -------------------------------------------------------
		blue_image = np.maximum(blue_hsv, blue_thresh)
		# showImage(np.hstack((blue_image, red_thresh)))
		# continue
		final_image = image_denoise.copy()

		_, red_contours, _ = cv2.findContours(red_thresh, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
		# for cnt in red_contours:
		# 	x, y, w, h = cv2.boundingRect(cnt)
		# 	cv2.rectangle(image_denoise, (x, y), (x + w, y + h), (0, 0, 255), 2)
		# 	if 150.0 < cv2.contourArea(cnt) < 5000.0:
		# 		x, y, w, h = cv2.boundingRect(cnt)
		# 		cv2.rectangle(image_denoise, (x, y), (x + w, y + h), (0, 255, 255), 2)
		# 		x, y, w, h = cv2.boundingRect(cnt)
		# 		if 0.3 < float(h) / w < 3.3:
		# 			cv2.rectangle(final_image, (x, y), (x + w, y + h), (200, 0, 200), 2)
					

		_, blue_contours, _ = cv2.findContours(blue_image, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
		for cnt in blue_contours:
			x, y, w, h = cv2.boundingRect(cnt)
			cv2.rectangle(image_denoise, (x, y), (x + w, y + h), (255, 0, 0), 2)
			if 150.0 < cv2.contourArea(cnt) < 5000.0:
				x, y, w, h = cv2.boundingRect(cnt)
				cv2.rectangle(image_denoise, (x, y), (x + w, y + h), (255, 255, 0), 2)
				x, y, w, h = cv2.boundingRect(cnt)
				if 0.3 < float(h) / w < 3.3:
					cv2.rectangle(final_image, (x, y), (x + w, y + h), (255, 100, 100), 2)
					croppedWindow = final_image[y:y+h,x:x+w]
					prediction = classifier(croppedWindow,hog,two_class_model)
					cv2.putText(final_image,str(prediction), (x, y),cv2.FONT_HERSHEY_SIMPLEX,1,(255,255,255),2)

		cv2.imwrite("outputs/testing/frame"+str(index)+".jpg", image_denoise)
		cv2.imwrite("outputs/testing/final/frame"+str(index)+".jpg", final_image)
		showImage(image_denoise)
		continue

		# # -------------------------------------------------------
		# # test_image = np.maximum(blue_hsv, blue_thresh)
		# # test_image = np.maximum(red_thresh, test_image)
		# # test_image = np.maximum(blue_image, red_image) # this is for MSER
		# # showImage(test_image)
		# # continue
		# # showImage(np.hstack((cv2.threshold(red_image, 50, 255, cv2.THRESH_BINARY)[1], cv2.threshold(blue_image, 180, 255, cv2.THRESH_BINARY)[1])))
		# # continue

		# # test_image = cv2.threshold(test_image, 50, 200, cv2.THRESH_BINARY)[1]

		# mser = cv2.MSER_create(_delta=4, _max_variation=0.3, _min_diversity=0.5, _max_area=4000, _min_area=200)
		# # mser = cv2.MSER_create()
		# # test_image = red_image+blue_image
		# regions, _ = mser.detectRegions(test_image)

		# for p in regions:
		# 	x, y, w, h = cv2.boundingRect(p.reshape(-1, 1, 2))
		# 	if h / w >= 0.6 and h / w <= 1.3:
		# 		cv2.rectangle(image, (x, y), (x + w, y + h), (0, 255, 0), 1)

		# # hulls = [cv2.convexHull(p.reshape(-1, 1, 2)) for p in regions]
		# # cv2.polylines(image, hulls, 1, (0, 255, 0))

		# showImage(image)


if __name__ == "__main__":
	main()
