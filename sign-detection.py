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
from scipy.stats import multivariate_normal


def showImage(image):
	cv2.namedWindow("Display frame", cv2.WINDOW_NORMAL)
	cv2.imshow("Display frame", image)
	cv2.waitKey(1)


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


def main():
	# Iterate through each image in order
	pathToImages = 'input'
	files = glob.glob(pathToImages + "/*.jpg")
	files.sort()

	count = 0
	# --------------testing---------------------------------------
	# image = cv2.imread('test.png', 0)
	# showImage(np.hstack((imadjust(image, tol=0),image)))
	# quit()
	# ------------------------------------------------------------

	blue_lower = np.array([80, 90, 0], np.uint8)
	blue_upper = np.array([200, 200, 255], np.uint8)
	# red_lower = np.array([20, 20, 100], np.uint8)
	# red_upper = np.array([40, 200, 255], np.uint8)

	path_to_images = 'denoised_images'
	files = glob.glob(pathToImages + "/*.jpg")
	files.sort()
	path_to_images_hsv = 'denoised_images_hsv'
	files_hsv = glob.glob(pathToImages + "/*.jpg")
	files_hsv.sort()

	for imagePath in files[915:]:

		image = cv2.imread(imagePath, -1)  # read image

		image_hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)

		image_denoise_hsv = denoiseColor(image_hsv[:700, :])  # denoise image

		# var = multivariate_normal(mean=[133.16, 71.916, 41.41],
		# 						  cov=[[225.60, 77.469, 4.742], [77.469, 37.356, 7.856], [4.742, 7.856, 10.08]])
		#
		# image = var.pdf(image)
		#
		# thresh = cv2.threshold(image, 1e-12, 255, cv2.THRESH_BINARY)[1]

		blue_hsv = cv2.inRange(image_denoise_hsv, blue_lower, blue_upper)
		# red_hsv = cv2.inRange(image_denoise_hsv, red_lower, red_upper)

		# showImage(image)
		# cv2.imshow('image', red_hsv)
		# continue



		image_denoise = denoiseColor(image[:700, :])  # denoise image
		b_, g_, r_ = cv2.split(image_denoise)  # split into channels

		b = contractStretching(b_).astype(float)  # adaptive histogram equivalent on each channel
		g = contractStretching(g_).astype(float)
		r = contractStretching(r_).astype(float)

		# b = imadjust(b_, tol=60).astype(float)
		# g = imadjust(g_, tol=60).astype(float)
		# r = imadjust(r_, tol=60).astype(float)

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

#-------------------------------------------------------
		test_image = np.maximum(blue_hsv, blue_thresh)
		test_image = np.maximum(red_thresh, test_image)
		# showImage(test_image)
		# continue
		_, contours, _ = cv2.findContours(test_image,cv2.RETR_TREE,cv2.CHAIN_APPROX_SIMPLE)
		for cnt in contours:
			x, y, w, h = cv2.boundingRect(cnt)
			cv2.rectangle(image, (x, y), (x + w, y + h), (255, 255, 0), 1)
			if cv2.contourArea(cnt)>100 and cv2.contourArea(cnt)<6000:
				x, y, w, h = cv2.boundingRect(cnt)
				cv2.rectangle(image, (x, y), (x + w, y + h), (255, 0, 255), 2)
				# epsilon = 0.1 * cv2.arcLength(cnt, True)
				# approx = cv2.approxPolyDP(cnt, epsilon, True)
				# x, y, w, h = cv2.boundingRect(cnt)
				# if h / w >= 0.6 and h / w <= 1.3:
				# 	cv2.rectangle(image, (x, y), (x + w, y + h), (0, 255, 0), 1)
		showImage(image)
		continue

#-------------------------------------------------------
		# test_image = np.maximum(blue_hsv, blue_thresh)
		# test_image = np.maximum(red_thresh, test_image)
		# test_image = np.maximum(blue_image, red_image) # this is for MSER
		# showImage(test_image)
		# continue
		# showImage(np.hstack((cv2.threshold(red_image, 50, 255, cv2.THRESH_BINARY)[1], cv2.threshold(blue_image, 180, 255, cv2.THRESH_BINARY)[1])))
		# continue

		# test_image = cv2.threshold(test_image, 50, 200, cv2.THRESH_BINARY)[1]

		mser = cv2.MSER_create(_delta=4, _max_variation=0.3, _min_diversity=0.5, _max_area=4000, _min_area=200)
		# mser = cv2.MSER_create()
		# test_image = red_image+blue_image
		regions, _ = mser.detectRegions(test_image)

		for p in regions:
			x, y, w, h = cv2.boundingRect(p.reshape(-1, 1, 2))
			if h / w >= 0.6 and h / w <= 1.3:
				cv2.rectangle(image, (x, y), (x + w, y + h), (0, 255, 0), 1)

		# hulls = [cv2.convexHull(p.reshape(-1, 1, 2)) for p in regions]
		# cv2.polylines(image, hulls, 1, (0, 255, 0))

		showImage(image)


if __name__ == "__main__":
	main()
