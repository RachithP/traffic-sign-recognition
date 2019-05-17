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

# Iterate through each image, convert to BGR, undistort(function takes all channels in input)
path_to_images = 'Training'
folders = glob.glob(path_to_images+"/*")

# Removing readme file
try:
	folders.remove(path_to_images+'/Readme.txt')
except:
	print 'Readme file not present'

# Iterate through iamges in each fodlers to compile on list of traffic sign images\
for folder in folders:
	images = glob.glob(folder+"/*.ppm")
	print images
	break
