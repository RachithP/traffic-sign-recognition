

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
	