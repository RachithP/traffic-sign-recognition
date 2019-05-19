# traffic-sign-recognition

## HOG SVM Classifiers

This repositorsy contains 3 hog svm classifers. Each of them can be used using the function `classifier` `in classifiers_svm.py`

Parameters : input_image,classifier_number

classifier_number

* Pass 1 for classification between whether the image is a sign or not, Output: 1 = sign, 2 = Not sign

* Pass 2 for classification between whether the image is one of the signs we need or not, Output: 1 = yes, 2 = no

* Pass 3 for classification between whether the image is a sign or not, Output: label of sign


In order to train for 61 classes run `python train_61_class_svm.py`. This will save the model.
Then run `python main.py` to start the traffic sign detection algorithm.
