# SSD-for-person-detection

## Dependencies
* Python 3.x
* Numpy
* TensorFlow 1.x
* Keras 2.x
* OpenCV

## Sub sampling weights
To avoid training our model from scratch, we can use the VGG16 that is pre trained on MSCOCO dataset that has 80 (+1 background) classes. The weights for this model can be downloaded [here](https://drive.google.com/file/d/1vmEF7FUsWfHquXyCqO17UaXOPpRbwsdj/view).  
Since our model requires only one class 'Person' that is already present in the MSCOCO dataset, we'll pick exactly those elements from the tensor that are responsible for the classification of 'Person'.   
There are 6 different classifier layers that contribute to 8732 predictions in the original SSD paper. We can change the dimensions of these layers so that instead of predicting confidences of 81 classes, they predict the confidence of 2 classes (including background).   
For instance the conv4_3_norm_mbox layer has a shape of (3, 3, 512, 324). The 'conv4_3_norm_mbox_loc' layer predicts 4 boxes for each spatial position, so the 'conv4_3_norm_mbox_conf' layer has to predict one of the 81 classes for each of those 4 boxes. Hence 4 x 81 = 324. However, we would only require 2 x 4 = 8 elements for the final axis.    
So out of every 81 in the 324 elements, I want to pick 2 elements. Corresponding to these elements, we will extract tensors from the original weights to our new weights file. 

## Creating Anchor Boxes
The aim is to create a large number of anchor boxes with different centres, and aspect ratios. For each of our 6 classiffier layer, a different number of anchor boxes are created. So we start from the top-left of an image, choosing a point as a center. Each point has a number of aspect ratios to create a box, after which the next point is chosen based on the value of step. For instance, the first layer for a given 8 pixel step size and 0.5 offset, will have 38 x 38 centre points and 4 boxes for each of these points. That gives us about 38x38x4 = 5776 boxes for the first layer.


## Architecture 
Since the VGG model is considered a benchmark for image classification tasks, it is no suprise models like SSD and YOLO use the VGG architecture as a backbone to extract features from the image. On top of this, SSD employs 6 additional convolutional layers that individually predict localisation and confidence scores to carry out object detection. Each of these 6 convolutional layers have different sizes so as to enable prediction of objects of varying scales in the image.
![Architecture](https://iq.opengenus.org/content/images/2021/09/Untitled--1--2.png)
