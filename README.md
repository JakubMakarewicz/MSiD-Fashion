# MSiD-Fashion

## Introduction
The objective was to create and compare some algorithms at their ability to predict fashion-mnist data.

## Methods

### Feature Extractors
- prewitt
- canny
- sobel
- gabor


### Models
- KNearestNeighbors
- Naive Bayes
- Convolutional Neural Networks
##### Model previews:
1: Total params: 523,330

Layer (type) |  Output Shape   |  Param #
:------------- | :-------------:| :-------------:
conv2d (Conv2D) | (None, 9, 9, 32) | 320
max_pooling2d (MaxPooling2D) | (None, 4, 4, 32) | 0
dropout (Dropout) | (None, 4, 4, 32) | 0
flatten (Flatten) | (None, 512) | 0
dense (Dense) | (None, 1000) | 513000
dropout_1 (Dropout) | (None, 1000) | 0 
dense_1 (Dense) | (None, 10) | 10010 

2: Total params: 16,091,242

Layer (type) |  Output Shape   |  Param #
:------------- | :-------------:| :-------------:
input_1 (InputLayer) | [(None, 28, 28, 1) | 0
conv2d_1 (Conv2D) | (None, 28, 28, 40) | 400
conv2d_2 (Conv2D) | (None, 28, 28, 40) | 14440
conv2d_3 (Conv2D) | (None, 28, 28, 40) | 14440
flatten_1 (Flatten) | (None, 31360) | 0
dense_2 (Dense) | (None, 512) | 16056832
dense_3 (Dense) | (None, 10) | 5130

3: Total params: 40,507,842

Layer (type) |  Output Shape   |  Param #
:------------- | :-------------:| :-------------:
input_2 (InputLayer) | [(None, 28, 28, 1)] | 0
conv2d_4 (Conv2D) | (None, 28, 28, 100) | 1000  
conv2d_5 (Conv2D) | (None, 28, 28, 100) | 90100 
conv2d_6 (Conv2D) | (None, 28, 28, 100) | 90100 
conv2d_7 (Conv2D) | (None, 28, 28, 100) | 90100 
conv2d_8 (Conv2D) | (None, 28, 28, 100) | 90100 
flatten_2 (Flatten) | (None, 78400) | 0  
dense_4 (Dense) | (None, 512) | 40141312
dense_5 (Dense) | (None, 10) | 5130

4:  Total params: 81,009,642

Layer (type) |  Output Shape   |  Param #
:------------- | :-------------:| :-------------:
input_3 (InputLayer) | [(None, 28, 28, 1)] | 0         
conv2d_9 (Conv2D) | (None, 28, 28, 200) | 2000      
conv2d_10 (Conv2D) | (None, 28, 28, 200) | 360200    
conv2d_11 (Conv2D) | (None, 28, 28, 200) | 360200    
flatten_3 (Flatten) | (None, 156800) | 0         
dense_6 (Dense) | (None, 512) | 80282112  
dense_7 (Dense) | (None, 10) | 5130 

5: Total params: 91,692,242

Layer (type) |  Output Shape   |  Param #
:------------- | :-------------:| :-------------:
input_4 (InputLayer)| [(None, 28, 28, 1)]| 0         
conv2d_12 (Conv2D)| (None, 28, 28, 225)| 2250      
conv2d_13 (Conv2D)| (None, 28, 28, 225)| 455850    
conv2d_14 (Conv2D)| (None, 28, 28, 225)| 455850    
conv2d_15 (Conv2D)| (None, 28, 28, 225)| 455850    
flatten_4 (Flatten)| (None, 176400)| 0         
dense_8 (Dense)| (None, 512)| 90317312  
dense_9 (Dense)| (None, 10)| 5130

6(a variation on 4): Total params: 20,798,442

Layer (type) |  Output Shape   |  Param #
:------------- | :-------------:| :-------------:
input_1 (InputLayer) | [(None, 28, 28, 1)] | 0
conv2d (Conv2D) | (None, 28, 28, 200) | 2000
conv2d_1 (Conv2D) | (None, 28, 28, 200) | 360200
conv2d_2 (Conv2D) | (None, 28, 28, 200) | 360200
max_pooling2d (MaxPooling2D) | (None, 14, 14, 200) | 0
flatten (Flatten) | (None, 39200) | 0
dense (Dense) | (None, 512) | 20070912
dropout (Dropout) | (None, 512) | 0
dense_1 (Dense) | (None, 10) | 5130

## Results
I started off with finding the best gabor filter using naive bayes, here are the results:

gabor kernel = ksize, sigma, theta, lambda, gamma, phi

Gabor Kernel | accuracy 
:-------------:| :-------------:
3, 5, 3.92, 0.785, 0.5, 0.4 | 0.6886 
3, 3, 3.927, 0.785, 0.5, 0.4 | 0.6886
3, 5, 3.927, 0.785, 0.05, 0.4 | 0.6883
3, 3, 3.927, 0.785, 0.05, 0.4 | 0.6883
: | :
3, 5, 1.5708, 2.3562, 0.5, 0.4 | 0.4633 
3, 7, 1.5708, 2.3562, 0-05, 0-4 | 0.4607

best gabor kernel:  ksize=3 sigma=5 theta=3.9269908169872414 lambda=0.7853981633974483 gamma=0.5 phi=0.4

#### KNN and Naive Bayes

Feature Extractor | KNN, k=3/vs Benchmark | Naive Bayes/vs Benchmark
:------------- | :-------------:| :-------------:
None |  0.8541/+0.007 | 0.5856/+0.02
Prewitt_H | 0.8167/-0.03 | 0.5999/+0.036
Prewitt_V | 0.8542/+0.007 | 0.6619/+0.098
Prewitt | 0.8537/+0.007 | 0.6318/+0.068
Canny | 0.8036/-0.043 | 0.5535/-0.011
Sobel | 0.8625/+0.016 | 0.6204/+0.056
Gabor | 0.8483/+0.001 | 0.6886/+0.125

#### Neural Neworks 

Model | Accuracy | Notes
:------------- | :-------------: | :-------------:
1 | 0.8723 | None
2 | 0.9247 | None
3 | 0.9261 | None
4 | 0.9298 | None
5 | 0.9247 | None
6 | 0.9344 | Augmented data: random cropping and flipping + normalization + early stopping
6 | 0.9286 | Augmented data: random cropping, flipping and rotating + normalization + early stopping

## Usage
#### Required Modules are listed in requirements.txt

To launch the app, just run app.py. However there may be some bugs. 
Some trained models are available in release, just download and extract "TrainedModels.zip".
