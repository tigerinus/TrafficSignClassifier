**Traffic Sign Recognition** 

---

**Build a Traffic Sign Recognition Project**

The goals / steps of this project are the following:
* Load the data set (see below for links to the project data set)
* Explore, summarize and visualize the data set
* Design, train and test a model architecture
* Use the model to make predictions on new images
* Analyze the softmax probabilities of the new images
* Summarize the results with a written report


[//]: # (Image References)

[image1]: ./examples/visualization.png "Visualization"
[image2]: ./examples/grayscale.jpg "Grayscaling"
[image3]: ./examples/random_noise.jpg "Random Noise"
[image4]: ./examples/test0.png "Traffic Sign 1"
[image5]: ./examples/test1.png "Traffic Sign 2"
[image6]: ./examples/test17.png "Traffic Sign 3"
[image7]: ./examples/test2.png "Traffic Sign 4"
[image8]: ./examples/test22.png "Traffic Sign 5"
[image9]: ./examples/test25.png "Traffic Sign 6"
[image10]: ./examples/test3.png "Traffic Sign 7"
[image11]: ./examples/test35.png "Traffic Sign 8"

## Rubric Points
### Here I will consider the [rubric points](https://review.udacity.com/#!/rubrics/481/view) individually and describe how I addressed each point in my implementation.  

---
### Writeup / README

#### 0. Provide a Writeup / README that includes all the rubric points and how you addressed each one. You can submit your writeup as markdown or pdf. You can use this template as a guide for writing the report. The submission includes the project code.

You're reading it! and here is a link to my [project code](https://github.com/tigerinus/TrafficSignClassifier/blob/master/Traffic_Sign_Classifier.ipynb)

### Data Set Summary & Exploration

#### 1. Provide a basic summary of the data set. In the code, the analysis should be done using python, numpy and/or pandas methods rather than hardcoding results manually.

I used the pandas library to calculate summary statistics of the traffic
signs data set:

* The size of training set is `n_train = len(y_train)`.
* The size of validation set is `n_validation = len(y_valid)`.
* The size of test set is `n_test = len(y_test)`.
* The shape of a traffic sign image is `X_train[0].shape`.
* The number of unique classes/labels in the data set is `len(np.unique(y_train))`.

#### 2. Include an exploratory visualization of the dataset.

Here is an exploratory visualization of the data set. It is a bar chart showing how the count of samples per class in training set.

![alt text][image1]

### Design and Test a Model Architecture

#### 1. Describe how you preprocessed the image data. What techniques were chosen and why did you choose these techniques? Consider including images showing the output of each preprocessing technique. Pre-processing refers to techniques such as converting to grayscale, normalization, etc. (OPTIONAL: As described in the "Stand Out Suggestions" part of the rubric, if you generated additional data for training, describe why you decided to generate additional data, how you generated the data, and provide example images of the additional data. Then describe the characteristics of the augmented training set like number of images in the set, number of images for each class, etc.)

As a first step, I decided to convert the images to grayscale because colors of a sign actually do not contribute to the information for classifying it. In fact many images provided do not have the signs in the right color. Here is an example of a traffic sign image before and after grayscaling.

![alt text][image2]

Then I normalized each image to range between -1 and 1 and centered at 0, for better gradient descent.

At last I decided to add some gaussian noise to each image, to improve generalization because data set is unbalanced as seen in the bar chart above. Here is an example of an original image and an image added with gaussian noise:

![alt text][image3]

I did not generate additional data because I ran out of time. But this is how I would do it - I would grab a standard image of each sign from internet, then argument the image with minor rotation, scale up or down, as well as tilting, to mimic real world perspectives of the sign.

#### 2. Describe what your final model architecture looks like including model type, layers, layer sizes, connectivity, etc.) Consider including a diagram and/or table describing the final model.

My final model consisted of the following layers:

| Layer         		|     Description	        					| 
|:---------------------:|:---------------------------------------------:| 
| Input         		| 32x32x1 grayscale image   		    		| 
| Convolution 3x3     	| 1x1 stride, valid padding, outputs 28x28x9 	|
| RELU					|												|
| Max pooling	      	| 2x2 stride,  outputs 14x14x9   				|
| Convolution 3x3     	| 1x1 stride, valid padding, outputs 10x10x27 	|
| RELU					|												|
| Max pooling	      	| 2x2 stride,  outputs 5x5x27   				|
| Flatten       		| outputs 675 									|
| Fully connected		| outputs 300  									|
| RELU          		|             									|
| Drop out              | prob 0.5                                      |
| Fully connected		| outputs 150  									|
| RELU          		|             									|
| Drop out              | prob 0.5                                      |
| Softmax				| outputs `n_classes`							|
|						|												|
|						|												|
 


#### 3. Describe how you trained your model. The discussion can include the type of optimizer, the batch size, number of epochs and any hyperparameters such as learning rate.

* Adam optimizer (recommended by https://machinelearningmastery.com/adam-optimization-algorithm-for-deep-learning/)
* Batch size increases from 50 with each increment = 50. (recommended by https://arxiv.org/abs/1711.00489)
* Epochs = 20 seems to out perform Epochs = 10

#### 4. Describe the approach taken for finding a solution and getting the validation set accuracy to be at least 0.93. Include in the discussion the results on the training, validation and test sets and where in the code these were calculated. Your approach may have been an iterative process, in which case, outline the steps you took to get to the final solution and why you chose those steps. Perhaps your solution involved an already well known implementation or architecture. In this case, discuss why you think the architecture is suitable for the current problem.

My final model results were:
* training set accuracy of ?
* validation set accuracy of ? 
* test set accuracy of ?

If an iterative approach was chosen:
* What was the first architecture that was tried and why was it chosen?

  The first architecture was the default one for MNIST number recognizer. Never built a CNN before that's why it was chosen.

* What were some problems with the initial architecture?

  Validation accuracy was too low, somewhere around 88%

* How was the architecture adjusted and why was it adjusted? Typical adjustments could include choosing a different model architecture, adding or taking away layers (pooling, dropout, convolution, etc), using an activation function or changing the activation function. One common justification for adjusting an architecture would be due to overfitting or underfitting. A high accuracy on the training set but low accuracy on the validation set indicates over fitting; a low accuracy on both sets indicates under fitting.

  Added more channels to conv1 and conv2 layers. Added DropOut layer with `prob = 0.5`.

* Which parameters were tuned? How were they adjusted and why?

  Batch size is incremental per epoch, with 20 epochs in total. See comment to question 3 above.

* What are some of the important design choices and why were they chosen? For example, why might a convolution layer work well with this problem? How might a dropout layer help with creating a successful model?

  TODO

### Test a Model on New Images

#### 1. Choose five German traffic signs found on the web and provide them in the report. For each image, discuss what quality or qualities might be difficult to classify.

Here are five German traffic signs that I found on the web:

![alt text][image4] ![alt text][image5] ![alt text][image6] 
![alt text][image7] ![alt text][image8] ![alt text][image9]
![alt text][image10] ![alt text][image11]     

The first image might be difficult to classify because ...

TODO

#### 2. Discuss the model's predictions on these new traffic signs and compare the results to predicting on the test set. At a minimum, discuss what the predictions were, the accuracy on these new predictions, and compare the accuracy to the accuracy on the test set (OPTIONAL: Discuss the results in more detail as described in the "Stand Out Suggestions" part of the rubric).

Here are the results of the prediction:

| Image			        |     Prediction	        					| 
|:---------------------:|:---------------------------------------------:| 
| Stop Sign      		| Stop sign   									| 
| U-turn     			| U-turn 										|
| Yield					| Yield											|
| 100 km/h	      		| Bumpy Road					 				|
| Slippery Road			| Slippery Road      							|


The model was able to correctly guess 4 of the 5 traffic signs, which gives an accuracy of 80%. This compares favorably to the accuracy on the test set of ...

#### 3. Describe how certain the model is when predicting on each of the five new images by looking at the softmax probabilities for each prediction. Provide the top 5 softmax probabilities for each image along with the sign type of each probability. (OPTIONAL: as described in the "Stand Out Suggestions" part of the rubric, visualizations can also be provided such as bar charts)

The code for making predictions on my final model is located in the 11th cell of the Ipython notebook.

For the first image, the model is relatively sure that this is a stop sign (probability of 0.6), and the image does contain a stop sign. The top five soft max probabilities were

| Probability         	|     Prediction	        					| 
|:---------------------:|:---------------------------------------------:| 
| .60         			| Stop sign   									| 
| .20     				| U-turn 										|
| .05					| Yield											|
| .04	      			| Bumpy Road					 				|
| .01				    | Slippery Road      							|


For the second image ... 

### (Optional) Visualizing the Neural Network (See Step 4 of the Ipython notebook for more details)
#### 1. Discuss the visual output of your trained network's feature maps. What characteristics did the neural network use to make classifications?


