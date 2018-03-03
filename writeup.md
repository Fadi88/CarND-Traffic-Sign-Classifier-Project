# **Traffic Sign Recognition** 

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

[image1]: ./graphs/trainings_vis.jpg "Visualization"
[image2]: ./graphs/input_image.png "Input image"
[image3]: ./graphs/norm.image.png "Output image"
[image4]: ./examples/placeholder.png "Traffic Sign 1"
[image5]: ./examples/placeholder.png "Traffic Sign 2"
[image6]: ./examples/placeholder.png "Traffic Sign 3"
[image7]: ./examples/placeholder.png "Traffic Sign 4"
[image8]: ./examples/placeholder.png "Traffic Sign 5"

## Rubric Points
### Here I will consider the [rubric points](https://review.udacity.com/#!/rubrics/481/view) individually and describe how I addressed each point in my implementation.  

---
### Writeup 

### Data Set Summary & Exploration

#### 1. Basic summary of the data set. 
I used the pandas library to calculate summary statistics of the traffic
signs data set:

* The size of training set is 34799 images
* The size of the validation set is 4410images
* The size of test set is 4410 images
* The shape of a traffic sign image is 32x32 pxiel
* The number of unique classes/labels in the data set is 43

#### 2. Exploratory visualization of the dataset.

Here is an exploratory visualization of the data set. It is a bar chart showing how the data ...

![alt text][image1]

as well as in the HTML report for every sign class the first 7 samples are displayed

### Design and Test a Model Architecture

Unlike the LeNet model i deciced to keep the colors in the images and make the conv. layer accpet 3 chanels and this was to add extra features to train the network, for traffic signs not only the shape is important but even for us as humans colors means alot.

what did happen however was the normalization for every channel in the three chanel to keep them avearges with with minimum standard divation to give the network a good starting point for training. 
here is a sample of before and after
![alt text][image2] ![alt text][image3] 

all the data set were modifed and stored before passing it to the network

there was some trials with the gray scale iamges (including with precentage of each channel to take) but it was found that the color images yielded and better training accuracy.

there was also some trails with the gaussian blur, but it did not lead to any improvement in the accuracy so the image preprocessing with only the normalisation.

there was no need to add any more data since the model was getting up to 94% accuracy with the validation and test data 



#### 2. Network model.

My final model consisted of the 2 convolution layers and 3 fully connected layers (just like LeNet) as follows:

| Layer         		      |     Description	        					                 | 
|:---------------------:|:---------------------------------------------:| 
| Input         		      | 32x32x3 RGB image   						                   	| 
| Convolution 3x3      	| 1x1 stride, valid padding, outputs 30x30x10 	 |
| RELU					             |							                                   					|
| Max pooling	         	| 2x2 stride,  outputs 15x15x10 			            	|
| Convolution 5x5	      |1x1 stride, valid padding, outputs 11x11x20 	  |
| RELU	                	|         							                               |
| Fully connected(1)			 | faltten 2420 from the conv. layer and output 600 	|
|					RELU             	|												                                   |
|	Fully connected(2)   	|			600 inputs, 250 outputs				                	|
| RELU                  |                                               |
| Fully connected(3)    | 250 inputs, 43 output                         |
| output layer          | softmax cross entropy                         |

PS: in order to keep the features in the model high to get better accuracy, the second pooling layer was removed
 


#### 3. Model Training.

the training happened with 0.001 learning rate, using Adam optimizer, epochs number of 50 and 64 element in the batch size.
all those values were ontained by trial and error

#### 4. Describe the approach taken for finding a solution and getting the validation set accuracy to be at least 0.93. Include in the discussion the results on the training, validation and test sets and where in the code these were calculated. Your approach may have been an iterative process, in which case, outline the steps you took to get to the final solution and why you chose those steps. Perhaps your solution involved an already well known implementation or architecture. In this case, discuss why you think the architecture is suitable for the current problem.

My final model results were:
* training set accuracy of ?
* validation set accuracy of ? 
* test set accuracy of ?

If an iterative approach was chosen:
* What was the first architecture that was tried and why was it chosen?
* What were some problems with the initial architecture?
* How was the architecture adjusted and why was it adjusted? Typical adjustments could include choosing a different model architecture, adding or taking away layers (pooling, dropout, convolution, etc), using an activation function or changing the activation function. One common justification for adjusting an architecture would be due to overfitting or underfitting. A high accuracy on the training set but low accuracy on the validation set indicates over fitting; a low accuracy on both sets indicates under fitting.
* Which parameters were tuned? How were they adjusted and why?
* What are some of the important design choices and why were they chosen? For example, why might a convolution layer work well with this problem? How might a dropout layer help with creating a successful model?

If a well known architecture was chosen:
* What architecture was chosen?
* Why did you believe it would be relevant to the traffic sign application?
* How does the final model's accuracy on the training, validation and test set provide evidence that the model is working well?
 

### Test a Model on New Images

#### 1. Choose five German traffic signs found on the web and provide them in the report. For each image, discuss what quality or qualities might be difficult to classify.

Here are five German traffic signs that I found on the web:

![alt text][image4] ![alt text][image5] ![alt text][image6] 
![alt text][image7] ![alt text][image8]

The first image might be difficult to classify because ...

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


