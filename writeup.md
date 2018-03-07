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
[image4]: ./tests/0.jpg "Traffic Sign 1"
[image5]: ./tests/1.jpg "Traffic Sign 2"
[image6]: ./tests/2.jpg "Traffic Sign 3"
[image7]: ./tests/3.jpg "Traffic Sign 4"
[image8]: ./tests/4.jpg "Traffic Sign 5"
[image9]: ./tests/5.jpg "Traffic Sign 6"
[image10]: ./tests/6.jpg "Traffic Sign 7"
[image11]: ./tests/7.jpg "Traffic Sign 8"

## Rubric Points
### Here I will consider the [rubric points](https://review.udacity.com/#!/rubrics/481/view) individually and describe how I addressed each point in my implementation.  

---

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

Unlike the LeNet model i deciced to keep the colors in the images and make the conv. layer accpet 3 channels and this was to add extra features to train the network, for traffic signs not only the shape is important but even for us as humans colors means alot unlike the MNIST data set where the shape and outline were enough.

what did happen however was the normalization for every channel in the three channel to keep them avearged with minimum standard divation to give the network a good starting point for training. 
here is a sample of before and after
![alt text][image2] ![alt text][image3] 

all the data set were modifed and stored before passing it to the network

there was some trials with the gray scale iamges (including with precentage of each channel to take) but it was found that the color images yielded a better training accuracy.

there was also some trails with the gaussian blur, but it did not lead to any improvement in the accuracy so the image preprocessing with only the normalisation.

there was no need to add any more data since the model was getting up to 93% accuracy with the validation and test data 



#### 2. Network model.

My final model consisted of the 2 convolution layers and 4 fully connected layers as follows:

| Layer         		      |     Description	        					                 | 
|:---------------------:|:---------------------------------------------:| 
| Input         		      | 32x32x3 RGB image   						                   	| 
| Convolution 7x7      	| 1x1 stride, valid padding, outputs 26x26x10 	 |
| RELU					             |							                                   					|
| Max pooling	         	| 2x2 stride,  outputs 14x14x10 			            	|
| Convolution 7x7	      | 1x1 stride, valid padding, outputs 8x8x16  	  |
| RELU	                	|         							                               |
| dropout              	|   adaptive value                              |
| Fully connected(1)			 | faltten 784 from the conv. layer and output 550 	|
|					RELU             	|												                                   |
| dropout              	|   adaptive value                              |
|	Fully connected(2)   	|			550 inputs, 400 outputs				                	|
| RELU                  |                                               |
| Fully connected(3)    |   400 inputs, 220 outputs                     |
| RELU                  |                                               |
|	Fully connected(5)   	|			220 inputs, 43 outputs				                 	|
| output layer          |     softmax cross entropy                     |

PS: in order to keep the features in the model high to get better accuracy, the second pooling layer was removed
 


#### 3. Model Training.

the batch size was 50 after trial and error

for the epochs a limit of 80 was picked, however to avoid overfitting early termination was used when validation accrucay got to 94.5% or more

the inital learning rate chosen for the training was 0.023

the initial dropout rate chosen was 20% (0.8 keep prob.)

however to an adaptive approch was picked to control the value of both the dropout ratio and the learning rate 
when the model got over 90% the learning was decreased to make it converge 

#### 4. Describe the approach taken for finding a solution and getting the validation set accuracy to be at least 0.93. Include in the discussion the results on the training, validation and test sets and where in the code these were calculated. Your approach may have been an iterative process, in which case, outline the steps you took to get to the final solution and why you chose those steps. Perhaps your solution involved an already well known implementation or architecture. In this case, discuss why you think the architecture is suitable for the current problem.

My final model results were:
* training set accuracy of 99.55%
* validation set accuracy of 94.54%
* test set accuracy of 93%

If an iterative approach was chosen:
* the first network tested was the same as LeNet with the final layer changed
* the main problems with this aproch was overfitting (high training reuslts vs above average test results) and not getting very high accurcay
* to gain more felxibalty the size of the kernel in each conv.layer was increased to 7*7 to gain more paramters 
* the second pooling layer was removed and replaced with a dropout to counter the overfitting observed with the orignal approch
* 2 more fully connected layers were added to add more paramters and more degree of freedoms to the network  
* it was decided to use a relatively learning rate with a batch size that is not so big wichi might lead to unsteablty so an adaptive learning rate was chosen for when good results were obtained the rate change to help the model converge more towards the least possible error 
 

### Test a Model on New Images

#### 1. Choose five German traffic signs found on the web and provide them in the report. For each image, discuss what quality or qualities might be difficult to classify.

Here are five German traffic signs that I found on the web:

![alt text][image4] ![alt text][image5] ![alt text][image6] 
![alt text][image7] ![alt text][image8] ![alt text][image9]
![alt text][image10] ![alt text][image11]

#### 2. internet images predictions result.
please check the attached html report 

#### 3. reulsts discussion 

5 images were predicted corerctly with high confidence (almost 100% for 4 and one with 99.91%)

the sign of bicycle crossing is predicted as children corssing mostly due to high resemblance specialy at this low resolution

the sign of no vehicles detected as traffic signals with very low confidence (the top result 50.1% vs 49.8% for the correct one)

wild animals crossing was detected as slippery road due to high resemblance specialy at this low resolution



