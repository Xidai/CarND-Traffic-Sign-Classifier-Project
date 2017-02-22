#**Traffic Sign Recognition** 


**Build a Traffic Sign Recognition Project**

The goals / steps of this project are the following:
* Load the data set (see below for links to the project data set)
* Explore, summarize and visualize the data set
* Design, train and test a model architecture
* Use the model to make predictions on new images
* Analyze the softmax probabilities of the new images
* Summarize the results with a written report


[//]: # (Image References)

[image1]: ./examples/visualization.jpg "Visualization"
[image2]: ./examples/grayscale.jpg "Grayscaling"
[image3]: ./examples/random_noise.jpg "Random Noise"
[image4]: ./examples/placeholder.png "Traffic Sign 1"
[image5]: ./examples/placeholder.png "Traffic Sign 2"
[image6]: ./examples/placeholder.png "Traffic Sign 3"
[image7]: ./examples/placeholder.png "Traffic Sign 4"
[image8]: ./examples/placeholder.png "Traffic Sign 5"
[exploration]: ./chart-output/data.png
[extra-0]: ./more-signs/0.jpg
[extra-13]: ./more-signs/13.jpg
[extra-14]: ./more-signs/14.jpg
[extra-15]: ./more-signs/15.jpg
[extra-37]: ./more-signs/37.jpg
[0-chart]: ./chart-output/0-chart.png
[13-chart]: ./chart-output/13-chart.png
[14-chart]: ./chart-output/14-chart.png
[15-chart]: ./chart-output/15-chart.png
[37-chart]: ./chart-output/37-chart.png

###Data Set Summary & Exploration

####1. Provide a basic summary of the data set and identify where in your code the summary was done. In the code, the analysis should be done using python, numpy and/or pandas methods rather than hardcoding results manually.

The code for this step is contained in the third code cell of the IPython notebook.  

I used Numpy to calculate summary statistics of the traffic
signs data set:

* The size of training set is 34799
* The size of test set is 12630
* The shape of a traffic sign image is (32, 32, 3)
* The number of unique classes/labels in the data set is 3

####2. Include an exploratory visualization of the dataset and identify where the code is in your code file.

The code for this step is contained in the second code cell of the IPython notebook.  

Here is an exploratory visualization of the data set. It is a bar chart showing how many images of each category are included in the training set, sorted by the number. I draw this image in order to analyze if the training result is related to the size of training data.

![alt text][exploration]

###Design and Test a Model Architecture

####1. Describe how, and identify where in your code, you preprocessed the image data. What tecniques were chosen and why did you choose these techniques? Consider including images showing the output of each preprocessing technique. Pre-processing refers to techniques such as converting to grayscale, normalization, etc.

The code for this step is contained in the fourth code cell of the IPython notebook.

As a first step, I decided to convert the images to grayscale because the intensity of different image may have imapact on the result

As a last step, I normalized the image data to avoid the calculation result getting too large or too small, which will cause big calculation error

####2. Describe how, and identify where in your code, you set up training, validation and testing data. How much data was in each set? Explain what techniques were used to split the data into these sets. (OPTIONAL: As described in the "Stand Out Suggestions" part of the rubric, if you generated additional data for training, describe why you decided to generate additional data, how you generated the data, identify where in your code, and provide example images of the additional data)

The dataset seems changed that there is already a validation set. But if there were not one, we can split a validation from the training data. I would do it like this:

1. shuffle the training set.
2. use the 80% of the data as traning data, and the rest be the validation data.

####3. Describe, and identify where in your code, what your final model architecture looks like including model type, layers, layer sizes, connectivity, etc.) Consider including a diagram and/or table describing the final model.

The code for my final model is located in the seventh cell of the ipython notebook. 

My final model consisted of the following layers:

| Layer         		|     Description	        					| 
|:---------------------:|:---------------------------------------------:| 
| Input         		| 32x32x1 RGB image   							| 
| Convolution 5x5     	| 1x1 stride, same padding, outputs 28x28x6 	|
| RELU					|												|
| Dropout               | keep probability 1/2                          |
| Max pooling	      	| 2x2 stride,  outputs 14x14x6  				|
| Convolution 5x5	    | 1x1 stride, same padding, outputs 10x10x6     |
| RELU                  |                                               |
| Dropout               | keep probability 1/2                          |
| Max pooling           | 2x2 stride, outputs 5x5x16                    |
| Flatten               | outputs 400                                   |
| Fully connected		| outputs 120                                    |
| RELU                  |                                               |
| Dropout               | keep probability 1/2                          |
| Fully connected       | outputs 84                                    |
| RELU                  |                                               |
| Dropout               | keep probability 1/2                          |
| Fully connected       | outputs 43                                    |
| Softmax					 |					                                |           			
 


####4. Describe how, and identify where in your code, you trained your model. The discussion can include the type of optimizer, the batch size, number of epochs and any hyperparameters such as learning rate.

The code for training the model is located in the eigth cell of the ipython notebook. 

To train the model, I used the hyperparameters as following:

1. learning rate: 0.001. I used 0.001 as the first trial, and tried to make bigger, 0.003, 0.005 and 0.01, and also tried to make it smaller, 0.0001, for example, and found that 0.0001 would be too slow that may take too much epochs, and the others to be too big that the result would not be that stable with the process moving on, for example, the validation accuracy would be 0.95 in the 180 epoch and suddenly down to 0.91 in the next epoch. So finally I decide to use 0.001, which is not that slow, and the result differences are kept in the range between [-0.001, 0.001]
2. epochs: 200.
3. batch size: 128
4. dropout keep probability: 0.5


####5. Describe the approach taken for finding a solution. Include in the discussion the results on the training, validation and test sets and where in the code these were calculated. Your approach may have been an iterative process, in which case, outline the steps you took to get to the final solution and why you chose those steps. Perhaps your solution involved an already well known implementation or architecture. In this case, discuss why you think the architecture is suitable for the current problem.

The code for calculating the accuracy of the model is located in the ninth cell of the Ipython notebook.

My final model results were:
* training set accuracy of ?
* validation set accuracy of ? 
* test set accuracy of ?

I choose LeNetn because it's a well know architecture for its good performance on handwriting classification.
 
The result is:

- accuracy on training set: 100%;
- accuracy on validation set: 94%;
- accuracy on test set: 93%

###Test a Model on New Images

####1. Choose five German traffic signs found on the web and provide them in the report. For each image, discuss what quality or qualities might be difficult to classify.

Here are five German traffic signs that I found on the web:

![alt text][extra-0] 
![alt text][extra-13] 
![alt text][extra-14] 
![alt text][extra-15] 
![alt text][extra-37]

The first image might be difficult to classify because its quantity in training data is small, and number "20" on it is also a reason to make it difficult to classify.

For the third image, the word "stop" on it would make it difficult to classify.

####2. Discuss the model's predictions on these new traffic signs and compare the results to predicting on the test set. Identify where in your code predictions were made. At a minimum, discuss what the predictions were, the accuracy on these new predictions, and compare the accuracy to the accuracy on the test set (OPTIONAL: Discuss the results in more detail as described in the "Stand Out Suggestions" part of the rubric).

The code for making predictions on my final model is located in the tenth cell of the Ipython notebook.

Here are the results of the prediction:

| Image			        |     Prediction	        					| 
|:---------------------:|:---------------------------------------------:| 
| Speed limit (20km/h)  | Roundabout mandatory							| 
| Yield     			| Yield 										|
| Go straight or left   | Go straight or left           				|
| Stop	           		| Speed limit (50km/h)  		 				|
| No vehicles       	| Keep right        							|


The model was able to correctly guess 2 of the 5 traffic signs, which gives an accuracy of 40%. 

####3. Describe how certain the model is when predicting on each of the five new images by looking at the softmax probabilities for each prediction and identify where in your code softmax probabilities were outputted. Provide the top 5 softmax probabilities for each image along with the sign type of each probability. (OPTIONAL: as described in the "Stand Out Suggestions" part of the rubric, visualizations can also be provided such as bar charts)

The code for making predictions on my final model is located in the 11th cell of the Ipython notebook.

For the first image, the model is relatively sure that this is a Roundabout mandatory (probability of 0.742), but the image is Speed limit (20km/h). The top five soft max probabilities were

![alt text][0-chart]

For the second image, the model is very sure this is a Yield(probability of 1!), and the image is indeed a Yield. The top five soft max probabilities were:

![alt text][13-chart]

For the third image, the model is also very sure this is a Go straight or left with probability of 1, and again it is right. The top five soft max probabilities were:

![alt text][14-chart]

For the fourth image, the model is very sure this is a Speed limit (50km/h) (probability of 0.942!), but this time it is wrong. The right answer is Stop. The top five soft max probabilities were:

![alt text][15-chart]

For the last image, the model is not very confident with its result of Keep right (probability of 0.342), and the right answer should be No vehicles. The top five soft max probabilities were:

![alt text][37-chart]
