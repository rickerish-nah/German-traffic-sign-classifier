##Project:  TRAFFIC SIGN CLASSIFIER
[![Udacity - Self-Driving Car NanoDegree](https://s3.amazonaws.com/udacity-sdc/github/shield-carnd.svg)](http://www.udacity.com/drive)
### Overview

In this project, I will be using convolutional neural networks to classify traffic signs. The data set used is, [German Traffic Sign Dataset](http://benchmark.ini.rub.de/?section=gtsrb&subsection=dataset). 
After the model is trained, we will then test it on new images of traffic signs we find on the web, or, if you're feeling free, pictures of traffic signs you find locally!


### Dependencies

This project requires the following Python libraries installed:

- [Jupyter](http://jupyter.org/)
- [NumPy](http://www.numpy.org/)
- [SciPy](https://www.scipy.org/)
- [scikit-learn](http://scikit-learn.org/)
- [TensorFlow](http://tensorflow.org)
- [Matplotlib](http://matplotlib.org/)

### Data Set Summary & Exploration

**Dataset**
1. [Download the dataset](https://d17h27t6h515a5.cloudfront.net/topher/2016/November/581faac4_traffic-signs-data/traffic-signs-data.zip). This is a pickled dataset in which we've already resized the images to 32x32. (Contains training, validation and testing images)

**Summary**
* The size of training set = 34799
* The size of the validation set is = 4410
* The size of test set is = 12630
* The shape of a traffic sign image = 32x32x3
* The number of unique classes/labels in the data set = 43
**Visualization:** Please look into the ipynb file at In[5] and In[6]

### Model Architecture

#### 1. Describe how you preprocessed the image data. What techniques were chosen and why did you choose these techniques? Consider including images showing the output of each preprocessing technique. Pre-processing refers to techniques such as converting to grayscale, normalization, etc. (OPTIONAL: As described in the "Stand Out Suggestions" part of the rubric, if you generated additional data for training, describe why you decided to generate additional data, how you generated the data, and provide example images of the additional data. Then describe the characteristics of the augmented training set like number of images in the set, number of images for each class, etc.)

Step 1: I converted all the color images to grayscale images
Step 2: I normalized all the images 
In general, It is better to work on grayscale images as they are computationally less expensive than color images. Normalization is generally done on most of the data before it is trained/tested. It improves the accuracy of the system.


#### 2. Describe what your final model architecture looks like including model type, layers, layer sizes, connectivity, etc.) Consider including a diagram and/or table describing the final model.

My final model consisted of the following layers:

| Layer                 |     Description                                | 
|:---------------------:|:---------------------------------------------:| 
| Input                  | 32x32x3 RGB image converted to grayscale   |
| Layer 1              |  32x32x1 --> 28x28x6
| Pooling              | 28x28x6 --> 14x14x6        |
| Layer 2             | 14x14x1 --> 10x10x16  |
| Pooling              | 10x10x16 --> 5x5x16  (400)    |
| FCC 1                  | 400 --> 120  |
| FCC 2                  | 120 --> 43    |
| Activation            | RELU |



#### 3. Describe how you trained your model. The discussion can include the type of optimizer, the batch size, number of epochs and any hyperparameters such as learning rate.

In training the model,
Cross entropy was used as the loss function.
Adam optimizer was used.
The Hyper-parameters that were set are : Epoch = 100, Batch size = 100, Learning rate = 0.001

#### 4. Describe the approach taken for finding a solution and getting the validation set accuracy to be at least 0.93. Include in the discussion the results on the training, validation and test sets and where in the code these were calculated. Your approach may have been an iterative process, in which case, outline the steps you took to get to the final solution and why you chose those steps. Perhaps your solution involved an already well known implementation or architecture. In this case, discuss why you think the architecture is suitable for the current problem.

My final model results were:
* training set accuracy = 99.9%
* validation set accuracy = 97.6 %
* test set accuracy = 95.1%
To get this accuracy result:
I only used LeNet architecture, but the play was on setting the hyper-parameters and the optimizer.
I tried using SGD initially but the loss it incurred while training were greater than that of adams, So I chose adam optimizer over the other. For the parameters, I tweeked the values for Batch-size, learning rate and epochs, especially learing rate and epochs as they go hand in hand. I observed that, if the number of iterations is small, then the model may have not completed its training and overfit for case vice-versa. And learing rate is inversely propotional to the time for the model to train. Smaller learning rates may train the system more accurately while it may take a very long time to train. So it is important to properly handle these hyperparameters.
### Test a Model on New Images

#### 1. Choose five German traffic signs found on the web and provide them in the report. For each image, discuss what quality or qualities might be difficult to classify.

The six German traffic signs that I found on the web: refer In[62].  [1,2]
																	 [3,4]
																	 [5,6]
Figures 1,4 and 5 are bright when compared to the figures 2,3 and 6 which are slightly dull.
These brighter images are also crisp in their details, i.e. good contrast images except for fig 1.
Amongst all these images, It can be seen that the image 1 is slightly blurry and little over bright making it little harder to predict than the rest.
Figure 6 has the apt qualities to be predicted better than the rest as it's details are well captured even though it appears pixelated. So while going through the convolutional layers, it will provide a better edge map over the other.


#### 2. Discuss the model's predictions on these new traffic signs and compare the results to predicting on the test set. At a minimum, discuss what the predictions were, the accuracy on these new predictions, and compare the accuracy to the accuracy on the test set (OPTIONAL: Discuss the results in more detail as described in the "Stand Out Suggestions" part of the rubric).

Here are the results of the prediction:

| Image                    |     Prediction                                | 
|:---------------------:|:---------------------------------------------:| 
| Speed Limit 30kmph   |  Mandatory Round-about                     | 
| Bumpy Road                | Bumpy Road                                        |
| Ahead only                    | Ahead Only                                      |
| Go straight or left                | Go straight or left                      |
| General caution           | General caution                                 |


The model was able to correctly guess 4 of the 6 traffic signs, which gives an accuracy about 70%. This compares favorably to the accuracy on the test set.

#### 3. Describe how certain the model is when predicting on each of the five new images by looking at the softmax probabilities for each prediction. Provide the top 5 softmax probabilities for each image along with the sign type of each probability. (OPTIONAL: as described in the "Stand Out Suggestions" part of the rubric, visualizations can also be provided such as bar charts)

The code for making predictions on my final model is located in the 66th cell of the Ipython notebook.

For 5 of the 6 images, The top 2 soft max probabilities are given below. For all 5 soft-max probabilities see 66th cell in the notebook.

| Image                          | Prediction 1                                 | Prediction 2       | 
| Speed Limit 30kmph   |  Mandatory Round-about  (74%) | Go straight or left (18%) |
| Bumpy Road                | Bumpy Road    (100%)              |  |
| Ahead only                    | Ahead Only     (100%)          | Go straight or left  (0%) |
| Go straight or left                | Go straight or left        (100 %)               | General Caution ( 0% ) |
| General caution           | General caution            (100%)                    | Pedestrain crossing (0%) |


