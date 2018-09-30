# **Behavioral Cloning** 

## Project Writeup

---

**Behavioral Cloning Project**

The goals / steps of this project are the following:
* Use the simulator to collect data of good driving behavior
* Build, a convolution neural network in Keras that predicts steering angles from images
* Train and validate the model with a training and validation set
* Test that the model successfully drives around track one without leaving the road
* Summarize the results with a written report


[//]: # (Image References)

[image1]: ./examples/architecture.png "Model Visualization"

## Rubric Points
### Here I will consider the [rubric points](https://review.udacity.com/#!/rubrics/432/view) individually and describe how I addressed each point in my implementation.  

---
### Files Submitted & Code Quality

#### 1. Submission includes all required files and can be used to run the simulator in autonomous mode

My project includes the following files:
* [model.py](./model.py) containing the script to create and train the model
* [drive.py](./drive.py) for driving the car in autonomous mode
* [model.h5](./model.h5) containing a trained convolution neural network 
* [writeup_report.md](./writeup_report.md) summarizing the results

#### 2. Submission includes functional code
Using the Udacity provided simulator and my drive.py file, the car can be driven autonomously around the track by executing 
```sh
python drive.py model.h5
```

#### 3. Submission code is usable and readable

The [model.py](./model.py) file contains the code for training and saving the convolution neural network. The file shows the pipeline I used for training and validating the model, and it contains comments to explain how the code works.

### Model Architecture and Training Strategy

#### 1. An appropriate model architecture has been employed

I experimented with multiple CNN architectures: namely, a super simple model, LeNet model and then finally
settled on the NVidia model as explained in this [paper](https://images.nvidia.com/content/tegra/automotive/images/2016/solutions/pdf/end-to-end-dl-using-px.pdf).
Specifically, the architectire has:

* normalization layer (plus mean centering)
* 5 Convolutional layers with 24, 36, 48, 64, 64 filters and 5x5, 5x5, 5x5, 3x3, 3x3 kernels respectively.
* All activations are [ReLU](https://arxiv.org/pdf/1803.08375.pdf).
* This is followed by flattening
* And then 3 fully connected layers with 100, 50 and 10 neurons respective
* The output is a Dense(1) layer
* I used [MSE loss function](https://keras.io/losses/) and [Adam optimizer](https://keras.io/optimizers/)

#### 2. Attempts to reduce overfitting in the model

While, I did not have a need to use dropout, I ensured the model was not overfitting. I verified this by
making sure that the training set error was equivalent to validaiton set error. I also reduced the epochs from
10 to 3 as I noticed some overfitting beyond those epochs.

#### 3. Model parameter tuning

The model used an adam optimizer, so the learning rate was not tuned manually.

#### 4. Appropriate training data

I used the training data provided as part of the assignment. I did not create any new training data of my
own as the provided training data was good for the training task.

### Model Architecture and Training Strategy

#### 1. Solution Design Approach

I first tried a simple model with a single Dense(1) layer. This turned out to be a very simple model and
did not have enough parameters to learn the driving task. Hence I did not pursue this further.

Next I tried using the LeNet model and that seems to have made progress. I was able to get the car to drive
a bit further, althrough it did not stay on the road for the entire lap.

Then I decided to try the NVidia model since it seemed to have worked for real-world driving. And it worked
fairly quickly!

I performed data normalization. Also, I performed data augmentation by flipping the provided images, so that
I get a good data size. This seems to have helped quite a bit. Another thing really helped was cropping the
images to remove uninteresting data. It reduced the noise significantly.

At the end of the process, the vehicle is able to drive autonomously around the track without leaving the road
as seen in the [video](./video.mp4). It has been also uploaded to Youtube and can be seen [here](https://www.youtube.com/watch?v=CjjyF9zNDBs).

#### 2. Final Model Architecture

The final model architecture consisted of a convolution neural network with the following layers:

![alt text][image1]

#### 3. Creation of the Training Set & Training Process

I used the provided data for the training set. Cropping of the training data and creating the flip images
was the most important data preparation and augmentation step in my view.

I randomly shuffled the data set and put 20% of the data into validation set. 

I used this training data for training the model. The validation set helped determine if the model was over or under fitting. The ideal number of epochs was 3. I used an adam optimizer so that manually training the learning rate wasn't necessary.
