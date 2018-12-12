[![Udacity - Robotics NanoDegree Program](https://s3-us-west-1.amazonaws.com/udacity-robotics/Extra+Images/RoboND_flag.png)](https://www.udacity.com/robotics)

## Deep Learning Project ##

The goal of this project is to train a deep neural network to identify and track a moving target using Fully Convolutional Network (FCN). So-called "follow me" applications. This network run real-time pixel-wise classification on images. The steps are following:

1. Download the training dataset and extract to the project `data` directory.
2. Implement the solution in `model_training.ipynb`
3. Train the network locally, or on [AWS](https://classroom.udacity.com/nanodegrees/nd209/parts/09664d24-bdec-4e64-897a-d0f55e177f09/modules/cac27683-d5f4-40b4-82ce-d708de8f5373/lessons/197a058e-44f6-47df-8229-0ce633e0a2d0/concepts/27c73209-5d7b-4284-8315-c0e07a7cd87f?contentVersion=1.0.0&contentLocale=en-us).
4. Continue to experiment with the training data and network until attain the score desired.
5. Once comfortable with performance on the training dataset, see how it performs in live simulation.

[image1]: ./docs/misc/fcn.png
[image2]: ./docs/misc/following.png
[image3]: ./docs/misc/patrol_target.png
[image4]: ./docs/misc/patrol_no_target.png
[image5]: ./docs/misc/sim_screenshot.png

![alt text][image5]


## Setup Instructions

**Download the data**

Save the following three files into the data folder of the cloned repository. 

* [Training Data](https://s3-us-west-1.amazonaws.com/udacity-robotics/Deep+Learning+Data/Lab/train.zip) 
* [Validation Data](https://s3-us-west-1.amazonaws.com/udacity-robotics/Deep+Learning+Data/Lab/validation.zip)
* [Sample Evaluation Data](https://s3-us-west-1.amazonaws.com/udacity-robotics/Deep+Learning+Data/Project/sample_evaluation_data.zip)

**Download the QuadSim binary**

The simulator binary can be downloaded [here](https://github.com/udacity/RoboND-DeepLearning/releases/latest)

**Install Dependencies**

Make sure you have following installed:

* Python 3.x
* Tensorflow 1.2.1
* NumPy 1.11
* SciPy 0.17.0
* eventlet 
* Flask
* h5py
* PIL
* python-socketio
* scikit-image
* transforms3d
* PyQt4/Pyqt5

The best way to get setup with these is to use Anaconda following along with the [RoboND-Python-Starterkit](https://github.com/udacity/RoboND-Python-StarterKit).


## Fully Convolutional Network

A typical CNN's (Convolutional Neural Network) fully connected layers have few disadvantages:

* don't preserve spatial information
* constrain the network's input size

FCN (Fully Convolutional Networks) can do what CNNs cannot do - it preserve the spatial information throughout the entire network, and will work with images of any size.

FCN is comprised of two parts: encoder and decoder.

![alt text][image1] 

### FCN: Encoder

The encoder is a convolution network that reduces to a deeper 1x1 convolution layer, in contrast to a flat fully connected layer that would be used for basic classification of images. This difference has the effect of preserving spacial information from the image.

In order to increase the encoder's efficiency, **separable convolutions** was used to reduce the number of parameters.

I further optimized the network with **batch normalization** by normalizing the output(i.e., input to the next layer) of each convolutional layer. This has following advantages:

* Networks train faster
* Allows higher learning rates

I did this in the function `separable_conv2d_batchnorm()`, cell 2 of `model.training.ipynb`.

### FCN: Decoder

The decoder upscales the output of encoder such that the network's output is the same size as the original image. 

Instead of transposed convolutions, I used **bilinear upsampling** to speed up the upsampling performance - note that this method is prone to lose some finer details. I did this in the function `bilinear_upsample()` in cell 3 of `model.training.ipynb`.

### FCN: Skip Connection

One effect of convolution or encoding is it narrows down the scope by looking closely at some pictures and lose the bigger pictre as a result. So even if we decode the output of the encoder back to the original image size, some information has been lost.

Skip connection is a way to retain information by connectin the output of one layer to a non-adjacent layer.

To implement the skip connection, I concatenate two layers, the upsampled layer and a layer with more spatial information than the upsampled one. Then I added some regular convolution layers for the model to be able to learn the finer spatial details.

I did this in the function `decoder_block()` in cell 5 of `model_training.ipynb`.

### FCN: Model

I started with simple models with a single encoder layer and a single decoder layer, and then added more layers to it as needed. I setup the model in the function `fcn_model()` in cell 6 of `model_training.ipynb`.


### FCN: Hyperparameters

| Hyperparameter   | Value |
| ---------------- | ----- |
| Batch Size       | 96    |
| Step per Epoch   | 43    |
| Validation Steps | 12    |
| Learning Rate    | 0.001 |
| Number of Epochs | 48    |

I set these values in cell 8 of `model_training.ipynb`.

#### Batch Size
Batch size is the number of training samples/images that get propagated through the network in a single pass. This value is limited by the available GPU memory, I started with 32, and pushed it to 96.

#### Steps per Epoch

This is the number of batches of training images that go through the network in 1 epoch. I set it to a value based on the total number of images in training dataset divided by the chosen batch size.

#### Validation Steps

number of batches of validation images that go through the network in 1 epoch. Similar to the steps per epoch, I set this value to be the total number of validation images divided by the chosen batch size.

#### Learning Rate

Learning rate scales the magnitude of the weight updates in gradient descent in order to minimize the network's loss. If set too low, the training will progress very slow; if set to high, there will be strong fluctuation in the loss function and may fail to converge to the minimum. I started with a large value, 0.1, then tried exponentially lower values: 0.01, 0.001, and found 0.001 to be optimal.

#### Number of Epochs

The number of times the entire training dataset gets propagated through the network. As the number of epochs increase, the network goes from underfitting to optimal to overfitting. I trained for 48 epochs.

### Limitation of Current Model

The model was trained with 3 classes:

1. the Hero (target person)
2. other people
3. everything else

In order to following another object (dog, cat, car, etc.) instead of a human, the model will have to work with more classes - this means to collect and label more images including the additional desired classes.

## Results

The network achieved an accuracy of **43.4%** using the Intersection over Union (IoU) metric.

Below are the predictions for three scenarios made from the evaluation dataset. These image columns are in following order: input image from camera(left), ground truth(center), the network's prediction(right).

* **Following:** the network identified the hero and following him/her.

![alt text][image2] 

* **Patrol with Target:** when the hero is visible from a distance.

![alt text][image3] 

* **Patrol without Target:** When the hero is not visible.

![alt text][image4] 


## Future Works

* Collect more training data to improve the accuracy.
* Use pre-trained networks in encoder instead.