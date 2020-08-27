# Android-Face-Recognition
Android Face-Recognition application using OpenCV for face detection and MobileFacenet for face verification

## Overview
Face verification is an important identity authentication technology used in more and more mobile and embedded applications such as device unlock, application login, 
mobile payment and so on. Some mobile applications equipped with face verification technology, for example, smartphone unlock, need to run offline. 
To achieve userfriendliness with limited computation resources, the face verification models deployed locally on mobile devices are expected to be not only accurate but also small and fast. 
However, modern high-accuracy face verification models are built upon deep and big convolutional neural networks (CNNs) which are supervised by novel loss functions during training stage. 
The big CNN models requiring high computational resources are not suitable for many mobile and embedded applications. 
MobileFaceNets is a class of extremely efficient CNN models tailored for high-accuracy real-time face verification on mobile and embedded devices.
Performance comparison with previous published face verification models on LFW Dataset shown in the table below:  
![Performance comparison](https://raw.githubusercontent.com/Malikanhar/Android-Face-Recognition/assets/github-assets/Mobile%20Facenet%20Performance.PNG)  
As we can see, MobileFacenet is very small in size but has very high accuracy

## Getting Started
### Pre-requisite
* [Download](https://sourceforge.net/projects/opencvlibrary/files/opencv-android/3.4.3/opencv-3.4.3-android-sdk.zip/download) and add OpenCV to the android project 
(you can see [this](https://medium.com/@sukritipaul005/a-beginners-guide-to-installing-opencv-android-in-android-studio-ea46a7b4f2d3) tutorial to add OpenCV library to your android project)
* Download pre-trained MobileFacenet from [sirius-ai/MobileFaceNet_TF](https://github.com/sirius-ai/MobileFaceNet_TF), convert the model to tflite using the following [notebook](https://colab.research.google.com/drive/1S1Lsiouui-odYj06tMnQwzzEGr0M22Fk?usp=sharing)
and put it in android assets folder

## Result
How is the result? Is it fast enough to run on a mobile device?  
![Demo Application](https://github.com/Malikanhar/Android-Face-Recognition/raw/assets/github-assets/app-demo.gif)

## References
[1] [MobileFaceNets: Efficient CNNs for Accurate Real-time Face Verification on Mobile Devices](https://arxiv.org/ftp/arxiv/papers/1804/1804.07573.pdf)  
[2] [sirius-ai/MobileFaceNet_TF](https://github.com/sirius-ai/MobileFaceNet_TF)
