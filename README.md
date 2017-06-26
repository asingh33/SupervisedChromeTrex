# SupervisedChromeTrex
Chrome Browser's TRex self playing AI via CNN neural network implemented in Keras + Theano + OpenCV

Key Requirements:
Python 2.7.13
OpenCV 2.4.8
Keras 2.0.2
Theano 0.9.0

Suggestion: Better to download Anaconda as it will take care of most of the other packages and easier to setup a virtual workspace to work with multiple versions of key packages like python, opencv etc.


# Key repo contents
- **main.py** : The main script launcher. This file contains all the code for UI options and OpenCV code to capture camera contents. This script internally calls interfaces to actionCNN.py.
- **actionCNN.py** : This script file holds all the CNN specific code to create CNN model, load the weight file (if model is pretrained), train the model using image samples present in **./imgfolder**.
- **imgfolder** : This folder contains all the 300 gameplay images I took in order to train the model.

# Usage
```bash
$ KERAS_BACKEND=theano python main.py 
```
We are setting KERAS_BACKEND to change backend to Theano, so in case you have already done it via Keras.json then no need to do that. But if you have Tensorflow set as default then this will be required.


# What are we trying to achieve this time ?
In this project I am using Supervised Machine Learning technique to teach our beloved TRex to play and then let it play the game itself!  Now keep in mind, I am not going after a perfect logic that will enable our TRex to play forever (that would be cheating right?).
 
# What is Supervised Learning ?
"Supervised learning is the machine learning task of inferring a function from labeled training data.[1] The training data consist of a set of training examples. In supervised learning, each example is a pair consisting of an input object (typically a vector) and a desired output value (also called the supervisory signal). A supervised learning algorithm analyzes the training data and produces an inferred function, which can be used for mapping new examples. An optimal scenario will allow for the algorithm to correctly determine the class labels for unseen instances. This requires the learning algorithm to generalize from the training data to unseen situations in a "reasonable" way (see inductive bias). - Wikipedia "
 
Simplest analogy I can think of is the way we are taught at school, where the teacher teaches by first asking a problem and then provides a solution to it. For eg: 2 + 3 = 5.

# How are we gonna teach ?
If you have played this game then you would already be knowing that this TRex character can be controlled with following actions:
- Jump (Up Arrow key or Space key)
- Crouch (Down Arrow key) or
- Just Run (default action)
 
So all we require to teach our TRex, is when to jump and when not to jump (For simplicity we are ignoring Crouch action, Crouching is for noobs ...).
For this, our TRex should have following two capabilities:
- Eyes : Ability to see what we are teaching it. We could use OpenCV library to capture the screen contents through an external camera Or directly read back the screen contents more like taking continues screenshots. I have used second option.
- Brain : Ability to make decision of when to jump and when not. We would use Convolution Neural Network for this. Images that we capture above will be fed into the Neural Network, first to train it and later for predictions. Since we have to make just two decisions of Jump or Not Jump, we will train this model for two sets of training image data. One set of image samples will tell when to Jump and the other set for when not to Jump.
 
Note: We don't want to capture the whole screen contents but just the game area of the browser where TRex is playing should be sufficient for our need.
 
Now lets generate our training image sample data. For this I would suggest you to play the game yourself, if you have not played before. Just open a Chrome browser, disconnect Internet and try to open any website. You should see "There is no internet connection" message. Just press Up Arrow key to initiate the game.
Carefully monitor when you are pressing the Up Arrow key to jump. You should easily notice that as soon as any obstacle comes near to TRex, up-to a certain distance, you make TRex to jump over the obstacle. You don't want to jump too early or too late. This also means that we are only concerned about that small region around TRex which extends till this 'certain distance' on the right of TRex and we will capture image contents of this region instead of complete game region for better efficiency due to less image processing involved.
 
In the code, I have added below logic to save training image sample
if  I press Up Arrow key:
      Save the screen contents as Jump sample image, before making TRex jump
else
      Save the screen contents as No Jump sample image
 
And here are how my training image samples look like:
 
Jump sample set (149 images):

![](https://github.com/asingh33/SupervisedChromeTrex/blob/master/misc/jump.gif)


No Jump sample set (151 images):

![](https://github.com/asingh33/SupervisedChromeTrex/blob/master/misc/NoJump.gif)
 
Once I trained the model using above image samples, I used the trained model to do the predictions on a live game execution. By the way, the prediction outputs (Jump or No Jump) are passed to the Chrome browser i.e Up Arrow Key (Jump) & no input (for No Jump) to control TRex. And this is how it turned out.

Youtube link - https://youtu.be/ZZgvklkQrss

![](https://j.gifs.com/DRg4mn.gif)


This is still not perfect for instance, as you progress the TRex velocity increases and this also impacts the instance you need to make it jump. Current image samples don't handle this scenario. But like I said before, I am not looking for a perfect model. Perhaps if you are interested then you can perfect it, will be good exercise.
