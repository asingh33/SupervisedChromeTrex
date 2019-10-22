#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Thu Apr  6 01:01:43 2017

@author: abhisheksingh
"""

from keras.models import Sequential
from keras.layers import Dense, Dropout, Activation, Flatten
from keras.layers import Conv2D, MaxPooling2D
from keras.utils import np_utils

# We require this for Theano lib ONLY. Remove it for TensorFlow usage
from keras import backend as K
#K.set_image_dim_ordering('th')
K.set_image_data_format('channels_first')

import numpy as np

import os
import theano
from PIL import Image
# SKLEARN
from sklearn.utils import shuffle
from sklearn.model_selection import train_test_split
import json

import cv2
#import matplotlib
#from matplotlib import pyplot as plt

# Get the output with maximum probability
import operator
import mychrome as chrome


# input image dimensions
img_rows, img_cols = 100, 100

# number of channels
# For grayscale use 1 value and for color images use 3 (R,G,B channels)
img_channels = 1


# Batch_size to train
batch_size = 32

## Number of output classes (change it accordingly)
## eg: In my case I wanted to predict 4 types of gestures (Ok, Peace, Punch, Stop)
## NOTE: If you change this then dont forget to change Labels accordingly
nb_classes = 2

# Number of epochs to train (change it accordingly)
nb_epoch = 25

# Total number of convolutional filters to use
nb_filters = 32
# Max pooling
nb_pool = 2
# Size of convolution kernel
nb_conv = 3

#%%
#  data
path = "./"

## Path2 is the folder which is fed in to training model
path2 = './imgfolder'

WeightFileName = ["newWeight.hdf5"]

# outputs
output = ["JUMP", "NOJUMP"]


#%%
def modlistdir(path):
    listing = os.listdir(path)
    retlist = []
    for name in listing:
        #This check is to ignore any hidden files/folders
        if name.startswith('.'):
            continue
        retlist.append(name)
    return retlist


# Load CNN model
def loadCNN(wf_index):
    global get_output
    model = Sequential()
    
    model.add(Conv2D(nb_filters, (nb_conv, nb_conv),
                        padding='valid',
                        input_shape=(img_channels, img_rows, img_cols)))
    convout1 = Activation('relu')
    model.add(convout1)
    model.add(Conv2D(nb_filters, (nb_conv, nb_conv)))
    convout2 = Activation('relu')
    model.add(convout2)
    model.add(MaxPooling2D(pool_size=(nb_pool, nb_pool)))
    model.add(Dropout(0.5))

    model.add(Flatten())
    model.add(Dense(128))
    model.add(Activation('relu'))
    model.add(Dropout(0.5))
    model.add(Dense(nb_classes))
    model.add(Activation('softmax'))
    
    model.compile(loss='categorical_crossentropy', optimizer='adadelta', metrics=['accuracy'])
    
    # Model summary
    model.summary()
    # Model conig details
    model.get_config()
    
    if wf_index >= 0:
        #Load pretrained weights
        fname = WeightFileName[int(wf_index)]
        print("loading ", fname)
        model.load_weights(fname)
    
    layer = model.layers[-1]
    get_output = K.function([model.layers[0].input, K.learning_phase()], [layer.output,])
    return model

# This function does the guessing work based on input images
def guessAction(model, img, chromebrowser):
    global output, get_output
    
    #Flatten it
    image = np.array(img).flatten()
  
    # reshape it
    #image = image.reshape(img_channels, img_rows,img_cols)
    
    # float32
    image = image.astype('float32') 
    
    # normalize it
    image = image / 255
    
    # reshape for NN
    rimage = image.reshape(1, img_channels, img_rows, img_cols)
    
    # Now feed it to the NN, to fetch the predictions
    #Method1-
    #index = model.predict_classes(rimage)
    #prob_array = model.predict_proba(rimage)
    
    #Method2-
    prob_array = get_output([rimage, 0])[0]

    d = {}
    i = 0
    for items in output:
        d[items] = prob_array[0][i] * 100
        i += 1
    
    # Get the output with maximum probability
    guess = max(d.items(), key=operator.itemgetter(1))[0]
    prob  = d[guess]
    if prob > 60.0:
        if guess == 'JUMP':
            chrome.dinojump(chromebrowser)
            return output.index(guess)
    else:
        return 1

#%%
def initializers():
    imlist = modlistdir(path2)
    
    image1 = np.array(Image.open(path2 +'/' + imlist[0])) # open one image to get size
    #plt.imshow(im1)
    
    m,n = image1.shape[0:2] # get the size of the images
    total_images = len(imlist) # get the 'total' number of images
    print( m,n)

    
    # create matrix to store all flattened images
    immatrix = np.array([np.array((Image.open(path2+ '/' + images).resize((img_rows,img_cols))).convert('L')).flatten()
                         for images in imlist], dtype = 'f')
    
    print(path2)
    print(immatrix.shape)
    
    raw_input("Press any key")
    
    #########################################################
    ## Label the set of images per respective gesture type.
    ##
    label=np.ones((total_images,),dtype = int)
    
#    samples_per_class = total_images / nb_classes
#    print "samples_per_class - ",samples_per_class
#    s = 0
#    r = samples_per_class
#    for classIndex in range(nb_classes):
#        label[s:r] = classIndex
#        s = r
#        r = s + samples_per_class

    # Following ranges are hardcoded !!!
    # So change them accordingly as per your sample data
    label[0:149]=0   # jump
    label[149:]=1    # nojump
    
    
    
    
    data,Label = shuffle(immatrix,label, random_state=2)
    train_data = [data,Label]
     
    (X, y) = (train_data[0],train_data[1])
     
     
    # Split X and y into training and testing sets
     
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.1, random_state=4)
     
    X_train = X_train.reshape(X_train.shape[0], img_channels, img_rows, img_cols)
    X_test = X_test.reshape(X_test.shape[0], img_channels, img_rows, img_cols)
     
    X_train = X_train.astype('float32')
    X_test = X_test.astype('float32')
     
    # normalize
    X_train /= 255
    X_test /= 255
     
    # convert class vectors to binary class matrices
    Y_train = np_utils.to_categorical(y_train, nb_classes)
    Y_test = np_utils.to_categorical(y_test, nb_classes)
    return X_train, X_test, Y_train, Y_test



def trainModel(model):

    # Split X and y into training and testing sets
    X_train, X_test, Y_train, Y_test = initializers()

    # Now start the training of the loaded model
    hist = model.fit(X_train, Y_train, batch_size=batch_size, epochs=nb_epoch,
                 verbose=1, validation_split=0.2)

    

    ans = raw_input("Do you want to save the trained weights - y/n ?")
    if ans == 'y':
        filename = raw_input("Enter file name - ")
        fname = path + str(filename) + ".hdf5"
        model.save_weights(fname,overwrite=True)
    else:
        model.save_weights("newWeight.hdf5",overwrite=True)

    visualizeHis(hist)
    # Save model as well
    # model.save("newModel.hdf5")
#%%

def visualizeHis(hist):
    # visualizing losses and accuracy

    train_loss=hist.history['loss']
    val_loss=hist.history['val_loss']
    train_acc=hist.history['acc']
    val_acc=hist.history['val_acc']
    xc=range(nb_epoch)

    plt.figure(1,figsize=(7,5))
    plt.plot(xc,train_loss)
    plt.plot(xc,val_loss)
    plt.xlabel('num of Epochs')
    plt.ylabel('loss')
    plt.title('train_loss vs val_loss')
    plt.grid(True)
    plt.legend(['train','val'])
    #print plt.style.available # use bmh, classic,ggplot for big pictures
    #plt.style.use(['classic'])

    plt.figure(2,figsize=(7,5))
    plt.plot(xc,train_acc)
    plt.plot(xc,val_acc)
    plt.xlabel('num of Epochs')
    plt.ylabel('accuracy')
    plt.title('train_acc vs val_acc')
    plt.grid(True)
    plt.legend(['train','val'],loc=4)

    plt.show()

