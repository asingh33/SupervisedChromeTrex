"""
Created on Thu Apr  6 01:01:43 2017

@author: abhisheksingh
"""

import time
import struct
import numpy as np
import cv2
import os

from pynput import keyboard
from PIL import ImageGrab,Image
from multiprocessing.pool import ThreadPool

import actionCNN as myNN
import mychrome as chrome


class ScreenCapture(object):
  
    numOfSamples = 300
    X = 30
    Y = 300
    width = 270
    height = 174
    
    @classmethod
    def getimage(self):
        img = ImageGrab.grab(bbox = (self.X, self.Y, self.width,self.height))
        img_np = np.array(img)
        
        #finalimg = cv2.cvtColor(img_np, cv2.COLOR_RGB2GRAY)
        return img_np

    @classmethod
    def getmssimage(self):
        import mss
        
        with mss.mss() as sct:
            mon = sct.monitors[1]
            
            L = mon["left"] + self.X
            T = mon["top"] + self.Y
            W = L + self.width
            H = T + self.height
            bbox = (L,T,W,H)
            #print(bbox)
            sct_img = sct.grab(bbox)

            img_pil = Image.frombytes('RGB', sct_img.size, sct_img.bgra, 'raw', 'BGRX')
            img_np = np.array(img_pil)
            #finalimg = cv2.cvtColor(img_np, cv2.COLOR_RGB2GRAY)
            return img_np
        
        

    @classmethod
    def saveROIImg(self, name,img,  counter):
        if counter <= self.numOfSamples:
            counter = counter + 1
            name = name + str(counter)
            print("Saving img:",name)
        cv2.imwrite("./imgfolder/"+name + ".png", img)
    
        return counter

    @classmethod
    def adjust(self,key):
        if key == ord('d'):
            self.X = self.X + 10
            print('X: ',self.X)
        elif key == ord('a'):
            self.X = self.X - 10
            print('X: ',self.X)

        if key == ord('w'):
            self.Y = self.Y - 10
            print('Y: ',self.Y)
        elif key == ord('s'):
            self.Y = self.Y + 10
            print('Y: ',self.Y)
            


# Globals
isEscape = False
saveImg = False
sc = ScreenCapture()
counter1 = 0
counter2 = 0
banner =  '''\nWhat would you like to do ?
    1- Use pretrained model for gesture recognition & layer visualization
    2- Train the model (you will require image samples for training under .\imgfolder. Use Option#3)
    3- Generate training image samples. Note: You need to be in 'sudo' i.e admin mode.
    '''

## These keyboard specific functions are used when user want to create new sample input images for traning.
# This function gets called when user presses any keyboard key
def on_press(key):
    global isEscape, saveImg, sc, counter1, counter2
    
    # Pressing 'UP arrow key' will initiate saving provided capture region images
    if key == keyboard.Key.up:
        saveImg = True
        #sc.capture()
        img = sc.getmssimage()
        counter1 = sc.saveROIImg("jump", img, counter1)
    
    # Pressing 'Right arrow key' will initiate saving provided capture region images
    if key == keyboard.Key.right:
        saveImg = True
        #sc.capture()
        img = sc.getmssimage()
        counter2 = sc.saveROIImg("nojump", img, counter2)

# This function gets called when user releases the keyboard key previously pressed
def on_release(key):
    global isEscape, saveImg, sc, counter1
    if key == keyboard.Key.esc:
        isEscape = True
        # Stop listener
        return False

# This function will create a keyboard listener to trace users keys while he/she is playing game
# in order to create new sample input images.
def listen():
    listener = keyboard.Listener(on_press = on_press,
                                 on_release = on_release)
    listener.start()


def main():
    global isEscape, saveImg, sc, counter2
 
    guess = False
    mod = 0
    
    #Call CNN model loading callback
    while True:
        ans = int(input( banner))
        if ans == 1:
            #mode = int(input("Aggresive Mode (0) or Conservative Mode (1)?"))
            mod = myNN.loadCNN(0)
            break
        elif ans == 2:
            mod = myNN.loadCNN(-1)
            myNN.trainModel(mod)
            raw_input("Press any key to continue")
            break
        elif ans == 3:
            listen()
            break
        else:
            print("Get out of here!!!")
            return 0

    driver = chrome.setup()
    chromebrowser = driver.find_element_by_tag_name('body')
    pool = ThreadPool(processes=30)
    cv2.namedWindow("ScreenCapture")
    cv2.moveWindow("ScreenCapture", 800,50)
    while True:
       
        img = sc.getmssimage()
        image = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        # Resize the image as per the Input Image size of our Neural Network
        rimage = cv2.resize(image, (myNN.img_rows, myNN.img_cols))
        
        if guess == True:
            # Method1: Call into directly the trained CNN's prediction
            #retvalue = myNN.guessAction(mod, rimage)
            
            # Method2: Call into tranined CNN's prediction using multithreaded approach
            #t = threading.Thread(target=myNN.guessAction, args = [mod, rimage,chromebrowser])
            #t.start()
            #t.join()
            
            # Method3: Call into trained CNN's prediction using multiprocessor/Threadpool approach
            ret = pool.apply_async(myNN.guessAction, (mod, rimage, chromebrowser)) # tuple of args for foo
            ret_action = ret.get()
            print('JUMP' if ret_action == 0 else 'NO JUMP')
            
        key = cv2.waitKey(1) & 0xff
        sc.adjust(key)

        # Exit
        if key == ord('q') or isEscape == True:
            break

        # Guess
        elif key == ord('g'):
           guess = not guess
           print("Prediction Mode - {}".format(guess))
        
        roi = cv2.resize(image, (320,232))
        cv2.imshow("ScreenCapture", roi)

    driver.quit()
    cv2.destroyAllWindows()


if __name__ == '__main__':
    main()



