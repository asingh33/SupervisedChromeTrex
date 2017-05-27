import time
import struct
import numpy as np
import cv2
import Quartz.CoreGraphics as CG
import os

from pynput import keyboard

import actionCNN as myNN
import time


class ScreenCapture(object):
  
    numOfSamples = 300
    
    @classmethod
    def capture(self):
        # Region size is hard coded, please change it
        # as per your need.
        X = 50
        Y = 160
        region = CG.CGRectMake(X, Y, 160, 116)
        
        if region is None:
            region = CG.CGRectInfinite
        else:
            # Capture region should be even sized else
            # you will see wrongly strided images i.e corrupted
            if region.size.width % 2 > 0:
                emsg = "Capture region width should be even (was %s)" % (region.size.width)
                raise ValueError(emsg)
        
        # Create screenshot as CGImage
        image = CG.CGWindowListCreateImage(
                                           region,
                                           CG.kCGWindowListOptionOnScreenOnly,
                                           CG.kCGNullWindowID,
                                           CG.kCGWindowImageDefault)
            
        # Intermediate step, get pixel data as CGDataProvider
        prov = CG.CGImageGetDataProvider(image)

        # Copy data out of CGDataProvider, becomes string of bytes
        self._data = CG.CGDataProviderCopyData(prov)
           
        # Get width/height of image
        self.width = CG.CGImageGetWidth(image)
        self.height = CG.CGImageGetHeight(image)
    
    
    @classmethod
    def getimage(self):
        imgdata=np.fromstring(self._data,dtype=np.uint8).reshape(len(self._data)/4,4)
        return imgdata[:self.width*self.height,:-1].reshape(self.height,self.width,3)
        

    @classmethod
    def saveROIImg(self, name,img,  counter):
        if counter <= self.numOfSamples:
            counter = counter + 1
            name = name + str(counter)
            print("Saving img:",name)
        cv2.imwrite("./imgfolder/"+name + ".png", img)
    
        return counter


# Globals
isEscape = False
saveImg = False
sp = ScreenCapture()
counter1 = 0
counter2 = 0
banner =  '''\nWhat would you like to do ?
    1- Use pretrained model for gesture recognition & layer visualization
    2- Train the model (you will require image samples for training under .\imgfolder)
    3- Generate training image samples. Note: You need to be in 'sudo' i.e admin mode.
    '''


# This function gets called when user presses any keyboard key
def on_press(key):
    global isEscape, saveImg, sp, counter1, counter2
    
    # Pressing 'UP arrow key' will initiate saving provided capture region images
    if key == keyboard.Key.up:
        saveImg = True
        sp.capture()
        img = sp.getimage()
        counter1 = sp.saveROIImg("jump", img, counter1)
    
    # Pressing 'Right arrow key' will initiate saving provided capture region images
    if key == keyboard.Key.right:
        saveImg = True
        sp.capture()
        img = sp.getimage()
        counter2 = sp.saveROIImg("nojump", img, counter2)

# This function gets called when user releases the keyboard key previously pressed
def on_release(key):
    global isEscape, saveImg, sp, counter1
    if key == keyboard.Key.esc:
        isEscape = True
        # Stop listener
        return False

def listen():
    listener = keyboard.Listener(on_press = on_press,
                                 on_release = on_release)
    listener.start()


def main():
    global isEscape, saveImg, sp, counter2
 
    guess = False
    lastAction = -1
    mod = 0
    
    
    #Call CNN model loading callback
    while True:
        ans = int(raw_input( banner))
        if ans == 1:
            mode = int(raw_input("Aggresive Mode (0) or Conservative Mode (1)?"))
            mod = myNN.loadCNN(mode)
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
            print "Get out of here!!!"
            return 0
    
    while True:
       
        sp.capture()
        img = sp.getimage()
        # Should we grayscale
        if myNN.img_channels == 1:
            image = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

        
        if guess == True:
            # Resize as per our need
            rimage = cv2.resize(image, (myNN.img_rows, myNN.img_cols))
            
            retvalue = myNN.guessAction(mod, rimage)
            
            if lastAction != retvalue:
                # This is specific to OSX, as I am using OSX action script
                # to send keyboard keys to Chrome app.
                # For Windows/Linux it will require some other way.
                if retvalue == 0:
                    jump = ''' osascript -e 'tell application "System Events" to key code 126' '''
                    os.system(jump)
                    #time.sleep(0.3)
                lastAction = retvalue
            
            print myNN.output[retvalue]
        
        cv2.imshow("ScreenCapture", image)
    
        key = cv2.waitKey(10) & 0xff

        # Exit
        if key == 27 or isEscape == True:
            break

        # Guess
        elif key == ord('g'):
           guess = not guess
           print "Prediction Mode - {}".format(guess)



    cv2.destroyAllWindows()


if __name__ == '__main__':
    main()



