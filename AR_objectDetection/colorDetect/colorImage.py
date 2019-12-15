
# Python program for Detection of a  
# specific color(blue here) using OpenCV with Python 
import cv2 
import numpy as np  
  
# Webcamera no 0 is used to capture the frames 

target_color = np.uint8([[[240, 100, 75]]])
print(cv2.cvtColor(target_color, cv2.COLOR_BGR2HSV ))


# This drives the program into an infinite loop. 
frame = cv2.imread('test.jpg')
    # Converts images from BGR to HSV 
hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV) 
lower_red = np.array([0, 75, 40]) 
upper_red = np.array([20, 255, 255]) 
  
# Here we are defining range of bluecolor in HSV 
# This creates a mask of blue coloured  
# objects found in the frame. 
mask = cv2.inRange(hsv, lower_red, upper_red) 
  
# The bitwise and of the frame and mask is done so  
# that only the blue coloured objects are highlighted  
# and stored in res 
res = cv2.bitwise_and(frame,frame, mask= mask) 



cv2.imshow('mask',cv2.resize(mask,(800,800) )) 
#cv2.imshow('res',res)
    
contours, hier = cv2.findContours(mask, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)

cont = max(contours, key = cv2.contourArea)
    
(x,y,w,h) = cv2.boundingRect(cont)
print(w,h)
cv2.rectangle(frame, (x,y),(x+w,y+h),0,-1)
        
    # This displays the frame, mask  
    # and res which we created in 3 separate windows. 

cv2.imshow('frame', cv2.resize(frame, (600,600)))
cv2.waitKey(8000)

cv2.destroyAllWindows()
