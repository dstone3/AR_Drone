
# Python program for Detection of a  
# specific color(blue here) using OpenCV with Python 
import cv2 
import numpy as np  
  
# Webcamera no 0 is used to capture the frames 
cap = cv2.VideoCapture(0)  

target_color = np.uint8([[[220, 120, 140]]])
print(cv2.cvtColor(target_color, cv2.COLOR_BGR2HSV ))
# This drives the program into an infinite loop. 
while(1):        
    # Captures the live stream frame-by-frame 
    _, frame = cap.read()  
    # Converts images from BGR to HSV 
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV) 
    lower_red = np.array([90, 100, 160]) 
    upper_red = np.array([165, 205, 245]) 
  
    # Here we are defining range of bluecolor in HSV 
    # This creates a mask of blue coloured  
    # objects found in the frame. 
    mask = cv2.inRange(hsv, lower_red, upper_red) 
  
    # The bitwise and of the frame and mask is done so  
    # that only the blue coloured objects are highlighted  
    # and stored in res 
    res = cv2.bitwise_and(frame,frame, mask= mask) 



    cv2.imshow('mask',mask) 
    cv2.imshow('res',res)
    
    contours, hier = cv2.findContours(mask, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
    for cont in contours:
        (x,y,w,h) = cv2.boundingRect(cont)
        cv2.rectangle(frame, (x,y),(x+w,y+h),0,-1)
        
    # This displays the frame, mask  
    # and res which we created in 3 separate windows. 

    cv2.imshow('frame', frame)
    
    cv2.waitKey(5)
  
    # Destroys all of the HighGUI windows. 
cv2.destroyAllWindows() 
  
# release the captured frame 
cap.release() 
