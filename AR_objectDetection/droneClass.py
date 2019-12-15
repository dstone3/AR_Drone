import time, sys
import ps_drone                                    # Import PS-Drone-API
import numpy as np
import cv2
import math
from pyimagesearch.centroidtracker import CentroidTracker

class ourDrone:

    def __init__(self):
        
        self.prototxt = "deploy.prototxt"
        self.model = "res10_300x300_ssd_iter_140000.caffemodel"
        self.ct = CentroidTracker()
        self.net = cv2.dnn.readNetFromCaffe(self.prototxt, self.model)
        self.classes = ["background", "aeroplane", "bicycle", "bird", "boat",
	           "bottle", "bus", "car", "cat", "chair", "cow", "diningtable",
	           "dog", "horse", "motorbike", "person", "pottedplant", "sheep",
	           "sofa", "train", "tvmonitor"]
        self.COLORS = np.random.uniform(0,255,size=(len(self.classes), 3))

        self.net2 = cv2.dnn.readNetFromCaffe("net.prototxt.txt", "netMode.caffemodel")

        self.drone = ps_drone.Drone()                           # Start using drone
        self.drone.startup()                                    # Connects to drone and starts subprocesses

        self.drone.reset()                                      # Sets drone's status to good
        while (self.drone.getBattery()[0]==-1): time.sleep(0.1) # Wait until drone has done its reset
        print "Battery: "+str(self.drone.getBattery()[0])+"% "+str(self.drone.getBattery()[1]) # Battery-status
        self.drone.useDemoMode(True) # Set 15 basic dataset/sec

        ##### Variables for states #####
        self.search = True

        ##### Mainprogram begin #####
        self.drone.setConfigAllID() # Go to multiconfiguration-mode
        self.drone.hdVideo()
        self.drone.groundCam()  # Choose ground view, alternative is frontCam()
        CDC = self.drone.ConfigDataCount
        while CDC==self.drone.ConfigDataCount: time.sleep(0.001) # Wait until it is done (after resync)

        self.IMC = None
        self.stop = False
        self.ground= False

        self.imgW = 0
        self.imgH = 0
        self.objBox = (0,0,0,0) # (x,y,x+w, y+h)
        print("Drone initialization complete")

    def startVideo(self):
        drone.startVideo()
        #drone.showVideo()
        
        IMC =    drone.VideoImageCount # Number of encoded videoframes
        time.sleep(5)
        print("Video stream started")

    def trackColor(self, img):
        
        hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV) 
        lower_red = np.array([0, 75, 40]) 
        upper_red = np.array([20, 255, 255]) 
  
        
        mask = cv2.inRange(hsv, lower_red, upper_red) 
        res = cv2.bitwise_and(frame,frame, mask= mask) 
        
        #cv2.imshow('mask',cv2.resize(mask,(800,800) )) 
        #cv2.imshow('res',res)
    
        contours, hier = cv2.findContours(mask, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)

        cont = max(contours, key = cv2.contourArea)
    
        (x,y,w,h) = cv2.boundingRect(cont)

        self.objBox = (x,y,w+x,h+y)
        cv2.rectangle(frame, (x,y),(x+w,y+h),0,-1)
        
        self.imgW = 600
        self.imgH = 600
        
        cv2.imshow('frame', cv2.resize(frame, (600,600)))
        key = cv2.waitKey(1) & 0xFF
    
    def findPerson(self, img):
        frame = img
        
        # if the frame dimensions are None, grab them
        (H,W) = frame.shape[:2]
        self.imgW = W
        self.imgH = H
	# construct a blob from the frame, pass it through the network,
	# obtain our output predictions, and initialize the list of
	# bounding box rectangles
        blob = cv2.dnn.blobFromImage(frame, 1.0, (W, H),(104.0, 177.0, 123.0))
        
        self.net.setInput(blob)
        detections = self.net.forward()
        rects = []
        
        
        # loop over the detections
        for i in range(0, detections.shape[2]):
	    # filter out weak detections by ensuring the predicted
	    # probability is greater than a minimum threshold
            if detections[0, 0, i, 2] > .5:
	        # compute the (x, y)-coordinates of the bounding box for
	        # the object, then update the bounding box rectangles list
                box = detections[0, 0, i, 3:7] * np.array([W, H, W, H])
                if i == 0: # only first detection
                    self.objBox = box
                rects.append(box.astype("int"))
                            
	        # draw a bounding box surrounding the object so we can
	        # visualize it
                (startX, startY, endX, endY) = box.astype("int")
                cv2.rectangle(frame, (startX, startY), (endX, endY),
			(0, 255, 0), 2)

	# update our centroid tracker using the computed set of bounding
	# box rectangles
        objects = self.ct.update(rects)
        
	# loop over the tracked objects
        for (objectID, centroid) in objects.items():
		# draw both the ID of the object and the centroid of the
		# object on the output frame
            text = "ID {}".format(objectID)
            cv2.putText(frame, text, (centroid[0] - 10, centroid[1] - 10),
            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
            cv2.circle(frame, (centroid[0], centroid[1]), 4, (0, 255, 0), -1)
            
    	# show the output frame
        cv2.imshow("Frame", frame)
        key = cv2.waitKey(1) & 0xFF

	

    def findObject(self, image):

        (h,w) = image.shape[:2]
        self.imgW = w
        self.imgH = h
        
        blob = cv2.dnn.blobFromImage(cv2.resize(image, (300, 300)), 0.007843, (300, 300), 127.5)

        # pass the blob through the network and obtain the detections and
        self.net2.setInput(blob)
        detections = self.net2.forward()
    
        for i in np.arange(0, detections.shape[2]):
            # extract the confidence (i.e., probability) associated with the
	    # prediction
            confidence = detections[0, 0, i, 2]
        
	    # filter out weak detections by ensuring the `confidence` is
	    # greater than the minimum confidence
            if confidence > .6:
                idx = int(detections[0, 0, i, 1])
                box = detections[0, 0, i, 3:7] * np.array([w, h, w, h])
                (startX, startY, endX, endY) = box.astype("int")
                self.objBox = box
                
		# display the prediction
                label = "{}: {:.2f}%".format(self.classes[idx], confidence * 100)
                print("[INFO] {}".format(label))
                cv2.rectangle(image, (startX, startY), (endX, endY),
		    self.COLORS[idx], 2)
                y = startY - 15 if startY - 15 > 15 else startY + 15
                cv2.putText(image, label, (startX, y),
			cv2.FONT_HERSHEY_SIMPLEX, 0.5, self.COLORS[idx], 2)
        cv2.imshow("Frame", frame)
        key = cv2.waitKey(1) & 0xFF      

    def move(self, side, front, vertical, turn):
        self.drone.move(side, front, vertical, turn)

    def center(self):
        img = self.drone.videoImage
        self.trackColor(img)

        #compute center
        x = ( self.objBox[0] + self.objBox[2] ) / 2.0
        y = ( self.objBox[1] + self.objBox[3] ) / 2.0
        x_error = x - 320
        y_error =  180 - y
        max_speed = 0.3
        x_speed = max_speed * x_error / 320.0
        y_speed = max_speed * y_error / 180.0

        # print statement
        print ("x: ", x_speed, "y: ", y_speed)

        #move drone
        #self.drone.move(x_speed, y_speed, 0, 0)



    def foundAlgorithm(self):
        ## Make movements based off on object's box location
        ## Algorithm assumes camera is ground camera
        ## Object should be corrected to be on the horizon but in some experimental x value
        
        ## Normalize coordinates, should range from 0-1
        objX = (self.objBox[2] - self.objBox[0]) / 2 / self.imgW
        objY = (self.objBox[3] - self.objBox[1]) / 2 / self.imgH
        
        pickupX = 50 / self.imgW ## experimental value, so needs to be changed
        pickupY = self.imgH / 2 ## horizon line
        
        
        
    def takeOff(self):
        x=1


    

## main function
if __name__ == '__main__':
    stop = False

    thisDrone = ourDrone()
    thisDrone.startVideo()
    #thisDrone.takeOff()

    while not stop:
        #img = drone.videoImage
        #thisDrone.followPerson(img) ## call sample object detection method
        #thisDrone.moveAlgorithm() ## Move drone...drone should take off before this..?

        thisDrone.center()
