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
        self.drone.useDemoMode(True)                            # Set 15 basic dataset/sec

        ##### Mainprogram begin #####
        self.drone.setConfigAllID()                              # Go to multiconfiguration-mode
        self.drone.hdVideo()
        self.drone.frontCam()                                    # Choose front view
        CDC = self.drone.ConfigDataCount
        while CDC==self.drone.ConfigDataCount: time.sleep(0.001) # Wait until it is done (after resync)

        self.IMC = None
        self.stop = False
        self.ground= False
        print("Drone initialization complete")

    def startVideo(self):
        drone.startVideo()
        #drone.showVideo()
        
        IMC =    drone.VideoImageCount # Number of encoded videoframes
        time.sleep(5)
        print("Video stream started")
        
    def findPerson(self, img):
        frame = img
        
        # if the frame dimensions are None, grab them
        (H,W) = frame.shape[:2]
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
    

stop = False

while not stop:
   
    thisDrone = ourDrone()
    thisDrone.startVideo()
    thisDrone.followPerson()
    #cv2.imshow("testFrame", img)
    
    trackPerson(img)
    
#    cv2.imshow("Frame", img)

    #cv2.waitKey(1) 
    
    ## move drone based off image here
