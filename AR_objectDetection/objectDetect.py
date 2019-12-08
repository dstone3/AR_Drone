import time, sys
import ps_drone                                    # Import PS-Drone-API
import numpy as np
import cv2
import math


def findObject(image):

    classes = ["background", "aeroplane", "bicycle", "bird", "boat",
	"bottle", "bus", "car", "cat", "chair", "cow", "diningtable",
	"dog", "horse", "motorbike", "person", "pottedplant", "sheep",
	"sofa", "train", "tvmonitor"]
    COLORS = np.random.uniform(0,255,size=(len(classes), 3))

    net = cv2.dnn.readNetFromCaffe("net.prototxt.txt", "netMode.caffemodel")

    (h,w) = image.shape[:2]
    blob = cv2.dnn.blobFromImage(cv2.resize(image, (300, 300)), 0.007843, (300, 300), 127.5)

    # pass the blob through the network and obtain the detections and
    net.setInput(blob)
    detections = net.forward()
    
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
            label = "{}: {:.2f}%".format(classes[idx], confidence * 100)
            print("[INFO] {}".format(label))
            cv2.rectangle(image, (startX, startY), (endX, endY),
		    COLORS[idx], 2)
            y = startY - 15 if startY - 15 > 15 else startY + 15
            cv2.putText(image, label, (startX, y),
			cv2.FONT_HERSHEY_SIMPLEX, 0.5, COLORS[idx], 2)
            
    return image


drone = ps_drone.Drone()                           # Start using drone
drone.startup()                                    # Connects to drone and starts subprocesses

drone.reset()                                      # Sets drone's status to good
while (drone.getBattery()[0]==-1): time.sleep(0.1) # Wait until drone has done its reset
print "Battery: "+str(drone.getBattery()[0])+"% "+str(drone.getBattery()[1]) # Battery-status
drone.useDemoMode(True)                            # Set 15 basic dataset/sec

##### Mainprogram begin #####
drone.setConfigAllID()                              # Go to multiconfiguration-mode
drone.hdVideo()                                     # Choose lower resolution (try hdVideo())
drone.frontCam()                                    # Choose front view
CDC = drone.ConfigDataCount
while CDC==drone.ConfigDataCount: time.sleep(0.001) # Wait until it is done (after resync)
drone.startVideo()                                  # Start video-function
drone.showVideo()                                   # Display the video

##### And action !
print "Use to toggle front- and groundcamera, any other key to stop"
IMC =    drone.VideoImageCount # Number of encoded videoframes
stop =   False
ground = False
while not stop:
    while drone.VideoImageCount==IMC: time.sleep(0.01) # Wait until the next video-frame
    IMC = drone.VideoImageCount
    key = drone.getKey() # dont really need the key
    image = drone.VideoImage

    image = findObject(image)
    
    cv2.imshow("Frame", image)
    #    cv2.waitKey(1) ## dont need this line because of the time.sleep on VideoImageCount
    
    ## move drone based off image here

    
    if key==" ":
        if ground:    ground = False
        else:         ground = True
        drone.groundVideo(ground)                    # Toggle between front- and groundcamera.
    elif key and key != " ": stop = True
