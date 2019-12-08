import olympe
from olympe.messages.ardrone3.Piloting import TakeOff, Landing, moveBy
from olympe.messages.ardrone3.PilotingState import FlyingStateChanged
from olympe.messages.ardrone3.PilotingSettings import MaxTilt
from olympe.messages.ardrone3.GPSSettingsState import GPSFixStateChanged
from olympe.messages import gimbal
import csv
import cv2
import math
import os
import shlex
import subprocess
import tempfile
import numpy as np
import time
import imutils
from pyimagesearch.centroidtracker import CentroidTracker
    


class StreamingExample:

    def __init__(self):
        # Create the olympe.Drone object from its IP address
        self.drone = olympe.Drone(
            #"192.168.42.1",
            "10.202.0.1",
            loglevel=3,
        )
        self.tempd = tempfile.mkdtemp(prefix="olympe_streaming_test_")
        print("Olympe streaming example output dir: {}".format(self.tempd))

        self.stopFlying = 0
        
        self.objectDetected = 0
        self.distance = 0

        self.h264_frame_stats = []
        self.h264_stats_file = open(
            os.path.join(self.tempd, 'h264_stats.csv'), 'w+')
        self.h264_stats_writer = csv.DictWriter(
            self.h264_stats_file, ['fps', 'bitrate'])
        self.h264_stats_writer.writeheader()
        self.drone(FlyingStateChanged(state="landed"))

        self.prototxt = "deploy.prototxt"
        self.model = "res10_300x300_ssd_iter_140000.caffemodel"
        self.ct = CentroidTracker()
        
        self.H = None
        self.W = None
        self.net = cv2.dnn.readNetFromCaffe(self.prototxt, self.model)
        self.cv2image = None
        self.objectX = -1
        self.objectY = -1
        self.centerX = 360
        self.centerY = 640
        self.initBox = None
        self.currBox = None
        self.firstRun = True
        
    def start(self):
        # Connect the the drone
        self.drone.connection()

        # You can record the video stream from the drone if you plan to do some
        # post processing.
        self.drone.set_streaming_output_files(
            h264_data_file=os.path.join(self.tempd, 'h264_data.264'),
            h264_meta_file=os.path.join(self.tempd, 'h264_metadata.json'),
            # Here, we don't record the (huge) raw YUV video stream
            # raw_data_file=os.path.join(self.tempd,'raw_data.bin'),
            # raw_meta_file=os.path.join(self.tempd,'raw_metadata.json'),
        )

        # Setup your callback functions to do some live video processing
        self.drone.set_streaming_callbacks(
            raw_cb=self.yuv_frame_cb,
            h264_cb=self.h264_frame_cb
        )
        # Start video streaming
        self.drone.start_video_streaming()
        
#        self.drone(olympe.messages.gimbal.reset_orientation(gimbal_id=0)).wait()

       

    def stop(self):
        # Properly stop the video stream and disconnect
        self.drone.stop_video_streaming()
        self.drone.disconnection()
        self.h264_stats_file.close()

    def yuv_frame_cb(self, yuv_frame):
        """
        This function will be called by Olympe for each decoded YUV frame.

            :type yuv_frame: olympe.VideoFrame
        """
        # the VideoFrame.info() dictionary contains some useful informations
        # such as the video resolution
        info = yuv_frame.info()
        height, width = info["yuv"]["height"], info["yuv"]["width"]
        
        # convert pdraw YUV flag to OpenCV YUV flag
        cv2_cvt_color_flag = {
            olympe.PDRAW_YUV_FORMAT_I420: cv2.COLOR_YUV2BGR_I420,
            olympe.PDRAW_YUV_FORMAT_NV12: cv2.COLOR_YUV2BGR_NV12,
        }[info["yuv"]["format"]]

        # yuv_frame.as_ndarray() is a 2D numpy array with the proper "shape"
        # i.e (3 * height / 2, width) because it's a YUV I420 or NV12 frame

        # Use OpenCV to convert the yuv frame to RGB
        cv2frame = cv2.cvtColor(yuv_frame.as_ndarray(), cv2_cvt_color_flag)

        
        frame = imutils.resize(cv2frame, width=1000)
        
        #frame = cv2frame
        # if the frame dimensions are None, grab them
        if self.W is None or self.H is None:
            (self.H, self.W) = frame.shape[:2]

	# construct a blob from the frame, pass it through the network,
	# obtain our output predictions, and initialize the list of
	# bounding box rectangles
        blob = cv2.dnn.blobFromImage(frame, 1.0, (self.W, self.H),(104.0, 177.0, 123.0))
        
        self.net.setInput(blob)
        detections = self.net.forward()
        rects = []
        #print("detections = ", detections)
        
    # loop over the detections
        for i in range(0, detections.shape[2]):
	# filter out weak detections by ensuring the predicted
	# probability is greater than a minimum threshold
            if detections[0, 0, i, 2] > .5:
	    # compute the (x, y)-coordinates of the bounding box for
	    # the object, then update the bounding box rectangles list
                box = detections[0, 0, i, 3:7] * np.array([self.W, self.H, self.W, self.H])
                rects.append(box.astype("int"))
                if i == 0 and self.firstRun:
                    self.initBox = box.astype("int")
                self.currBox = box
                self.firstRun = False
                print("box= ")
                print(box.astype("int"))
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
            if(objectID == 0):
                self.objectX = centroid[0]
                self.objectY = centroid[1]
    	# show the output frame
        cv2.imshow("Frame", frame)
        key = cv2.waitKey(1) & 0xFF

	# if the `q` key was pressed, break from the loop
        if key == ord("q"):
            self.stopFlying = 1
        
          
        
        # Use OpenCV to show this frame
        #cv2.imshow("Olympe Streaming Example", cv2frame)
        #cv2.waitKey(1)  # please OpenCV for 1 ms...
        #key = cv2.waitKey(1) & 0xFF

	# if the `q` key was pressed, break from the loop
        #if key == ord("q"):
        #    self.stopFlying = 1
        
    def h264_frame_cb(self, h264_frame):
        """
        This function will be called by Olympe for each new h264 frame.

            :type yuv_frame: olympe.VideoFrame
        """

        # Get a ctypes pointer and size for this h264 frame
        frame_pointer, frame_size = h264_frame.as_ctypes_pointer()

        # For this example we will just compute some basic video stream stats
        # (bitrate and FPS) but we could choose to resend it over an another
        # interface or to decode it with our preferred hardware decoder..

        # Compute some stats and dump them in a csv file
        #info = h264_frame.info()
        #frame_ts = info["ntp_raw_timestamp"]
        #if not bool(info["h264"]["is_sync"]):
        #    if len(self.h264_frame_stats) > 0:
        #        while True:
        #            start_ts, _ = self.h264_frame_stats[0]
        #            if (start_ts + 1e6) < frame_ts:
        #                self.h264_frame_stats.pop(0)
        #            else:
        #                break
        #    self.h264_frame_stats.append((frame_ts, frame_size))
        #    h264_fps = len(self.h264_frame_stats)
        #    h264_bitrate = (
        #        8 * sum(map(lambda t: t[1], self.h264_frame_stats)))
        #    self.h264_stats_writer.writerow(
        #        {'fps': h264_fps, 'bitrate': h264_bitrate})

    def boxCalc(self):

        if not self.firstRun:
            initBoxsize = (self.initBox[1] - self.initBox[0]) * (self.initBox[3] - self.initBox[2])
            newBoxsize = (self.currBox[1] - self.currBox[0]) * (self.currBox[3] - self.currBox[2])
            print(initBoxsize, " ", newBoxsize)
            
            return (newBoxsize - initBoxsize) / 500
        
        
    def fly(self):
        # Takeoff, fly, land, ...
        self.drone(
            TakeOff(_no_expect=True)
            & FlyingStateChanged(
                state="hovering", _timeout=10, _policy="wait")
            
        ).wait()
        
#        time.sleep(5)
    
        self.drone(gimbal.set_target(
           gimbal_id=0,
           control_mode="position",
           yaw_frame_of_reference="none",   # None instead of absolute
           yaw=0.0,
           pitch_frame_of_reference="absolute",
           pitch=0,
           roll_frame_of_reference="none",     # None instead of absolute
           roll=0.0,
        )).wait()
        
        #self.drone(moveBy(0,-.8,0,0, _timeout=20)).wait().success() forward/-backward, right/-left, down/-up
        
        while self.stopFlying == 0:
            yDisplacement = -(self.objectY - self.centerY) / 1280
            rotDisplacement = (self.objectX - self.centerX) / 1420
            zDisplacement = self.boxCalc()

            yDisplacement = 0
            zDisplacement = 0
            #rotDisplacement = 0

            if yDisplacement > 1:
                yDisplacement = 0
            if zDisplacement > .5:
                zDisplacement = 0
            if rotDisplacement > .5:
                rotDisplacement = 0

                
            if(self.objectX != -1):
                print(yDisplacement, "y, rot = ", rotDisplacement, " z= ", zDisplacement)
                print(self.objectX, " ", self.objectY)
                print(self.centerX, " ", self.centerY)
                
                
            if not self.firstRun:
                self.drone(moveBy(zDisplacement, 0, yDisplacement, rotDisplacement)).wait().success() 
            
            
        self.drone(
            Landing()
            >> FlyingStateChanged(state="landed", _timeout=5)
        ).wait()
        print("Landed\n")

    def postprocessing(self):
        # Convert the raw .264 file into an .mp4 file
        h264_filepath = os.path.join(self.tempd, 'h264_data.264')
        mp4_filepath = os.path.join(self.tempd, 'h264_data.mp4')
        subprocess.run(
            shlex.split('ffmpeg -i {} -c:v copy {}'.format(
                h264_filepath, mp4_filepath)),
            check=True
        )

                       
if __name__ == "__main__":
    streaming_example = StreamingExample()
    # Start the video stream
    streaming_example.start()
    # Perform some live video processing while the drone is flying
    streaming_example.fly()

    # Stop the video stream
    streaming_example.stop()
    
    # Recorded video stream postprocessing
    streaming_example.postprocessing()
