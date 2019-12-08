# -*- coding: UTF-8 -*-

import olympe
from olympe.messages.ardrone3.Piloting import TakeOff, moveBy, Landing, CancelMoveBy
from olympe.messages.ardrone3.PilotingState import FlyingStateChanged
from olympe.messages.ardrone3.PilotingEvent import moveByEnd

from time import sleep
import cv2

class myDrone(object):
    """This provides a higher level interface with the anafi drone"""

    # ----------------------------------------------------------------
    # Constants for configuration
    CONNECT_SIMULATION = '10.202.0.1'
    CONNECT_PHYSICAL = '192.168.42.1'
    CONNECT_CONTROLLER = '192.168.42.1'

    # ----------------------------------------------------------------
    # Setup the drone 
    def __init__(self, connection, logLevel=1):
        """takes in a string and connects to physical or simulated drone""" 
        if (connection == self.CONNECT_CONTROLLER):
            # TODO
            xyz = 10
        else:
            self.drone = olympe.Drone(connection, loglevel=logLevel)
            self.drone.connection()

    # -----------------------------------------------------------------
    # Flying Functions

    def TakeOff(self):
        takeOffAction = self.drone(
                TakeOff()
                >> FlyingStateChanged(state="hovering", _timeout=5)
                ).wait()
        return takeOffAction

    def moveBy(self, x, y, z, v):
        moveAction = self.drone(
                moveBy(x, y, z, v)
                >> FlyingStateChanged(state="hovering", _timeout=5)
                ).wait()
        return moveAction

    def move(self, x, y, z, v):
        self.drone(moveBy(x, y, z, v))
        print("moved")

    def cancel(self):
        self.drone(CancelMoveBy())
        print("canceled move")

    def Landing(self):
        LandingAction = self.drone(
                Landing()
                ).wait()
        return LandingAction

    def get_flying_state(self):
        state = self.drone.get_state(FlyingStateChanged)["state"]
        print ("current state: ", state)
        return state

    def get_move_end(self):
        move_dict = self.drone.get_state(moveByEnd)
        x = move_dict["dX"]
        print("x: ", x)
        return move_dict

    # ------------------------------------------------------------------
    # Connection Functions

    def disconnection():
        self.drone.disconnection()

    # ------------------------------------------------------------------
    # Video Streaming Functions

    def start_stream(self):
        self.drone.set_streaming_callbacks(
                raw_cb=self.yuv_frame_cb
                )
        self.drone.start_video_streaming()

    def stop_stream(self):
        self.drone.stop_video_streaming()

    def yuv_frame_cb(self, yuv_frame):
        """
        This function is called for each YUV frame returned from the drone

            :type yuv_frame: olympe.VideoFrame
        """
        # return a dictionary with useful information
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

        # Use OpenCV to show this frame
        cv2.imshow("Olympe Streaming Example", cv2frame)
        cv2.waitKey(1)  # please OpenCV for 1 ms...










######################################################################
################        test code if main       ######################
######################################################################

if __name__ == "__main__":
    drone = myDrone(myDrone.CONNECT_SIMULATION) 
    drone.TakeOff()
    sleep(1)
    drone.start_stream()
    sleep(60)
    drone.stop_stream()
    drone.Landing()
    drone.disconnection()






