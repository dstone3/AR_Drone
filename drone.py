# -*- coding: UTF-8 -*-

import olympe
from olympe.messages.ardrone3.Piloting import TakeOff, moveBy, Landing
from olympe.messages.ardrone3.PilotingState import FlyingStateChanged

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

    def Landing(self):
        LandingAction = self.drone(
                Landing()
                ).wait()
        return LandingAction

    # ------------------------------------------------------------------
    # Connection Functions

    def disconnection():
        self.drone.disconnection()










######################################################################
################        test code if main       ######################
######################################################################

if __name__ == "__main__":
    drone = myDrone(myDrone.CONNECT_SIMULATION) 
    drone.TakeOff()
    drone.moveBy(10,0,0,0)
    drone.Landing()
    drone.disconnection()






