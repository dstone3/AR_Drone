import ps_drone
from droneClass import ourDrone

import time
import sys

## create drone object
drone = ourDrone()

####################################################################
########### main loop   ############################################
####################################################################

while True:
    # quit for get key
    if drone.drone.getKey():
        sys.exit()

    # move

