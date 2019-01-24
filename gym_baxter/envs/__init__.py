from gym_baxter.envs.baxter_reach import BaxterReachEnv

import os
import rostopic
import subprocess
import time

# Check to see if roscore is already running, if not then spin-up roscore



# Flag to see if roscore is running
roscore_running = False
try:
    rostopic.get_topic_class('/rosout')
    roscore_running = True
except rostopic.ROSTopicIOException as e:
    print("Did not find roscore running")

if not roscore_running:
    port = os.environ.get("ROS_PORT_SIM", "11311")
    ros_path = os.path.dirname(subprocess.check_output(["which", "roscore"]))
    roscore = subprocess.Popen(['/usr/bin/python', os.path.join(ros_path, b"roscore"), "-p", port],stdout=None)
    time.sleep(1)
    print("Roscore launched")

