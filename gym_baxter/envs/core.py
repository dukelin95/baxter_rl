import gym
import rospy
import docker
import baxter_interface
from baxter_interface import CHECK_VERSION
from baxter_core_msgs.msg import AssemblyState

import time
import os
import sys
import subprocess


class baxter_base(gym.Env):
    """
    The super class for the Baxter gym environment
    This code is a variation of the code in https://github.com/erlerobot/gym-gazebo
    """
    def __init__(self):
        # TODO: Separate roscore initialization for launching multiple instances
        self.port = os.environ.get("ROS_PORT_SIM", "11311")
        ros_path = os.path.dirname(subprocess.check_output(["which", "roscore"]))

        # start roscore with python2 instead of python3
        self._roscore = subprocess.Popen(['/usr/bin/python', os.path.join(ros_path, b"roscore"), "-p", self.port])
        time.sleep(2)
        print ("Roscore launched!")

        # Initialize and run the docker
        client = docker.from_env()
        ROS_MASTER_URI="http://100.80.231.101:11311"
        self.cont1=client.containers.run("rosbaxter:2", # Name of the container
                              detach=True, # Detach the container
                              environment={"ROS_MASTER_URI":ROS_MASTER_URI}, # Assign ROS MASTER Node
                              publish_all_ports=True # Exposing all ports for communications
        )

        rospy.init_node("BaxterControl",anonymous=True)

        self.baxter_state = None
        state_topic='robot/state'
        rospy.Subscriber(state_topic, AssemblyState, self.state_callback)
        check = lambda : self.baxter_state is None
        # Wait for Baxter to spawn
        print("Waiting for Baxter to spawn")
        while check():
            time.sleep(1)
        print("Baxter is loaded")

        self.enable_baxter()


    def state_callback(self,msg):
         self.baxter_state = msg

    def enable_baxter(self):
        # Getting robot state
        self.rs = baxter_interface.RobotEnable(True)
        self.init_state = self.rs.state().enabled

        # Enable the robot
        print("Enabling robot... ")
        self.rs.enable()

        # Initializing the right arm
        self.limbR = baxter_interface.Limb("right")
        self.jointsR = self.limbR.joint_names()

    # A function to convert Point to list
    get_xyz = lambda self,X:[X.x,X.y,X.z]
    # A function to convert Quaternion to list
    get_xyzw = lambda self,X:[X.x,X.y,X.z,X.w]

    def _get_robot_obs(self):
        obs = [self.limbR.joint_angle(joint) for joint in self.jointsR]
        obs.extend([self.limbR.joint_velocity(joint) for joint in self.jointsR])
        obs.extend([self.limbR.joint_effort(joint) for joint in self.jointsR])
        obs.extend(self.get_xyz(self.limbR.endpoint_pose()['position']))
        obs.extend(self.get_xyzw(self.limbR.endpoint_pose()['orientation']))
        obs.extend(self.get_xyz(self.limbR.endpoint_velocity()['linear']))
        obs.extend(self.get_xyz(self.limbR.endpoint_velocity()['angular']))
        obs.extend(self.get_xyz(self.limbR.endpoint_effort()['force']))
        obs.extend(self.get_xyz(self.limbR.endpoint_effort()['torque']))

        return obs


    def step(self,action):
        raise NotImplementedError

    def render(self):
        raise NotImplementedError

    def _close(self):
        raise NotImplementedError

    def close(self):
        self._close()
        # Disabling the robot
        self.rs.disable()

        # Shut down the docker
        self.cont1.stop()

        # Kill roscore
        self._roscore.terminate()

    def _render(self):
        raise NotImplementedError