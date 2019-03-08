import gym
from gym import spaces
import rospy
import docker
try:
    import baxter_interface
except ImportError:
    print("Source the baxter_interface code for communicating with the Baxter")
from baxter_interface import CHECK_VERSION
from baxter_core_msgs.msg import AssemblyState
import numpy as np

import time
import os
import sys
import subprocess


class baxter_base(gym.GoalEnv):
    """
    The super class for the Baxter gym environment
    This code is a variation of the code in https://github.com/erlerobot/gym-gazebo
    """
    def __init__(self):
        # Initialize and run the docker
        client = docker.from_env()
        ROS_MASTER_URI="http://100.80.230.29:11311"


        self.cont1=client.containers.run("rosbaxter:2", # Name of the container
                              detach=True, # Detach the container
                              environment={"ROS_MASTER_URI":ROS_MASTER_URI}, # Assign ROS MASTER Node
                              volumes = {"/tmp/.gazebo/":{"bind":"/root/.gazebo/","mode":"rw"}},
                              publish_all_ports=True # Exposing all ports for communications
        )
        """
        self.cont1 = client.containers.run(
            "rosbaxter_view",
            detach=True,
            environment = {
                "DISPLAY":os.environ.get("DISPLAY"),
                "QT_X11_NO_MITSHM":1,
                "XAUTHORITY":os.environ.get("XAUTH"),
                "ROS_MASTER_URI":ROS_MASTER_URI
            },
            volumes={"/tmp/.X11-unix":{"bind":"/tmp/.X11-unix","mode":"rw"},"/tmp/.gazebo/":{"bind":"/root/.gazebo/","mode":"rw"}},
            publish_all_ports=True,
            runtime="nvidia"
        )
        """
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

        self.metadata ={
            'render.modes': ['human']
        }

        #Joint angles and joint velocity limits for normalization
        self.joint_min = [-1.7016,-2.147,-3.0541,-0.05,-3.059,-1.5707,-3.059]
        self.joint_max = [1.7016,1.047,3.0541,2.618,3.059,2.094,3.059]
        self.speed_lim = [2,2,2,2,4,4,4]
        self.effort_lim = [50,50,50,50,15,15,15]



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
    # A lambda function for normalization
    norm_val = lambda self,X,x_min,x_max:(X-x_min)*2/(x_max-x_min)-1
    def get_robot_obs(self):
        obs = [self.norm_val(self.limbR.joint_angle(joint),self.joint_min[i],self.joint_max[i]) for i,joint in enumerate(self.jointsR)]
        obs.extend([self.limbR.joint_velocity(joint)/self.speed_lim[i] for i,joint in enumerate(self.jointsR)])
        obs.extend([self.limbR.joint_effort(joint)/self.effort_lim[i] for i,joint in enumerate(self.jointsR)])
        obs.extend(self.get_xyz(self.limbR.endpoint_pose()['position']))
        # obs.extend(self.get_xyzw(self.limbR.endpoint_pose()['orientation']))
        # obs.extend(self.get_xyz(self.limbR.endpoint_velocity()['linear']))
        # obs.extend(self.get_xyz(self.limbR.endpoint_velocity()['angular']))
        # obs.extend(self.get_xyz(self.limbR.endpoint_effort()['force']))
        # obs.extend(self.get_xyz(self.limbR.endpoint_effort()['torque']))

        return np.asarray(obs)


    def step(self,action):
        raise NotImplementedError()

    def render(self):
        raise NotImplementedError()

    def _close(self):
        raise NotImplementedError()

    def _get_obs(self):
        raise NotImplementedError()

    def _set_goal(self):
        raise NotImplementedError()

    def close(self):
        self._close()
        # Disabling the robot
        self.rs.disable()

        # Shut down the docker
        self.cont1.stop()


    def _render(self):
        raise NotImplementedError
