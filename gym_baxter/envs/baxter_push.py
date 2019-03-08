import rospy
import rospkg
import baxter_interface
from baxter_interface import CHECK_VERSION
import os
from gym import spaces

import numpy as np
import time

# Loading inverse kinematics solver
from geometry_msgs.msg import (
    PoseStamped,
    Pose,
    Point,
    Quaternion,
)

from std_msgs.msg import Header
from std_srvs.srv import Empty


from baxter_core_msgs.srv import (
    SolvePositionIK,
    SolvePositionIKRequest,
)

from gazebo_msgs.srv import (
    SpawnModel,
    DeleteModel,
)

# Baxter environment base class
from gym_baxter.envs.core import baxter_base

class BaxterPushEnv(baxter_base):

    def __init__(self):
        super().__init__()

        self.model_path = os.path.join(os.path.dirname(__file__), 'assets')

        self._set_goal()
        self.load_marker()

        end_effector_point = np.asarray(self.get_xyz(self.limbR.endpoint_pose()['position']))
        self.initial_goal_dis = np.linalg.norm(self.goal - end_effector_point)

        # Environment attributes
        self._set_goal()
        obs = self._get_obs()
        self.action_space = spaces.Box(-1., 1., shape=(4,), dtype='float32')
        self.observation_space = spaces.Dict(dict(
            desired_goal=spaces.Box(-np.inf, np.inf, shape=obs['achieved_goal'].shape, dtype='float32'),
            achieved_goal=spaces.Box(-np.inf, np.inf, shape=obs['achieved_goal'].shape, dtype='float32'),
            observation=spaces.Box(-np.inf, np.inf, shape=obs['observation'].shape, dtype='float32'),
        ))

        # Simulation time
        self.step_size = 0.002
        self.num_step = 5
        self.pause = rospy.ServiceProxy('gazebo/pause_physics', Empty)
        self.unpause = rospy.ServiceProxy('gazebo/unpause_physics', Empty)
        self.pause()