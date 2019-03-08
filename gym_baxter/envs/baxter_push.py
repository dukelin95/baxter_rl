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

        # for loading in objects
        self.model_path = os.path.join(os.path.dirname(__file__), 'assets')

        self._set_goal() # TODO
        self._load_marker() # TODO

        end_effector_point = np.asarray(self.get_xyz(self.limbR.endpoint_pose()['position']))
        self.initial_goal_dis = np.linalg.norm(self.goal - end_effector_point)

        # Environment atrributes
        obs = self._get_obs() # TODO
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

    def step(self,action):
        assert len(action)==4
        action = np.asarray([action[0],action[1],0.0,action[2],0.0,action[3],0.0])

        #Scale the action
        action = action*np.asarray([2,2,2,2,4,4,4])
        action = dict(zip(self.limbR.joint_names(),action))
        # Start simulation
        self.unpause()
        # for _ in range(self.num_step):
        self.limbR.set_joint_velocities(action)
        time.sleep(self.num_step*self.step_size)

        # Stop simulation
        self.pause()
        # Collect state and terminal information
        obs = self._get_obs()

        reward = self._compute_reward(obs['achieved_goal'])
        # Augment the reward to penalize for fast actions
        reward = reward-np.linalg.norm(obs['observation'][7:14])
        #done = self._terminate(obs['achieved_goal']) # get the endeffector point
        return obs,reward

    def _compute_reward(self, achieved_goal):
        # TODO check return of _get_puck_location
        d = np.linalg.norm(self._get_puck_location() - achieved_goal,axis=-1)
        return -(d/self.initial_goal_dis)

    # TODO is the goal on table??
    def _set_goal(self):
        x, y, z = [0.8, -0.7, 0.056]
        self.goal = np.array([x, y, z])

    def _get_obs(self):
        obs = self.get_robot_obs()
        # Normalize with respect to goal position
        obs[-3:] = obs[-3:]/self.initial_goal_dis
        achieved_goal = np.asarray(self.get_xyz(self.limbR.endpoint_pose()['position']))
        desired_goal = self.goal
        return {
            'observation':obs,
            'achieved_goal':achieved_goal,
            'desired_goal':desired_goal
        }

    # TODO
    def _load_marker(self):
        ref_frame = "baxter"
        # TODO set to a location on the table
        #  also load in table
        #  also play with puck properties <inertia> http://gazebosim.org/tutorials?tut=build_model
        block_pose = Pose(position=Point(
            x=self.goal[0],
            y=self.goal[1],
            z=self.goal[2]
        ))

        # spawn
        with open(os.path.join(self.model_path,"unit_box_0/model.sdf")) as marker_file:
            marker_xml = marker_file.read().replace('\n','')

            # Load the marker
            rospy.wait_for_service('/gazebo/spawn_sdf_model')
            try:
                spawn_sdf = rospy.ServiceProxy('/gazebo/spawn_sdf_model', SpawnModel)
                resp_sdf = spawn_sdf("unit_box_0", marker_xml, "/",block_pose, ref_frame)
            except rospy.ServiceException as e:
                rospy.logerr("Spawn URDF service call failed: {0}".format(e))

    # TODO
    def _delete_marker(self):
        delete_model = rospy.ServiceProxy('/gazebo/delete_model', DeleteModel)
        resp_delete = delete_model("unit_box_0")

    def _get_puck_location(self):
        # TODO track puck location -> gazebo service http://gazebosim.org/tutorials/?tut=ros_comm
        raise NotImplementedError()
