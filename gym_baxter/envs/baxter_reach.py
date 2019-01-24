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


class BaxterReachEnv(baxter_base):
    """
    A Baxter class for the reach task, similar to the one to Fetch environments in OpenAI Gym
    """

    def __init__(self, reward_type='sparse'):
        super().__init__()
        # Get initial robot endeffector position

        temp = self.limbR.endpoint_pose()['position']
        self.initial_ef_pos = [temp.x, temp.y, temp.z]
        self.initial_ef_ori = self.limbR.endpoint_pose()['orientation']
        self.thresh = 0.05

        self.model_path= os.path.join(os.path.dirname(__file__),'assets')
        self.reward_type = reward_type
        self._set_goal()
        self.load_marker()

        end_effector_point= np.asarray(self.get_xyz(self.limbR.endpoint_pose()['position']))
        self.initial_goal_dis = np.linalg.norm(self.goal-end_effector_point)

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

    def load_marker(self):
        ref_frame = "baxter"
        block_pose = Pose(position=Point(
            x=self.goal[0],
            y=self.goal[1],
            z=self.goal[2]
        ))
        marker_xml = ''
        with open(os.path.join(self.model_path,"unit_box_0/model.sdf")) as marker_file:
            marker_xml = marker_file.read().replace('\n','')

            # Load the marker
            rospy.wait_for_service('/gazebo/spawn_sdf_model')
            try:
                spawn_sdf = rospy.ServiceProxy('/gazebo/spawn_sdf_model', SpawnModel)
                resp_sdf = spawn_sdf("unit_box_0", marker_xml, "/",block_pose, ref_frame)
            except rospy.ServiceException as e:
                rospy.logerr("Spawn URDF service call failed: {0}".format(e))

    def delete_marker(self):
        delete_model = rospy.ServiceProxy('/gazebo/delete_model', DeleteModel)
        resp_delete = delete_model("unit_box_0")

    def _set_goal(self):
        # Set a goal point
        # x = np.random.uniform(0.8,0.6,1)
        # y = np.random.uniform(-0.9,-0.8,1)
        # z = np.random.uniform(0.00,0.065,1)
        # z = np.asarray([0.056])
        x,y,z = [0.8,-0.7,0.056]
        self.goal = np.squeeze([x,y,z])
        # self.goal = self.initial_ef_pos + np.random.uniform(-0.5,0.5,size=3)

    def _get_obs(self):
        obs = self._get_robot_obs()
        # Normalize with respect to goal position
        obs[-3:]=obs[-3:]/self.initial_goal_dis
        achieved_goal = np.asarray(self.get_xyz(self.limbR.endpoint_pose()['position']))
        desired_goal = self.goal
        return {
            'observation':obs,
            'achieved_goal':achieved_goal,
            'desired_goal':desired_goal
        }

    def compute_reward(self, achieved_goal,goal,info):
        d = np.linalg.norm(self.goal-achieved_goal,axis=-1)
        if self.reward_type == 'sparse':
            return -(d>self.thresh).astype(np.float32)
        else:
            #Scale the reward with respect to initial goal position
            return -(d/self.initial_goal_dis)

    def _terminate(self,achieved_goal):
        d = np.linalg.norm(self.goal-achieved_goal)
        return d<self.thresh

    def _is_success(self,achieved_goal):
        d = np.linalg.norm(self.goal-achieved_goal,axis=-1)
        #return (d<self.thresh).astype(np.float32)
        return d

    def step(self,action):
        assert len(action)==4
        action = np.asarray([action[0],action[1],0.0,action[2],0.0,action[3],0.0])

        #Scale the action
        action = action*np.asarray([2,2,2,2,4,4,4])
        done = False
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
        info = {
            'is_success':self._is_success(obs['achieved_goal']),
        }
        reward = self.compute_reward(obs['achieved_goal'],obs['desired_goal'],info)
        # Augment the reward to penalize for fast actions
        reward = reward-np.linalg.norm(obs['observation'][7:14])
        done = False#self._terminate(obs['achieved_goal']) # get the endeffector point
        return obs,reward,done,info

    def reset(self):
        self.unpause()
        # Uncomment the following if training on the real robot
        # self.limbR.exit_control_mode()
        self.delete_marker()
        self.limbR.move_to_neutral()
        self._set_goal()
        self.load_marker()
        self.pause()

        return self._get_obs()

    def _close(self):
        self.unpause()
        self.delete_marker()
