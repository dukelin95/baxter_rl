import rospy
import rospkg
import baxter_interface
from baxter_interface import CHECK_VERSION
import os

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
        self.thresh = 0.001

        self.reward_type = reward_type
        self._set_goal()
        self.load_marker()
        self.model_path= os.path.join(os.path.dirname(__file__),'assets')

        # Simulation time
        self.step_size = 0.002
        self.num_step = 10
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
        with open(os.path.join(os.path.dirname(__file__),'assets',"unit_box_0/model.sdf")) as marker_file:
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
        x = np.random.uniform(0.7,1,1)
        y = np.random.uniform(-0.5,-0.07,1)
        z = np.random.uniform(0.00,0.065,1)
        self.goal = np.squeeze([x,y,z])
        # self.goal = self.initial_ef_pos + np.random.uniform(-0.5,0.5,size=3)

    def _get_obs(self):
        obs = self._get_robot_obs()
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
            return -d

    def _terminate(self,achieved_goal):
        d = np.linalg.norm(self.goal-achieved_goal)
        return d<self.thresh

    def step(self,action):
        assert len(action)==4
        action = np.asarray([action[0],action[1],0.0,action[2],0.0,action[3],0.0])

        #Scale the action
        action = action*np.asarray([2,2,2,2,4,4,4])
        done = False
        action = dict(zip(self.limbR.joint_names(),action))
        # Start simulation
        self.unpause()
        for _ in range(self.num_step):
            self.limbR.set_joint_velocities(action)

        # Stop simulation
        self.pause()
        # Collect state and terminal information
        obs = self._get_obs()
        info = {}
        reward = self.compute_reward(obs['achieved_goal'],obs['desired_goal'],info)
        done = self._terminate(obs['achieved_goal']) # get the endeffector point
        return obs,reward,done,{}

    def reset(self):
        self.unpause()
        self.delete_marker()
        self.limbR.move_to_neutral()
        self._set_goal()
        self.load_marker()
        self.pause()

        return self._get_obs()

    def _close(self):
        self.unpause()
        self.delete_marker()
