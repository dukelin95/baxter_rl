# from baxter_reach import baxter_reach
import gym_baxter
from baxter_reach import BaxterReachEnv
import numpy as np
import time
import math
import random
import gym


joint_min = [-1.7016,-2.147,-3.0541,-0.05,-3.059,-1.5707,-3.059]
joint_max = [1.7016,1.047,3.0541,2.618,3.059,2.094,3.059]
unnorm_val = lambda Y,x_min,x_max:((Y+1)*(x_max - x_min)*0.5) + x_min

# Env = gym.make('BaxterReachDense-v0')
Env = BaxterReachEnv()

# sleep to prevent failed initialization (todo check on why it hangs on move_to_neutral)
time.sleep(1)
Env.reset()

def remap_joints(jumbled):
    # np.array(list(Env.limbR.joint_angles().keys()))
    # array(['right_e0', 'right_e1', 'right_s0', 'right_s1', 'right_w0',
    #        'right_w1', 'right_w2'], dtype='<U8')
    corr = np.array([jumbled['right_s0'], jumbled['right_s1'],
                     jumbled['right_e0'], jumbled['right_e1'],
                     jumbled['right_w0'], jumbled['right_w1'], jumbled['right_w2']])
    return corr

def position_move():
    pos = {'right_s0': 0.5624933674210334, 'right_s1': -0.7845044836167014, 'right_e0': -0.3603666161670889,
     'right_e1': 1.4797406649817513, 'right_w0': 0.14539719383497565, 'right_w1': 0.36563868778521813,
     'right_w2': 0.25149351849122836}
    Env.reset()
    Env.unpause()
    Env.limbR.move_to_joint_positions(pos)
    Env.pause()


def pid_test_control(active=[1,1,1,1], Kps=[1, 1, 5, 10], Tis=[math.inf, math.inf, math.inf, math.inf], Tds=[0, 0, 0, 0], trials=100000):
    obs = Env.reset()
    Env.unpause()

    pos_points = Env.limbR.endpoint_pose()['position']
    curr_xyz = np.array([pos_points.x, pos_points.y, pos_points.z])
    curr_joint_angle = remap_joints(Env.limbR.joint_angles())

    print("Start angles: ", end='')
    print(curr_joint_angle[0], curr_joint_angle[1], curr_joint_angle[3], curr_joint_angle[5])

    goal_xyz = obs['desired_goal']
    goal_joint_angle = Env.ik_req(goal_xyz)
    if not goal_joint_angle:
        Env.close()
        exit()
    print(goal_joint_angle['right_s0'], goal_joint_angle['right_s1'], goal_joint_angle['right_e1'],goal_joint_angle['right_w1'])
    # joints_considered = np.array([1, 1, 0, 1, 0, 1, 0])
    joints_considered = np.array([active[0], active[1], 0, active[2], 0, active[3], 0])
    Kp = np.array([Kps[0], Kps[1], 0, Kps[2], 0, Kps[3], 0])
    Ti = np.array([Tis[0], Tis[1], math.inf, Tis[2], math.inf, Tis[3], math.inf])
    Td = np.array([Tds[0], Tds[1], 0, Tds[2], 0, Tds[3], 0])
    iteration = 0
    track = time.time()
    acc_err = np.zeros(7)
    pre_err = (np.asarray([(goal_joint_angle['right_s0'] - curr_joint_angle[0]),
                           (goal_joint_angle['right_s1'] - curr_joint_angle[1]), 0,
                           (goal_joint_angle['right_e1'] - curr_joint_angle[3]), 0,
                           (goal_joint_angle['right_w1'] - curr_joint_angle[5]), 0]) * joints_considered)
    graph = [[] for _ in range(5)]
    for _ in range(trials):

        graph[0].append(curr_joint_angle[0])
        graph[1].append(curr_joint_angle[1])
        graph[2].append(curr_joint_angle[3])
        graph[3].append(curr_joint_angle[5])
        graph[4].append(time.time())

        error = (np.asarray(
            [(goal_joint_angle['right_s0'] - curr_joint_angle[0]),
             (goal_joint_angle['right_s1'] - curr_joint_angle[1]),
             0,
             (goal_joint_angle['right_e1'] - curr_joint_angle[3]),
             0,
             (goal_joint_angle['right_w1'] - curr_joint_angle[5]),
             0]
        ) * joints_considered)

        acc_err = acc_err + (error * (time.time()-track))
        der = (pre_err - error)/track
        track = time.time()

        action = Kp * (error + ((1/Ti) * acc_err) + (Td * der))
        action = dict(zip(Env.limbR.joint_names(), action))
        Env.limbR.set_joint_velocities(action)

        pos_points = Env.limbR.endpoint_pose()['position']
        curr_xyz = np.array([pos_points.x, pos_points.y, pos_points.z])
        curr_joint_angle = remap_joints(Env.limbR.joint_angles())

        if iteration % 1000 == 0:
            # print("current xyz: " + str(curr_xyz))
            print("---", end ='')
            # print(monitor[0], monitor[1], monitor[3], monitor[5])
            print(curr_joint_angle[0], curr_joint_angle[1], curr_joint_angle[3], curr_joint_angle[5])
            print('---')
        iteration += 1

    Env.pause()
    print(curr_xyz, np.linalg.norm(curr_xyz - goal_xyz))
    np.save('errors', np.array(graph))


def p_test_control(K = 0.1, trials = 10000):
    obs = Env.reset()
    Env.unpause()

    goal_xyz = obs['desired_goal']
    pos_points = Env.limbR.endpoint_pose()['position']
    curr_xyz = np.array([pos_points.x, pos_points.y, pos_points.z])

    curr_joint_angle = remap_joints(Env.limbR.joint_angles())
    # print("Goal XYZ: " + str(goal_xyz))
    print("Start angles: ", end='')
    print(curr_joint_angle[0], curr_joint_angle[1], curr_joint_angle[3], curr_joint_angle[5])
    goal_joint_angle = Env.ik_req(goal_xyz)
    if not goal_joint_angle:
        Env.close()
        exit()
    # goal_joint_angle = {'right_s0': 0.0005955154106877078, 'right_s1': -0.8500784216756604,
    #  'right_e0': -0.0009278224313717157, 'right_e1': 1.6036966395875676,
    #  'right_w0': -0.0014741221073159346, 'right_w1': 0.24205723589822334, 'right_w2': 0.00166156136501434}
    # only using right_s0, right_s1, right_e1, right_w1

    # 7 DoF
    iteration = 1
    # while not (np.round(curr_xyz, 3) == goal_xyz).all():
    while not (np.round(curr_joint_angle[0], 2) == np.round(goal_joint_angle['right_s0'], 2)):
    # for _ in range(trials):
        # action = (curr_joint_angle - np.array(list(goal_joint_angle.values()))) * K
        # action = K * (np.asarray(
        #     [(curr_joint_angle[0] - goal_joint_angle['right_s0']) * 1,
        #      (curr_joint_angle[1] - goal_joint_angle['right_s1']) * 0,
        #      (curr_joint_angle[2] - goal_joint_angle['right_e0']) * 0,
        #      (curr_joint_angle[3] - goal_joint_angle['right_e1']) * 0,
        #      (curr_joint_angle[4] - goal_joint_angle['right_w0']) * 0,
        #      (curr_joint_angle[5] - goal_joint_angle['right_w1']) * 0,
        #      (curr_joint_angle[6] - goal_joint_angle['right_w2']) * 0]
        # ))
        action = K * (np.asarray(
            [(goal_joint_angle['right_s0'] - curr_joint_angle[0]) * 1,
             (goal_joint_angle['right_s1'] - curr_joint_angle[1]) * 0,
             0,
             (goal_joint_angle['right_e1'] - curr_joint_angle[3]) * 0,
             0,
             (goal_joint_angle['right_w1'] - curr_joint_angle[5]) * 0,
             0]
        ))

        monitor = action
        action = dict(zip(Env.limbR.joint_names(),action))

        Env.limbR.set_joint_velocities(action)

        # obs = Env._get_obs()
        curr_joint_angle = remap_joints(Env.limbR.joint_angles())
        # curr_joint_angle = np.array([unnorm_val(obs['observation'][i], joint_min[i], joint_max[i]) for i, _ in enumerate(joint_min)])
        # curr_xyz = obs['observation'][-3:]
        pos_points = Env.limbR.endpoint_pose()['position']
        curr_xyz = np.array([pos_points.x, pos_points.y, pos_points.z])

        if iteration % 1000 == 0:
            # print("current xyz: " + str(curr_xyz))
            print("---", end ='')
            # print(monitor[0], monitor[1], monitor[3], monitor[5])
            print(curr_joint_angle[0], curr_joint_angle[1], curr_joint_angle[3], curr_joint_angle[5])
            print('---')
        iteration += 1
    Env.pause()





