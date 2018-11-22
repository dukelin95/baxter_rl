from baxter_reach import baxter_reach

import time


Env = baxter_reach()

action = [0.1]*7

start_time= time.time()
for _ in range(1):
    obs,r,done,_=Env.step(action)
    # print(r,done)

print(time.time()-start_time)


Env.reset()


