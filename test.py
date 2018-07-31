import  agent
import learning_process
import numpy as np
import time
import rl_vrep
import robot

rl_vrep.connect()
rl_vrep.start()
time.sleep(0.5)
agent.setup_task()
time.sleep(0.5)
agent.setup()
#robot.setup(["MOBILE_BASE", "DISTANCE_SENSOR"],[])
time.sleep(0.5)
policy = np.zeros((65536,1),dtype=np.int)

with open("policy_results.txt",'r') as f:
    for line in f:
        line = line.strip()
        if len(line) == 0: continue
        results = line.split(":")
        policy[int(results[0])]=int(results[1])

learning_process.policy = policy
print('policy:')
print(learning_process.policy)
i=0

while 1:
    s=agent.observestate()
    print('%d-s:' % (i),s)
    a=learning_process.policy[s]
    print('%d-a:' % (i),a)
    agent.execute_action(a)
    time.sleep(0.5)
    i+=1
