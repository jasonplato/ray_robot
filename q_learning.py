import numpy as np
import agent
import learning_process as lp
import time

def execute():
    s=lp.s
    q=lp.q
    v=lp.v
    policy = lp.policy
    alpha = lp.alpha
    gamma = lp.gamma
    a=ap=agent.actionselection(s)
    agent.execute_action(a)
    time.sleep(lp.step_time)
    sp=agent.observestate()
    r=agent.get_reward()

    q[s,a]=q[s,a]+alpha*(r+gamma*q[sp,ap]-q[s,a])

    v[s]=np.max(q[s])
    policy[s]=np.argmax(q[s])
    lp.s = s
    lp.a = a
    lp.sp = sp
    lp.ap = ap
    lp.r = r
    lp.q = q
    lp.v = v
    lp.policy = policy

    return

