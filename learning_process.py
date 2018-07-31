import numpy as np
import agent
import q_learning
import task
import exp

step_time = 0
step = -1  # Current learning step
a = -1  # Current action
s = -1  # current state
sp = -1  # state reached (s')
s0 = -1  # initial state
ap = -1  # next action selected (a')
r = 0  # current reward obtained (R)
alpha=exp.ALPHA
gamma = exp.GAMMA
n_states = -1
q=None
v= None
policy = None
sasr_step=None
ave_r_step = None  # Average R obtained per step

def setup():
    global step_time, step, s, sp, a, ap, r,s0, q, v, policy
    global ave_r_step,sasr_step,n_states
    agent.setup()
    #print('agent setup!')

    step_time=task.STEP_TIME / exp.SPEED_RATE

    step=0

    s=agent.observestate()

    s0=s
    sp=-1

    a= task.INITIAL_POLICY
    ap=-1
    r=0

    q=np.zeros((task.n_states,task.n_actions),dtype=np.float32)
    v=np.zeros(task.n_states,dtype=np.float64)
    policy = np.full(task.n_states, task.INITIAL_POLICY, dtype=np.uint32)
    sasr_step = np.zeros((exp.N_STEPS, 4))
    ave_r_step = np.zeros(exp.N_STEPS)
    n_states = task.n_states

    return 

def run():
    global step, s, sp, a, ap, r,sasr_step,n_states
    global q, v, policy,ave_r_step

    for step in range(0,exp.N_STEPS):
        #print('enter lp run!')
        q_learning.execute()
        #print('qlearning execute finish!')
        sasr_step[step,0]=s
        sasr_step[step,1]=a
        sasr_step[step,2]=sp
        sasr_step[step,3]=r
        if step ==0 :
            ave_r_step[step]= sasr_step[step,3]
        else :
            ave_r_step[step]=np.average(sasr_step[0:step,3])
        
        s=sp
        a=ap
        print('ave_r_step %d:%f' %(step,ave_r_step[step]))
        #print('states:',n_states)

    print('lp run finish!')
    with open('policy_results.txt','a') as f:
        for s in range(n_states):
            f.write(str(s)+":"+str(policy[s])+"\n")

    return 


