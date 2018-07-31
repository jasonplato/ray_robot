import random
import learning_process
import exp
import task

epsilon=exp.EPSILON

def execute(s):
    if exp.ACTION_STRATEGY == "exploit":
        selected_action=exploit_policy(s)
    elif exp.ACTION_STRATEGY == "random":
        selected_action=random_action()
    elif exp.ACTION_STRATEGY == "egreedy":
        selected_action=egreedy(s,epsilon)
    return selected_action

def exploit_policy(s):
    selected_action=learning_process.policy[s]
    return selected_action

def random_action():
    selected_action = random.randint(0,task.n_actions-1)
    return selected_action

def egreedy(s,e):
    if random.random() < e:
        selected_action = random_action()
    else:
        selected_action = exploit_policy(s)
    return selected_action 
