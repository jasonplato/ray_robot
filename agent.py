import numpy as np
import robot
import task
import action_selection

n_inputs = None
in_values = [None]
in_sizes = [None]
n_outputs = None
out_values = [None]
out_sizes = [None]

n_states = None
n_actions = None
#initiated = False

sub_s = np.empty(0)
sub_a = np.empty(0)
in_to_states = np.empty(0)
_count = np.empty(0)


def setup_task():
    global n_inputs, in_values, n_outputs, out_values, in_sizes, out_sizes, n_states, n_actions
    
    inputvar = task.INPUT_VARIABLES
    outputvar = task.OUTPUT_VARIABLES

    n_inputs = len(inputvar)
    in_values = n_inputs * [None]
    in_names = n_inputs * [None]
    in_sizes = n_inputs * [int]

    i = 0
    for key, value in inputvar.items():
        in_names[i] = key
        in_values[i] = value
        in_sizes[i] = len(value)
        i += 1
    n_states = int(np.prod(in_sizes))
    input_data = np.zeros(n_inputs)

    n_outputs=len(outputvar)
    out_values = [None] * n_outputs
    out_names = [None] * n_outputs
    out_sizes = [int] * n_outputs
    i = 0
    for key, value in outputvar.items():
        out_names[i] = key
        out_values[i] = value
        out_sizes[i] = len(value)
        i += 1

    n_actions=int(np.prod(out_sizes))
    output_data=np.zeros(n_outputs)

    task.n_inputs = n_inputs
    task.in_values = in_values
    task.in_names = in_names
    task.in_sizes = in_sizes
    task.n_states = n_states
    task.in_data = input_data

    task.n_outputs = n_outputs
    task.out_values = out_values
    task.out_names = out_names
    task.out_sizes = out_sizes
    task.n_actions = n_actions
    task.out_data = output_data


def setup():
    global sub_a,sub_s,in_to_states,_count
    sub_s = np.empty(0)
    sub_a = np.empty(0)
    in_to_states = np.full((n_inputs, int(max(in_sizes)), n_states), -1, dtype=np.int)
    _count = np.full((n_inputs, int(max(in_sizes))), 0, dtype=np.int)
    robot.setup(task.AGENT_ELEMENT,task.ENV_ELEMENT)
    print('robot setup!')
    generate_subs()
    print('generate subs!')
    generate_suba()
    print('generate suba!')
    generate_intostates()
    print('generate intostates!')

    #initiated=True
    return

def observestate():
    global n_inputs,in_values,in_sizes
    unwrappedstate=np.zeros(n_inputs)

    robot.update()
    inputdata=task.get_input_data()

    print("inputdata:",inputdata)

    for i in range(n_inputs):
        aux = np.digitize(inputdata[i],in_values[i],right=True)
        unwrappedstate[i]=int(np.clip(aux-1,0,in_sizes[i]-1))

    state = wrap_state(unwrappedstate)
    print('state:',state)

    return state

def actionselection(s):
    a = action_selection.execute(s)
    return a

def execute_action(a):
    global n_outputs
    unwrapped_a=unwrap_action(a)
    act_factors = np.zeros(n_outputs)
    for i in range(n_outputs):
        act_factors[i]=sub_a[i,unwrapped_a[i]]
    
    task.execute_action(act_factors)
    return 

def get_reward():
    r = task.get_reward()
    return r

def generate_subs():
    global sub_s,in_sizes,n_inputs
    sub_s = np.zeros([n_inputs,int(max(in_sizes))])
    for i in range(n_inputs):
        for j,item in enumerate(in_values[i]):
            sub_s[i,j]=item

def generate_suba():
    global sub_a,out_sizes,n_outputs
    sub_a = np.zeros([n_outputs,int(max(out_sizes))])
    for i in range(n_outputs):
        for j,item in enumerate(out_values[i]):
            sub_a[i,j]=item

def generate_intostates():
    global in_to_states,_count,in_sizes,n_states
    print('nstates:',n_states)
    in_to_states=np.full((n_inputs,int(max(in_sizes)),n_states),-1,dtype=np.int)
    _count = np.full((n_inputs,int(max(in_sizes))),0,dtype=np.int)

    for s in range(n_states):
        #print('enter s for')
        ss=unwrap_state(s)
        #print('ss.size',ss.size)
        for i in range(ss.size):
            #print('enter i for')
            j=ss[i]
            k=_count[i,j]
            in_to_states[i,j,k]=s
            _count[i,j]+=1
    return

def wrap_state(un_s):
    global n_inputs,in_sizes
    s=un_s[0]
    for i in range(1,n_inputs):
        pro=1
        for j in range(0,i):
            pro*=in_sizes[j]
        s+=pro*un_s[i]
    return int(s)

def unwrap_state(s):
    global n_inputs,in_sizes
    unwrapped_s = np.zeros(n_inputs, dtype=np.int)
    pro = s
    for i in range(n_inputs):
        unwrapped_s[i] = pro % in_sizes[i]
        pro = int(pro / in_sizes[i])
    #unwrapped_s[n_inputs - 1] = prod
    return unwrapped_s

def wrap_action(un_a):
    global n_outputs,out_sizes
    a=un_a[0]
    for i in range(1,n_outputs):
        pro=1
        for j in range(0,i):
            pro*=out_sizes[j]
        a+=pro*un_a[i]
    return int(a)

def unwrap_action(a):
    global out_sizes,n_outputs
    unwrapped_a = np.zeros(n_outputs, dtype=np.int)
    pro = a
    for i in range(n_outputs):
        unwrapped_a[i] = pro % out_sizes[i]
        pro = int(pro / out_sizes[i])
    #unwrapped_s[n_inputs - 1] = prod
    return unwrapped_a
