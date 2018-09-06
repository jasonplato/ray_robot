import multiprocessing
import threading
import tensorflow as tf
import numpy as np
import agent
import rl_vrep
import time
import scipy.signal
import itertools

N_WORKERS = multiprocessing.cpu_count()
MAX_STEP = 20000
UPDATE_GLOBAL_ITER = 10
GAMMA = 0.9
ENTROPY_BETA = 0.01
#LR_A = 0.0001
#LR_C = 0.001
#GLOBAL_RUNNING_R = []
N_S = 10
N_A = 9
units = 128
MEMORY_SIZE = 3000


class ACNet(object):
    def __init__(self,scope,trainer=None):
        self.scope = scope
        if scope == "global":
            with tf.variable_scope(scope):
                self.s = tf.placeholder(shape = [None,N_S],dtype=tf.float32)
                #self.a_params,self.c_params = self.build_net(scope)[-2:]
                self.build_net(scope)
        else:
            with tf.variable_scope(scope):
                self.s = tf.placeholder(shape = [None,N_S],dtype=tf.float32)
                #self.a = tf.placeholder(shape = [None,N_A],dtype = tf.float32)
                #self.v_target = tf.placeholder(shape = [None,1],dtype=tf.float32)
                #mu,sigma,self.v,self.a_params,self.c_params = self.build_net(scope)
                self.build_net(scope)
                self.update_net(trainer)

                """
                td = tf.subtract(self.v_target,self.v)
                with tf.name_scope('c_loss'):
                    self.c_loss = tf.reduce_mean(tf.square(td))

                with tf.name_scope('wrap_a_out'):
                    mu,sigma = mu * A_BPOUND[1],sigma + 1e-4

                normal_dist = tf.distributions.Normal(mu,sigma)

                with tf.name_scope('a_loss'):
                    log_prob = normal_dist.log_prob(self.a)
                    exp_v = log_prob * tf.stop_gradient(td)
                    entropy = normal_dist.entropy()
                    self.exp_v = ENTROPY_BETA * entropy +  exp_v
                    self.a_loss = tf.reduce_mean(-self.exp_v)

                with tf.name_scope('choose_action'):
                    self.A = tf.clip_by_value(tf.squeeze(normal_dist.sample(1),axis = [0,1]),A_BPOUND[0],A_BPOUND[1])
                with tf.name_scope('local_grad'):
                    self.a_grads = tf.gradients(self.a_loss,self.a_params)
                    self.c_grads = tf.gradients(self.c_loss,self.c_params)

            with tf.name_scope('sync'):
                with tf.name_scope('pull'):
                    self.pull_a_params = [l_p.assign(g_p) for l_p,g_p in zip(self.a_params,globalAC.a_params)]
                    self.pull_c_params = [l_p.assign(g_p) for l_p,g_p in zip(self.c_params,globalAC.c_params)]

                with tf.name_scope('push'):
                    self.update_a = OPT_A.apply_gradients(zip(self.a_grads,globalAC.a_params))
                    self.update_c = OPT_C.apply_gradients(zip(self.c_grads,globalAC.c_params))
                """

    def build_net(self,scope):
        w_initializer = tf.truncated_normal_initializer(0,1.0)
        b_initializer = tf.constant_initializer(0.1)
        self.s_3d = tf.expand_dims(self.s,1)
        self.s_4d = tf.expand_dims(self.s_3d,3)
        with tf.variable_scope('actor'):
            with tf.variable_scope('conv1'):
                kernel1 = tf.get_variable(name = 'kernel1',shape = [1,2,1,1],initializer=w_initializer)
                conv1 = tf.nn.conv2d(self.s_4d,kernel1,strides = [1,1,1,1],padding= "VALID")
                conv1 = tf.nn.relu(conv1)
            with tf.variable_scope('conv2'):
                kernel2 = tf.get_variable(name = 'kernel2',shape = [1,1,1,1],initializer = w_initializer)
                conv2 = tf.nn.conv2d(conv1,kernel2,strides = [1,1,1,1],padding= "VALID")
                conv2 = tf.squeeze(conv2)
                conv2 = tf.nn.relu(conv2)
                conv2 = tf.reshape(conv2,shape = [-1,9])
            with tf.variable_scope('fc'):
                w1 = tf.get_variable(name= 'w1',shape = [9,units],initializer = w_initializer)
                b1 = tf.get_variable(name= 'b1',shape = [1,units],initializer = b_initializer)
                la = tf.nn.relu(tf.matmul(conv2,w1)+b1)

                w2 = tf.get_variable(name= 'w2',shape = [units,N_A],initializer = w_initializer)
                b2 = tf.get_variable(name= 'b2',shape = [1,N_A],initializer = b_initializer)
                self.policy = tf.nn.softmax(tf.matmul(la,w2)+b2)

        with tf.variable_scope('critic'):
            with tf.variable_scope('conv1'):
                kernel1 = tf.get_variable(name = 'kernel1',shape = [1,2,1,1],initializer=w_initializer)
                conv1 = tf.nn.conv2d(self.s_4d,kernel1,strides = [1,1,1,1],padding= "VALID")
                conv1 = tf.nn.relu(conv1)
            with tf.variable_scope('conv2'):
                kernel2 = tf.get_variable(name = 'kernel2',shape = [1,1,1,1],initializer = w_initializer)
                conv2 = tf.nn.conv2d(conv1,kernel2,strides = [1,1,1,1],padding= "VALID")
                conv2 = tf.squeeze(conv2)
                conv2 = tf.nn.relu(conv2)
                conv2 = tf.reshape(conv2,shape = [-1,9])
            with tf.variable_scope('fc'):
                w1 = tf.get_variable(name= 'w1',shape = [9,units],initializer = w_initializer)
                b1 = tf.get_variable(name= 'b1',shape = [1,units],initializer = b_initializer)
                lc = tf.nn.relu(tf.matmul(conv2,w1)+b1)

                w2 = tf.get_variable(name= 'w2',shape = [units,1],initializer = w_initializer)
                b2 = tf.get_variable(name= 'b2',shape = [1,1],initializer = b_initializer)
                self.value = tf.matmul(lc,w2)+b2
            #a_params = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES,scope = scope+'/actor')
            #c_params = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES,scope = scope+'/critic')
        #return a_params,c_params

    def update_net(self,trainer):
        self.actions = tf.placeholder(shape = [None],dtype = tf.int32)
        actions_onehot = tf.one_hot(self.actions,N_A,dtype = tf.float32)
        self.target_v = tf.placeholder(shape =[None],dtype = tf.float32)
        self.advantages = tf.placeholder(shape = [None],dtype = tf.float32)

        actions_prob = tf.reduce_sum(self.policy * actions_onehot,[1])

        self.critic_loss = 0.5 * tf.reduce_sum(tf.squared_difference(self.target_v,tf.reshape(self.value,[-1])))

        self.entropy = -tf.reduce_sum(self.policy * tf.log(self.policy + 1e-30),1)
        self.entropy_loss = -ENTROPY_BETA * tf.reduce_sum(self.entropy)

        self.actor_loss = -tf.reduce_sum(tf.log(actions_prob + 1e-30) * self.advantages)
        self.actor_loss += self.entropy_loss

        self.loss = self.actor_loss + self.critic_loss

        local_vars = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES,self.scope)
        self.grads = tf.gradients(self.loss,local_vars)

        global_vars = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES,'global')
        self.apply_grads = trainer.apply_gradients(zip(self.grads,global_vars))
class WORKER(object):
    def __init__(self,id,global_steps_counter):
        self.name = 'worker_' + str(id)
        self.local_steps = 0
        self.global_steps_counter = global_steps_counter
        self.lr = tf.Variable(0.001,dtype = tf.float32,trainable = False)
        #self.delta_lr = self.lr / (MAX_STEP / N_WORKERS)
        self.trainer = tf.train.AdamOptimizer(self.lr)
        self.local_net = ACNet(scope = self.name,trainer= self.trainer)
        self.update_local_op = self.update_local_vars()
        #self.anneal_lr_op = self.anneal_lr()
    def work(self,sess,coord):
        print('starting %s...\n' % self.name)
        while not coord.should_stop() :
            sess.run(self.update_local_op)
            memory = []
            s = agent.observestate()
            s = agent.unwrap_state(s)
            while True:
                p,v = sess.run([self.local_net.policy,self.local_net.value],feed_dict = {self.local_net.s:s[np.newaxis,:]})
                a = np.random.choice(range(N_A),p = p[0])

                agent.execute_action(a)
                print('action:',a)
                time.sleep(0.3)

                r = agent.get_reward()
                s1 = agent.observestate()
                s1 = agent.unwrap_state(s1)

                memory.append([s,a,r,s1,v[0][0]])
                s = s1
                global_steps = next(self.global_steps_counter)
                self.local_steps += 1
                #sess.run(self.anneal_lr_op)

                collide = None
                if np.where(s1 < 0):
                    collide = True
                else :
                    collide = False

                if not collide and len(memory) == MEMORY_SIZE:
                    v1 = sess.run(self.local_net.value,feed_dict = {self.local_net.s:s})
                    self.train(memory,sess,v1[0][0],global_steps)
                    memory = []
                    sess.run(self.update_local_op)
                if collide:
                    break

            if len(memory) != 0:
                self.train(memory,sess,0.0,global_steps)

            if global_steps >= MAX_STEP:
                COORD.request_stop()
            """  
                ep_r += r
                buffer_a.append(a)
                buffer_r.append(r)
                buffer_s.append(s)

                if total_step % UPDATE_GLOBAL_ITER == 0:
                    v_s_ = SESS.run(self.AC.v,{self.AC.s:s_[np.newaxis,:]})[0,0]
                    buffer_v_target = []
                    for i in buffer_r[::-1]:
                        v_s_ = r + GAMMA *v_s_
                        buffer_v_target.append(v_s_)
                    buffer_v_target.reverse()

                    buffer_s = np.vstack(buffer_s)
                    buffer_a = np.vstack(buffer_a)
                    buffer_v_target = np.vstack(buffer_v_target)
                    feed_dict = {
                        self.AC.s : buffer_s,
                        self.AC.a : buffer_a,
                        self.AC.v_target : buffer_v_target,
                    }
                    self.AC.update_global(feed_dict)
                    buffer_s = []
                    buffer_a = []
                    buffer_r = []
                    self.AC.pull_global()

                s = s_
                total_step += 1
            """
    def train(self,memory,sess,bootstrap_value,global_steps):
        memory = np.array(memory)
        observe,actions,rewards,next_observe,values = memory.T
        rewards_plus = np.asarray(rewards.tolist() + [bootstrap_value])
        discounted_rewards = scipy.signal.lfilter([1],[1,-GAMMA],rewards_plus[::-1],axis=0)[::-1][:-1]
        advantages = discounted_rewards - values
        _ = sess.run([self.local_net.apply_grads],feed_dict = {self.local_net.s : np.stack(observe),
                                                               self.local_net.actions: actions,
                                                               self.local_net.target_v:discounted_rewards,
                                                               self.local_net.advantages:advantages})

    def update_local_vars(self):
        global_vars = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES,'global')
        local_vars = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES,self.name)
        update_op =[]
        for g_v,l_v in zip(global_vars,local_vars):
            update_op.append(l_v.assign(g_v))
        return update_op
    #def anneal_lr(self):
        #return tf.cond((self.lr > 0.0),
                       #lambda:tf.assign_sub(self.lr,self.delta_lr),
                       #lambda:tf.assign(self.lr,0.0))

if __name__ == "__main__":
    rl_vrep.connect()
    rl_vrep.start()
    time.sleep(0.5)
    agent.setup_task()
    time.sleep(0.5)
    agent.setup()
    time.sleep(0.5)
    global_steps_counter = itertools.count()
    global_net = ACNet('global')
    workers = []
    for i in range(1,N_WORKERS+1):
        worker = WORKER(i,global_steps_counter)
        workers.append(worker)
    with tf.Session() as sess:
        COORD = tf.train.Coordinator()
        print('Initializing\n')
        sess.run(tf.global_variables_initializer())
        workers_threads = []
        for worker in workers:
            t = threading.Thread(target = lambda:worker.work(sess,COORD))
            t.start()
            time.sleep(0.5)
            workers_threads.append(t)
        COORD.join(workers_threads)
    """    
    SESS = tf.Session()
    

    with tf.device("/cpu:0"):
        OPT_A = tf.train.AdamOptimizer(LR_A)
        OPT_C = tf.train.AdamOptimizer(LR_C)
        GLOBAL_AC = ACNet("global")
        workers = []

        for i in range(N_WORKERS):
            i_name = 'w_%i' % i
            workers.append(WORKER(i_name,GLOBAL_AC))
    COORD = tf.train.Coordinator()
    SESS.run(tf.global_variables_initializer())

    worker_threads = []
    for worker in workers:
        job = lambda:worker.work()
        t = threading.Thread(target = job)
        t.start()
        worker_threads.append(t)
    COORD.join(worker_threads)
    """

