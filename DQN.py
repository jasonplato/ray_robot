import numpy  as np
import tensorflow as tf

training_iter = 100000
batch_size = 1

n_steps = 10
n_units = 128
n_outputs = 10

class DQN:
    def __init__(self,n_actions,n_features,learning_rate=0.001,reward_decay = 0.9,egreedy = 0.9,
                 replace_target_iter = 300,memorize_size = 500,batch_size = 256,egreedy_increment = None):
        self.n_actions = n_actions
        self.n_features = n_features
        self.weights = {
            'in': tf.Variable(tf.random_normal([self.n_features, n_units])),
            'out': tf.Variable(tf.random_normal([n_units, n_outputs]))
        }
        self.biases = {
        'in': tf.Variable(tf.constant(0.1, shape=[n_units, ])),
        'out': tf.Variable(tf.constant(0.1, shape=[n_outputs, ]))
        }

        self.lr = learning_rate
        self.gamma = reward_decay
        self.epsilon_max = egreedy
        self.replace_target_iter = replace_target_iter
        self.memory_size = memorize_size
        self.batch_size = batch_size
        self.epsilon_increment = egreedy_increment
        self.epsilon = 0.1 if egreedy_increment is not None else self.epsilon_max
        self.learn_step = 0
        self.memory = np.zeros((self.memory_size,n_features*2 + 2))
        self.build_net()
        self.build_rnn_net()
        t_params = tf.get_collection('target_params')
        e_params = tf.get_collection('eval_params')

        self.replace_target = [tf.assign(t,e) for t,e in zip(t_params,e_params)]

        self.sess = tf.Session()

        self.sess.run(tf.global_variables_initializer())
        self.cost = -1

    def build_net(self):
        """
        build eval_net
        """
        self.s = tf.placeholder(shape=[None,self.n_features],dtype= tf.float32,name = 's')
        self.q_target = tf.placeholder(shape = [None,self.n_actions],dtype = tf.float32, name = 'qtarget')
        self.s_3d = tf.expand_dims(self.s,1)
        self.s_4d = tf.expand_dims(self.s_3d,3)
        with tf.variable_scope("eval_net"):
            c_names = ['eval_params',tf.GraphKeys.GLOBAL_VARIABLES]
            units = 64
            w_initializer = tf.truncated_normal_initializer(0,1.0)
            b_initializer = tf.constant_initializer(0.1)

            with tf.variable_scope("conv_layer1"):
                kernel = tf.get_variable(name = 'kernel1',shape = [1,2,1,1],initializer=w_initializer,collections=c_names)
                conv_out = tf.nn.conv2d(self.s_4d,kernel,strides = [1,1,1,1],padding= "VALID")
                conv_out = tf.squeeze(conv_out)
                conv_out = tf.nn.relu(conv_out)
                conv_out = tf.reshape(conv_out,shape = [-1,9])

                #print('conv_out shape:',conv_out.shape)
            with tf.variable_scope("fc_layer1"):
                #fc1_in = tf.reshape(conv_out,shape=[self.batch_size,-1])
                #dim = fc1_in.get_shape()[1].value
                w1 = tf.get_variable(name= 'w1',shape = [9,units],initializer = w_initializer,collections= c_names)
                #w1 = tf.Variable(tf.truncated_normal([dim, units], dtype=tf.float32, stddev=0.01), name='w1',collections=c_names)
                b1 = tf.get_variable(name= 'b1',shape = [1,units],initializer = b_initializer,collections= c_names)
                l1 = tf.nn.relu(tf.matmul(conv_out,w1)+b1)

            with tf.variable_scope("fc_layer2"):
                w2 = tf.get_variable(name= 'w2',shape = [units,self.n_actions],initializer = w_initializer,collections= c_names)
                b2 = tf.get_variable(name= 'b2',shape = [1,self.n_actions],initializer = b_initializer,collections= c_names)
                self.q_eval = tf.matmul(l1,w2)+b2

        self.loss = tf.reduce_mean(tf.squared_difference(self.q_target,self.q_eval))
        self.train = tf.train.AdamOptimizer(self.lr).minimize(self.loss)

        """
        build target_net
        """
        self.s_ = tf.placeholder(shape= [None,self.n_features],dtype= tf.float32,name = 's_')
        self.s__3d = tf.expand_dims(self.s_,1)
        self.s__4d = tf.expand_dims(self.s__3d,3)
        with tf.variable_scope('target_net'):
            c_names = ['target_params',tf.GraphKeys.GLOBAL_VARIABLES]

            with tf.variable_scope("conv_layer1"):
                kernel = tf.get_variable(name = 'kernel1',shape = [1,2,1,1],initializer=w_initializer,collections=c_names)
                conv_out = tf.nn.conv2d(self.s__4d,kernel,strides = [1,1,1,1],padding= "VALID")
                conv_out = tf.squeeze(conv_out)
                conv_out = tf.nn.relu(conv_out)
                conv_out = tf.reshape(conv_out,shape = [-1,9])
                #print('conv_out shape:',conv_out.shape)

            with tf.variable_scope('layer1'):
                w1 = tf.get_variable(shape= [9,units],initializer= w_initializer,name= 'w1',collections=c_names)
                b1 = tf.get_variable(shape= [1,units],initializer= b_initializer,name= 'b1',collections=c_names)
                l1 = tf.nn.relu(tf.matmul(conv_out,w1)+b1)

            with tf.variable_scope('layer2'):
                w2 = tf.get_variable(shape= [units,self.n_actions],initializer= w_initializer,name= 'w2',collections=c_names)
                b2 = tf.get_variable(shape= [1,self.n_actions],initializer= b_initializer,name= 'b1',collections=c_names)
                self.q_next = tf.matmul(l1,w2)+b2


    def build_rnn_net(self):
        self.rnn_in = tf.placeholder(dtype=tf.float32, shape = [None, n_steps, self.n_features])
        #self.q_target = tf.placeholder(dtype=tf.float32, shape = [None, self.n_actions])

        X = tf.reshape(self.rnn_in,[-1,self.n_features])
        X_in = tf.matmul(X,self.weights['in'])+self.biases['in']
        X_in = tf.reshape(X_in,[-1,n_steps,n_units])

        cell = tf.nn.rnn_cell.BasicLSTMCell(n_units,forget_bias= 1.0,state_is_tuple=True)

        init_state = cell.zero_state(batch_size,dtype=tf.float32)

        outputs,final_state = tf.nn.dynamic_rnn(cell,X_in,initial_state = init_state,time_major = False)
        outputs = tf.unstack(tf.transpose(outputs,[1,0,2]))
        self.rnn_out = tf.matmul(outputs[-1],self.weights['out'])+self.biases['out']

    def store(self,s,a,r,s_):
        if not hasattr(self,'memory_counter'):
            self.memory_counter = 0
        term = np.array((a,r))
        term = term[np.newaxis,:]
        contents = np.hstack((s,term,s_))
        #contents = np.hstack((s,[a,r],s_))
        index = self.memory_counter % self.memory_size
        self.memory[index,:]=contents
        self.memory_counter += 1

    def choose_action(self,observation):
        #observation = observation[np.newaxis,:]

        if np.random.uniform() < self.epsilon:
            actions_value = self.sess.run([self.q_eval],feed_dict = {self.s:observation})
            #print('conv_out:',conv_out)
            action = np.argmax(actions_value)
        else :
            action = np.random.randint(0,self.n_actions)
        return action

    def learn(self):
        if self.learn_step % self.replace_target_iter == 0:
            self.sess.run(self.replace_target)

        if self.memory_counter > self.memory_size:
            sample = np.random.choice(self.memory_size,size= self.batch_size,replace= False)
        else:
            sample = np.random.choice(self.memory_counter,size= self.batch_size,replace= True)

        batch_memory = self.memory[sample,:]

        q_next,q_eval = self.sess.run([self.q_next,self.q_eval],feed_dict= {self.s:batch_memory[:,:self.n_features],
                                                                            self.s_:batch_memory[:,-self.n_features:]})

        q_target = q_eval.copy()
        batch_index = np.arange(self.batch_size,dtype = np.int32)
        eval_act_index = batch_memory[:,self.n_features].astype(int)
        reward = batch_memory[:,self.n_features + 1]

        q_target[batch_index,eval_act_index] = reward + self.gamma * np.max(q_next,axis=1)

        _,self.cost = self.sess.run([self.train,self.loss],feed_dict= {self.s:batch_memory[:,:self.n_features],
                                                                       self.q_target:q_target})

        print("cost:",self.cost)

        self.epsilon = self.epsilon + self.epsilon_increment if self.epsilon <self.epsilon_max else self.epsilon_max
        self.learn_step += 1


def run_DQN():
    import agent
    import time
    import exp
    import rl_vrep

    rl_vrep.connect()
    rl_vrep.start()
    time.sleep(0.5)
    agent.setup_task()
    time.sleep(0.5)
    agent.setup()
    time.sleep(0.5)
    step = 0
    for epi in range(30000):
        print('step:',step)
        observation = np.empty(shape= (10 ,10))
        observation_ = np.empty(shape = (10,10))
        for i in range(10):
            observation_i = agent.observestate()
            observation_i = agent.unwrap_state(observation_i)
            observation[i] = np.array(observation_i)
        observation = observation[np.newaxis,:]
        observation= Robot.sess.run(Robot.rnn_out,feed_dict= {Robot.rnn_in:observation})

        #observation = np.mean(observation,axis= 0)
        action = Robot.choose_action(observation)

        agent.execute_action(action)
        print('action:',action)
        #time.sleep(0.5)

        r = agent.get_reward()
        print('reward:',r)
        for i in range(10):
            observation_i = agent.observestate()
            observation_i = agent.unwrap_state(observation_i)
            observation_[i] = np.array(observation_i)
        #observation_ = np.mean(observation_,axis= 0)
        #.........
        observation_ = observation_[np.newaxis,:]
        observation_= Robot.sess.run(Robot.rnn_out,feed_dict= {Robot.rnn_in:observation_})
        Robot.store(observation,action,r,observation_)

        if (step >200) and (step % 10 == 0) :
            Robot.learn()

        observation = observation_

        step += 1
    print('run over!')

if __name__ =="__main__":
    Robot = DQN(9,10,learning_rate=0.01,reward_decay=0.9,egreedy=0.9,replace_target_iter=200,
             memorize_size=2000,egreedy_increment=0.05)
    run_DQN()

