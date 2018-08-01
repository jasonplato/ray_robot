import numpy  as np
import tensorflow as tf

class DQN:
    def __init__(self,n_actions,n_features,learning_rate=0.01,reward_decay = 0.9,egreedy = 0.9,
                 replace_target_iter = 300,memorize_size = 500,batch_size = 32,egreedy_increment = None):

        self.n_actions = n_actions
        self.n_features = n_features
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

        with tf.variable_scope("eval_net"):
            c_names = ['eval_params',tf.GraphKeys.GLOBAL_VARIABLES]
            units = 10
            w_initializer = tf.truncated_normal_initializer(0,1.0)
            b_initializer = tf.constant_initializer(0.1)

            with tf.variable_scope("layer1"):
                w1 = tf.get_variable(name= 'w1',shape = [self.n_features,units],initializer = w_initializer,collections= c_names)
                b1 = tf.get_variable(name= 'b1',shape = [1,units],initializer = b_initializer,collections= c_names)
                l1 = tf.nn.relu(tf.matmul(self.s,w1)+b1)

            with tf.variable_scope("layer2"):
                w2 = tf.get_variable(name= 'w2',shape = [units,self.n_actions],initializer = w_initializer,collections= c_names)
                b2 = tf.get_variable(name= 'b2',shape = [1,self.n_actions],initializer = b_initializer,collections= c_names)
                self.q_eval = tf.matmul(l1,w2)+b2

        self.loss = tf.reduce_mean(tf.squared_difference(self.q_target,self.q_eval))
        self.train = tf.train.AdamOptimizer(self.lr).minimize(self.loss)

        """
        build target_net
        """
        self.s_ = tf.placeholder(shape= [None,self.n_features],dtype= tf.float32,name = 's_')
        with tf.variable_scope('target_net'):
            c_names = ['target_params',tf.GraphKeys.GLOBAL_VARIABLES]

            with tf.variable_scope('layer1'):
                w1 = tf.get_variable(shape= [self.n_features,units],initializer= w_initializer,name= 'w1',collections=c_names)
                b1 = tf.get_variable(shape= [1,units],initializer= b_initializer,name= 'b1',collections=c_names)
                l1 = tf.nn.relu(tf.matmul(self.s_,w1)+b1)

            with tf.variable_scope('layer2'):
                w2 = tf.get_variable(shape= [units,self.n_actions],initializer= w_initializer,name= 'w2',collections=c_names)
                b2 = tf.get_variable(shape= [1,self.n_actions],initializer= b_initializer,name= 'b1',collections=c_names)
                self.q_next = tf.matmul(l1,w2)+b2


    def store(self,s,a,r,s_):
        if not hasattr(self,'memory_counter'):
            self.memory_counter = 0

        contents = np.hstack((s,[a,r],s_))
        index = self.memory_counter % self.memory_size
        self.memory[index,:]=contents
        self.memory_counter += 1

    def choose_action(self,observation):
        observation = observation[np.newaxis,:]

        if np.random.uniform() < self.epsilon:
            actions_value = self.sess.run(self.q_eval,feed_dict = {self.s:observation})
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
            sample = np.random.choice(self.memory_counter,size= self.batch_size,replace= False)

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
        observation = agent.observestate()
        observation = agent.unwrap_state(observation)

        action = Robot.choose_action(observation)

        agent.execute_action(action)
        r = agent.get_reward()
        observation_ = agent.observestate()
        observation_ = agent.unwrap_state(observation_)
        #.........
        Robot.store(observation,action,r,observation_)

        if (step >200) and (step % 5 == 0) :
            Robot.learn()

        observation = observation_

        step += 1
    print('run over!')

if __name__ =="__main__":
    Robot = DQN(9,7,learning_rate=0.01,reward_decay=0.9,egreedy=0.9,replace_target_iter=500,
             memorize_size=5000,egreedy_increment=0.05)
    run_DQN()

