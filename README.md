# ray_robot
python + V_REP<br>
这是一个完整的Pycharm Projects。
* 代码文件
    ~~~
    action_selection.py
    agent.py
    example.py
    exp.py
    learning_process.py
    q_learning.py
    rl_vrep.py
    rlrobot.py
    robot.py
    task.py
    test.py
    DQN.py    
    ~~~
* V-REP中的场景文件
    ~~~  
    demo.ttt    #目前主要是在这个场景中进行训练和测试
    python_vrep_obstacle_avoid.ttt  
    ~~~
* V-REP开发环境文件
    ~~~
    这些文件必须要包含在 import vrep 的项目文件夹中
    remoteApi.so
    simpleTest.py   #实际上没有影响
    vrep.py
    vrepConst.py
    vrep_python.py
    ~~~
* 如何训练
    ~~~
    Nature Q-learning: 
    在V-REP中打开场景并点击开始模拟后,运行 example.py
    训练的参数可以在 exp.py 中更改
    训练结果会保存在 policy_results.txt 中
    
    DQN:
    直接运行DQN.py就行
    ~~~
* 如何测试
    ~~~
    Nature Q-learning:
    在V-REP中打开场景并点击开始模拟后,运行 test.py
    ~~~
## VERSION_1:
### 做法
只能通过最naive的方法控制小车避障，即读取小车身上的传感器距离数据(z轴)，若该距离小于一个阈值，则转向，具体转向的角度与该传感器相对于小车车身的角度有关。越是靠两侧的传感器检测到距离太近，转向的角度越大。此版本参照了V-REP Pioneer_3dx 小车模型的Lua脚本。
### 效果
小车几乎不会撞到障碍，但是由于使用的方法非常naive，小车在狭窄的地方为了避障会疯狂摇摆。另外，小车也不能驶入一个狭窄的路口（即使能容得下它通行），它检   测到这样的路口便会调转方向（是因为小车车头有8个传感器，排列密集）。
## VERSION_2:
### 做法
使用强化学习方法，Natrue Q-Learning。
* 回报函数
    ~~~
    REWARDS = [-1, -0.1,-0.01, 1.0]
    if 2个以上的传感器距离数据 < 阈值 :
       r = REWARDS[0]
    else 1个或2个传感器距离数据 < 阈值 :
       r = REWARDS[1]
    else 如果小车移动的距离 >= 另一个阈值 ：
       r = REWARDS[3]
    else 
       r = REWARDS[2]
    ~~~
* 状态空间
    ~~~
    一开始非常纠结应该怎样设置状态空间，尝试过：将这8个传感器分为前、左、右三组，
    每组分别检测到与障碍物之间地距离为多少，是否超过阈值，以此来划分小车处于不同的状态。
    这种方法总觉得有点像是在背地图，不能应对随机的场景。
    后来参考了https://github.com/angelmtenor/RL-ROBOT 的状态空间设置，
    简单粗暴地将所有传感器检测到的距离等距地划分为几个状态，那么总状态数就是
    每个传感器等距划分出来的子状态数的 N 次方（N = 传感器数）。这种方法的状态空间将会很大，但是随机性更强。
    ~~~
* 动作空间
    ~~~
   动作空间就比较简单了，小车有两个轮，每个轮的速度即为一个动作，所以动作空间只需要 2 个动作即可。
    ~~~
* Q值迭代
    ~~~
    Q[S,A] = Q[S,A] + ALPHA * ( R + GAMMA * Q[S',A'] - Q[S,A] )
    ~~~
* 训练
    ~~~
    STEPS = 15000
    将学到的策略保存在 policy_results.txt 以用于测试。
    ~~~
### 问题
可能是由于训练的步数还不够多？小车在测试时还是显得很笨，能畅行通过的地方还在犹豫，
且由于只用了小车头部的8个传感器，小车对于尾部是否会撞到障碍物毫不知情。
   
### 改进
1. 尝试将小车尾部的传感器也加入进行学习，但是同时要试着减少状态空间。否则每加入一个传感器，状态空间都会指数增长。
2. 增大训练步数。
3. 调整 REWARDS
4. 调整阈值
   
## VERSION_3：
### 做法
加入神经网络，即使用DQN方法。加入了尾部传感器（因为使用了NN所以不用太在意过多的状态可能会带来的效率问题）。
### 进度
* 搭建网络架构
    ~~~
    网络分为：evaluate_Network 和 target_Network
    
    #### evaluate_Network:
    两层结构，隐藏层单元数 10
    最后得到 q_eval，即为Q估计值（时刻根据当前环境进行更新的）
    #### target_Network:
    两层结构，隐藏层单元数 10 
    最后得到 q_next，即为下一步的Q值
    ~~~
 * 网络输入：
    ~~~
    1.每隔0.5秒取一帧作为输入
    2.取连续5帧，取mean值作为输入(已测试过，效果比上一种稍强)
    3.取连续5帧，输入到LSTM网络中提取时序信息，输出最后step作为eval_network或target_network的输入，后续步骤不变
    ~~~
