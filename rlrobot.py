import numpy as np
import exp
import task
import robot
import learning_process

def run():
    task.setup()
    # Average Reward per step (aveR):
    ave_r = np.zeros((exp.N_REPETITIONS, exp.N_STEPS))
    # Mean(aveR) of all tests per step
    #mean_ave_r = np.zeros(exp.N_STEPS)
    # AveR per episode
    #epi_ave_r = np.zeros([exp.N_REPETITIONS, exp.N_EPISODES])

    robot.connect()

    for rep in range(exp.N_REPETITIONS):
        #last_q = last_v = last_policy = last_q_count=None
        print('enter rep')
        for epi in range(exp.N_EPISODES):
            print('enter epi')
            robot.start()
            print('robot start!')
            learning_process.setup()
            print('lp setup!')

            learning_process.run()
            print('Lp run!')
            #robot.stop()
            #print('robot stop!')
            ave_r[rep]=learning_process.ave_r_step
            print('ave_r:',ave_r[rep])
        #mean_ave_r = np.mean(ave_r,axis=0)
    
    #final_r=mean_ave_r[learning_process.step]

    robot.disconnect()
    return 

if __name__ == '__main__':
    run()
