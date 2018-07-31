
import rlrobot

rlrobot.run()


"""
import vrep
#vrep.simxFinish(-1)
clientID=vrep.simxStart('127.0.0.1',19999,True,True,5000,5)
if clientID!=-1:
    print('Connected to remote API Server')
else:
    print('Connection not successful')

errorcode ,left=vrep.simxGetObjectHandle(clientID,'leftMotor',vrep.simx_opmode_oneshot)
errorcode ,right=vrep.simxGetObjectHandle(clientID,'rightMotor',vrep.simx_opmode_oneshot)
#returncode, collisionhandle=vrep.simxGetCollisionHandle(clientID,'Cuboid5',vrep.simx_opmode_blocking)
vrep.simxSetJointTargetVelocity(clientID, left, -1, vrep.simx_opmode_streaming)
vrep.simxSetJointTargetVelocity(clientID, right, -1, vrep.simx_opmode_streaming)
vrep.simxStartSimulation(clientID,vrep.simx_opmode_oneshot)
"""
