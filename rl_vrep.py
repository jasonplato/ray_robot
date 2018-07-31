import time
import numpy as np
import vrep

LASER_DISTRIBUTION = ('sensor_left','sensor_left_front', 'sensor_front_left','sensor_front',
                       'sensor_front_right','sensor_right_front','sensor_right')
N_LASERS = 7  # 1 point laser each

robotID = -1
laserID = [-1] * N_LASERS
left_motorID = 0
right_motorID = 0
clientID = -1

distance = np.full(N_LASERS, -1, dtype=np.float64)  # distances from lasers (m)
pose = np.full(3, -1, dtype=np.float64)  # Pose 2d base: x(m), y(m), theta(rad)

def connect():
    global clientID
    vrep.simxFinish(-1)
    clientID=vrep.simxStart('127.0.0.1',19999,True,True,5000,5)
    if clientID!=-1:
        print('Connected to remote API Server')
    else:
        print('Connection not successful')

    time.sleep(0.5)

def disconnect():
    vrep.simxFinish(clientID)
    return

def start():
    #stop()
    setup()
    vrep.simxStartSimulation(clientID,vrep.simx_opmode_oneshot)
    time.sleep(0.5)

    #setup()
    #vrep.simxStartSimulation(clientID,vrep.simx_opmode_oneshot)
    #time.sleep(0.5)

    return 

def stop():
    vrep.simxStopSimulation(clientID,vrep.simx_opmode_oneshot)
    time.sleep(0.5)
    return


def setup():
    global robotID,left_motorID,right_motorID,laserID
    errorcode ,robotID=vrep.simxGetObjectHandle(clientID,'Pioneer_p3dx',vrep.simx_opmode_oneshot_wait)
    errorcode ,left_motorID=vrep.simxGetObjectHandle(clientID,'leftMotor',vrep.simx_opmode_oneshot_wait)
    errorcode ,right_motorID=vrep.simxGetObjectHandle(clientID,'rightMotor',vrep.simx_opmode_oneshot_wait)
    for i,item in enumerate(LASER_DISTRIBUTION):
        errorcode ,laserID[i]=vrep.simxGetObjectHandle(clientID,item,vrep.simx_opmode_oneshot_wait)
        print("laserID[i]:",laserID[i],",errorcode:",errorcode)
    vrep.simxGetObjectPosition(clientID, robotID, -1, vrep.simx_opmode_streaming)
    vrep.simxGetObjectOrientation(clientID, robotID, -1,vrep.simx_opmode_streaming)

    for i in laserID:
        vrep.simxReadProximitySensor(clientID, i, vrep.simx_opmode_streaming)  
    
    return 

def get_pose():
    errorcode, pos = vrep.simxGetObjectPosition(clientID, robotID, -1, vrep.simx_opmode_buffer)
    errorcode, ori = vrep.simxGetObjectOrientation(clientID, robotID, -1, vrep.simx_opmode_buffer)
    pos = np.array([pos[0], pos[1], ori[2]])
    return pos

def get_distance():
    global laserID
    for i in range(N_LASERS):
        errorcode, detectionState,detectedPoint,detectedObjectHandle,detectedSurfaceNormalVector=vrep.simxReadProximitySensor(clientID,laserID[i],vrep.simx_opmode_buffer)
        distance[i]=detectedPoint[2]
    #print("laserID:",laserID)
    return distance

def move_wheels(left,right):
    vrep.simxSetJointTargetVelocity(clientID, left_motorID, left, vrep.simx_opmode_streaming)
    vrep.simxSetJointTargetVelocity(clientID, right_motorID, right,vrep.simx_opmode_streaming)
    return

def stop_wheels():
    vrep.simxSetJointTargetVelocity(clientID, left_motorID, 0, vrep.simx_opmode_streaming)
    vrep.simxSetJointTargetVelocity(clientID, right_motorID, 0,vrep.simx_opmode_streaming)
    return
