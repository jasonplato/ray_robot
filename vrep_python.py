# -*- coding: utf-8 -*-
"""
Created on Tue Jan 06 22:00:39 2015

@author: Nikolai K.
"""
#Import Libraries:
import vrep                  #V-rep library
import sys
import time                #used to keep track of time
import numpy as np         #array library
import math
import matplotlib as mpl   #used for image plotting

#Pre-Allocation

PI=math.pi  #pi=3.14..., constant
GAMMA=0.9
maxdetectiondist=0.5
nodetectiondist=0.01
#go=[0.4,0.3,0.2,0.1,-0.1,-0.2,-0.3,-0.4]
vrep.simxFinish(-1) # just in case, close all opened connections

clientID=vrep.simxStart('127.0.0.1',19999,True,True,5000,5)

if clientID!=-1:  #check if client connection successful
    print ('Connected to remote API server')
    
else:
    print ('Connection not successful')
    sys.exit('Could not connect')


#retrieve motor  handles
errorCode,left_motor_handle=vrep.simxGetObjectHandle(clientID,'Pioneer_p3dx_leftMotor',vrep.simx_opmode_oneshot_wait)
errorCode,right_motor_handle=vrep.simxGetObjectHandle(clientID,'Pioneer_p3dx_rightMotor',vrep.simx_opmode_oneshot_wait)


sensor_h=[] #empty list for handles
sensor_val=np.array([]) #empty array for sensor measurements

#orientation of all the sensors: 
sensor_loc=np.array([-PI/2, -50/180.0*PI,-30/180.0*PI,-10/180.0*PI,10/180.0*PI,30/180.0*PI,50/180.0*PI,PI/2,PI/2,130/180.0*PI,150/180.0*PI,170/180.0*PI,-170/180.0*PI,-150/180.0*PI,-130/180.0*PI,-PI/2]) 

#for loop to retrieve sensor arrays and initiate sensors
for x in range(1,8+1):
    #print(x)
    errorCode,sensor_handle=vrep.simxGetObjectHandle(clientID,'Pioneer_p3dx_ultrasonicSensor'+str(x),vrep.simx_opmode_oneshot_wait)
    sensor_h.append(sensor_handle) #keep list of handles
    errorCode,detectionState,detectedPoint,detectedObjectHandle,detectedSurfaceNormalVector=vrep.simxReadProximitySensor(clientID,sensor_handle,vrep.simx_opmode_streaming)
        #sensor_val=np.append(sensor_val,np.linalg.norm(detectedPoint)) #get list of values
        
def avoid(sensor_sq,min_ind,mean,std):
    min_d=min_ind[0][0]
    kp=0
    if min_d>0.0:
        kp=min(1,1-1/((min_d-mean)/std))
        print("kp:",kp)
    if sensor_sq[min_d]<0.2:
        steer=-1/sensor_loc[min_d]

        if sensor_sq[min_d]<0.1:
            steer+=steer*GAMMA
            if sensor_sq[min_d]<0.05:
                steer+=steer*GAMMA
                #if sensor_sq[min_d]<0.01:
                    #steer+=steer*GAMMA
                    #if sensor_sq[min_d]<0.005:
                        #steer+=steer*GAMMA
    else:
        steer=0
    return steer*kp

t = time.time()


while 1:
    #Loop Execution
    v=1	#forward velocity
    #kp=0.75	#steering gain
    sensor_val=np.array([])    
    for x in range(1,8+1):
        #print(x)
        errorCode,detectionState,detectedPoint,detectedObjectHandle,detectedSurfaceNormalVector=vrep.simxReadProximitySensor(clientID,sensor_h[x-1],vrep.simx_opmode_buffer)
        sensor_val=np.append(sensor_val,np.linalg.norm(detectedPoint)) #get list of values
        #print("Point:",detectedPoint)

    
    #controller specific
    sensor_sq=sensor_val[0:8]*sensor_val[0:8] #square the values of front-facing sensors 1-8
    #print ("sensor_sq",sensor_sq)
    min_ind=np.where(sensor_sq==np.min(sensor_sq))
    sensor_mean=np.mean(sensor_sq)
    sensor_std=np.std(sensor_sq)
    #max_ind=np.where(sensor_sq==np.max(sensor_sq))
    #max_ind=max_ind[0][0]
    print(sensor_sq)
    steer=avoid(sensor_sq,min_ind,sensor_mean,sensor_std)



    vl=+v
    vr=0
    #print ("V_l =",vl)
    #print ("V_r =",vr)

    errorCode=vrep.simxSetJointTargetVelocity(clientID,left_motor_handle,vl, vrep.simx_opmode_streaming)
    errorCode=vrep.simxSetJointTargetVelocity(clientID,right_motor_handle,vr, vrep.simx_opmode_streaming)


    time.sleep(0.1) #loop executes once every 0.2 seconds (= 5 Hz)

#Post ALlocation
errorCode=vrep.simxSetJointTargetVelocity(clientID,left_motor_handle,0, vrep.simx_opmode_streaming)
errorCode=vrep.simxSetJointTargetVelocity(clientID,right_motor_handle,0, vrep.simx_opmode_streaming)
