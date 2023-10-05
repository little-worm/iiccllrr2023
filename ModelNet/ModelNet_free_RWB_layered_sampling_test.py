import numpy as np
import os
import random,time
from numpy import array,ones,sum
import itertools,ot,sys,os
cfd = os.path.dirname(__file__)
sys.path.append(os.path.abspath( cfd + '../WB' ))
print("sys.path = ",sys.path)
from Tools import my_mkdir
from Layered_sampling import layeredSamplingTest
time0 = time.strftime("%Y-%m-%d %H:%M:%S",time.localtime())


### data_size, zeta, mean, noise_var
zeta = 0.1; noise_mean = 50; noise_var = 50 
epsilon = 50000; testtimes = 10; freeSupportIter = 3; repeatTimes = 3
kClusters = 60; label=0; sampleSizeList = array([10,20,30,40,50,60])
data_size = 989; shift_mean = 0; shift_std = 50; shift_num = 60; Radius = 2000; logbase = 1.2
## point_lists,u_list,w_list
cfd = os.path.dirname(__file__)

locations_weights = np.loadtxt(cfd + 'Pointnet_Pointnet2_pytorch-master/8Kmeans.matrix_2023_0922', dtype=np.float32, delimiter=' ')
#print(max(locations_weights.flatten()),min(locations_weights.flatten()))
locations_weights = locations_weights.reshape(-1,60,4)
locations_weights = locations_weights[:data_size]
print("locations_weights = ",locations_weights.shape)
point_lists = locations_weights[:,:60,:3] * 50
point_lists[:shift_num] = array( [point_lists[i] + np.random.normal(shift_mean,shift_std,1) for i in range(shift_num)] )
u_list = locations_weights[:,:60,-1] 
u_list = array([u/sum(u) for u in u_list])

print("u_list = ",u_list.shape)
w_list = ones(data_size)/data_size   
print("point_lists,u_list = ",point_lists.shape,u_list.shape)



result_mean,result_std,result_var,layerPoints_list_,_,_ = layeredSamplingTest(point_lists,u_list,w_list,sampleSizeList,zeta,noise_mean,noise_var,repeatTimes,Radius,logbase,epsilon,freeSupportIter,testtimes)



timeStamp = time.strftime("%m-%d_%H:%M",time.localtime())
foldName = 'ModelNet_' + str(label) + '_' + str(data_size)+ '_'+ str(kClusters)+ '_'+ str(epsilon) +'_outlier_'+ str(zeta)+ '_' + str(noise_mean) + '_' + str(noise_var) + '_shift_' + str(shift_mean) + '_' + str(shift_std)+'_'+str(shift_num) +'_R'+str(Radius) +'_'+str(logbase)+ '_tt'+ str(testtimes) + '_' + str(timeStamp)
foldDir = cfd + 'results_free_support_RWB_layered_sampling/' + foldName
my_mkdir(foldDir)
outputPointer = open(foldDir+'/others.txt','a+')
print(os.path.abspath(foldDir + '/'+ foldName +'.txt-ave'))
##  np.savetxt(foldDir + '/' + foldName +'.txt-ave', [[11,22],[33,44]],fmt='%f',delimiter=' ',newline='\r\n')

np.savetxt(foldDir + '/' + foldName +'.txt-ave',result_mean,fmt='%f',delimiter=' ',newline='\r\n')
##  np.savetxt(foldDir + '/' + foldName + '.txt-var',result_var,fmt='%f',delimiter=' ',newline='\r\n')
np.savetxt(foldDir + '/' + foldName + '.txt-std',result_std,fmt='%f',delimiter=' ',newline='\r\n')
time1 = time.strftime("%Y-%m-%d %H:%M:%S",time.localtime())
print("time0,time1 = ", time0,time1,file=outputPointer ) 
print("-------------------------------------------------------",file=outputPointer )
print("data_size, kClusters,label         = ",data_size,kClusters,label,file=outputPointer)
print("zeta,noise_mean,noise_var          = ",zeta,noise_mean,noise_var,file=outputPointer)
print("epsilon,testtimes                  = ",epsilon,testtimes,file=outputPointer)
print("freeSupportIter,repeatTimes,Radius = ",freeSupportIter,repeatTimes,Radius,file=outputPointer)
print("[len(ll) for ll in layerPoints_list_] = ",[len(ll) for ll in layerPoints_list_],file=outputPointer)
print("shift_num,logbase               = ",shift_num,logbase,file=outputPointer)




