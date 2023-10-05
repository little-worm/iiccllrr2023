import numpy as np
import os
import random,time
from numpy import array,ones,sum
import itertools,ot,sys,os
sys.path.append(os.path.abspath( os.path.dirname(__file__) + '../WB' ))
#print("sys.path = ",sys.path)
from Tools import my_mkdir
from Fixed_RWB_reasonability import fixed_RWB_reasonability
time0 = time.strftime("%Y-%m-%d %H:%M:%S",time.localtime())



### data_size, zeta, mean, noise_var
data_size = 3000; kClusters = 60; zeta = 0.2; noise_mean = 150; noise_var = 150; 
epsilon = 50000; eta = 1; UOTRWBIter = 10; testtimes = 10; gamma_list = array([1,4,16,64,256,1024])

## point_lists,u_list,w_list,barycenter
cfd = os.path.dirname(__file__)
locations_weights = np.loadtxt(cfd+'kCluster60Kmeans.brainmatrix_2023_0830', dtype=np.float64, delimiter=' ')
locations_weights = locations_weights.reshape(int(locations_weights.shape[0]/60),60,5)
locations_weights = locations_weights[:data_size]
print("locations_weights = ",locations_weights.shape)
point_lists = locations_weights[:,:60,:4]
u_list = locations_weights[:,:60,-1] 
u_list = array([u/sum(u) for u in u_list])
print("u_list = ",u_list.shape)
w_list = ones(data_size)/data_size   
barycenter = array(point_lists[0])
barycenter[0] = noise_mean
print("point_lists,u_list = ",point_lists.shape,u_list.shape)



result_mean,result_std = fixed_RWB_reasonability(point_lists,u_list,barycenter,w_list,gamma_list,zeta,noise_mean,noise_var,epsilon,eta,UOTRWBIter, testtimes)





timeStamp = time.strftime("%m-%d_%H:%M",time.localtime())
cfd = os.path.dirname(__file__)
print("cfd = ",cfd)
foldName = 'HCP_' + '_' + str(data_size)+ '_'+ str(kClusters)+ '_'+ str(epsilon) +'_outlier_'+ str(zeta)+ '_' + str(noise_mean) + '_' + str(noise_var) + '_tt'+ str(testtimes) + '_' + str(timeStamp)
foldDir = cfd + 'results_fixed_WB_reasonability/' + foldName
my_mkdir(foldDir)
outputPointer = open(foldDir+'/others.txt','a+')
np.savetxt(foldDir + '/' + foldName +'.txt-ave',result_mean,fmt='%f',delimiter=' ',newline='\r\n')
np.savetxt(foldDir + '/' + foldName + '.txt-var',result_std,fmt='%f',delimiter=' ',newline='\r\n')
time1 = time.strftime("%Y-%m-%d %H:%M:%S",time.localtime())
print("time0,time1 = ", time0,time1,file=outputPointer )




print("-------------------------------------------------------",file=outputPointer )
print("data_size, kClusters             = ",data_size,kClusters,file=outputPointer)
print("zeta,noise_mean,noise_var        = ",zeta,noise_mean,noise_var,file=outputPointer)
print("epsilon,eta,UOTRWBIter,testtimes = ",epsilon,eta,UOTRWBIter,testtimes,file=outputPointer)
print("gamma_list                       = ",gamma_list,file=outputPointer) 







