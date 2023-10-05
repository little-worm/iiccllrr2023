import random,time,ot,sys,os
import numpy as np
from numpy import array,ones,sum
sys.path.append(os.path.abspath( os.path.dirname(__file__) + '/..' ))
import sysPath
#print("sys.path = ",sys.path)
from Free_WB import MyFreeSupport_RobustWBbyCraftedFastIBP_toHa,MyFreeSupport_FastIBP_toHa
from Fixed_RWB_reasonability import my_index,cost_distributions



def free_RWB_reasonability(point_lists,u_list,barycenter,w_list,zeta=0.1,noise_mean=28,noise_var=0,epsilon=1000,freeSupportIter = 5,testtimes=5):
    '''
    From:      myself
    Function:  demonstrate the effectiveness of our "fixed-support Robust WB"
    parameter: point_lists,u_list         : the locations and weights of the input distributions;   
               barycenter                 : the locations of Wasserstein barycenter  
               zeta, noise_mean, noise_var: the total proportion, the mean, the variance of outliers for given distribution   
               epsilon                    : the additive error we can tolerant for "our fixed-support RWB" and "FastIBP" 
    Return:    result_mean,result_std     : the mean and variance matrix
    '''
    point_lists = array(point_lists); u_list = array(u_list); barycenter = array(barycenter)
    AllReault_list = []
    for tt in range(testtimes):
        ## add "zeta" noise for every input distribution
        noise_dim = point_lists.shape[2]
        noisy_point_lists = np.copy(point_lists)
        for points,u in zip(noisy_point_lists,u_list):
            tmp_index = my_index(u,zeta)
            tmp_noise = np.random.normal(noise_mean,noise_var,size=(tmp_index,noise_dim))
            points[:tmp_index,:] = tmp_noise


        res_list = []
        ## the "WB" of "orginal dataset" 
        ##WB_weights,_ = WBbyFastIBP(point_lists,barycenter,u_list,w_list,epsilon)
        initBarycenter = random.choices(point_lists)[0]
        org_WB_weights,_,org_barycenter,_ = MyFreeSupport_FastIBP_toHa(point_lists,initBarycenter,u_list,w_list,epsilon,freeSupportIter)

        org_cost,_ = cost_distributions(org_barycenter,org_WB_weights,point_lists,u_list,w_list)

        ## baselibe1: use FastIBP compute noisy dataset
        time1 = time.process_time()
        ##noisyWB_weights,_ = WBbyFastIBP(noisy_point_lists,barycenter,u_list,w_list,epsilon)
        noisy_WB_weights_0,_,noisy_barycenter_0,_ = MyFreeSupport_FastIBP_toHa(noisy_point_lists,initBarycenter,u_list,w_list,epsilon,freeSupportIter)
        time11 = time.process_time()
        runtime1 = time11 - time1
        dis1 = ot.emd2(org_WB_weights,noisy_WB_weights_0,ot.dist(org_barycenter,noisy_barycenter_0))
        cost1,_ = cost_distributions(noisy_barycenter_0,noisy_WB_weights_0,point_lists,u_list,w_list)
        res_list.append(array([runtime1,dis1,cost1]))

        ## baseline3: our robust  AWB
        #print("-----our RWB--------------------------------")
        time3 = time.process_time()
        ##myRWB_weights,_,_,_,_ = MyRobustWBbyCraftedFastIBP_toHa(noisy_point_lists,barycenter,u_list,w_list,epsilon,zeta)
        myrobust_weights,_,_,myrobust_barycenter,_,_,_ = MyFreeSupport_RobustWBbyCraftedFastIBP_toHa(noisy_point_lists,initBarycenter,u_list,w_list,epsilon,zeta,freeSupportIter)

        time33 = time.process_time()
        runtime3 = time33 - time3
        dis3 =  ot.emd2(org_WB_weights,myrobust_weights,ot.dist(org_barycenter,myrobust_barycenter))
        cost3,_ = cost_distributions(myrobust_barycenter,myrobust_weights,point_lists,u_list,w_list)
        res_list.append(array([runtime3,dis3,cost3]))
        AllReault_list.append(res_list)
        ## index
    AllReault_list = array(AllReault_list)
    result_mean = sum(AllReault_list,axis=0) / testtimes
    result_var = sum(AllReault_list**2,axis=0) / testtimes - result_mean**2
    result_std = result_var**(0.5)
    return result_mean,result_std



