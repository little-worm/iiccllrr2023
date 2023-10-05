


import random,time,ot,sys,os
import numpy as np
from numpy import array,ones,sum
sys.path.append(os.path.abspath( os.path.dirname(__file__) + '/..' ))
import sysPath
#print("sys.path = ",sys.path)
from Fixed_WB import WBbyFastIBP    
from Fixed_RWB import MyRobustWBbyCraftedFastIBP_toHa,WBbyRobustUOTbasedIBP_toHa




def cost_distributions(WB_locations:array,WB_weights:array,point_lists:array,u_list:array,w_list:array)->float:
    '''
    From:      myself
    Function:  compute the sum of emd between a fixed barycenter and inputs distributions
    parameter: WB_locations,WB_weights: the locations and weights of the fixed Wasserstein barycenter
               point_lists,u_list     : the locations and weights of the input distributions;           
    Return:    res                    : see "Function"
    '''
    WB_locations = array(WB_locations); WB_weights = array(WB_weights); point_lists = array(point_lists); u_list = array(u_list)
    if np.isnan(sum(WB_weights)):
        res = 0
    else:
        cost_list = array([ w*ot.emd2(WB_weights,u,ot.dist(WB_locations,points)) for u,points,w in zip(u_list,point_lists,w_list) ]) 
        res = sum(cost_list)
    return res,cost_list








def my_index(arr:array,target:float)->int:
    '''
    From:      myself
    Function:  compute the biggest "index" of array "arr" where the sume of "arr[:index]" is <= "target"
    parameter: arr   : the input array   
               target: See "Function"        
    Return:    index : see "Function"
    '''
    arr = array(arr)
    index = 0
    sum = 0
    for i in arr:
        sum = sum + i
        if sum > target:
            #print(" sum,target = ",sum,target)
            break
        index = index + 1
    return index    







def fixed_RWB_reasonability(point_lists,u_list,barycenter,w_list,gamma_list,zeta=0.1,noise_mean=28,noise_var=0,epsilon=1000,eta=1,UOTRWBIter=10,testtimes=5):
    '''
    From:      myself
    Function:  demonstrate the effectiveness of our "fixed-support Robust WB"
    parameter: point_lists,u_list         : the locations and weights of the input distributions;   
               barycenter                 : the locations of Wasserstein barycenter  
               gamma_list                 : a list of parameter for "UOT" to control the violation of outliers
               zeta, noise_mean, noise_var: the total proportion, the mean, the variance of outliers for given distribution   
               epsilon                    : the additive error we can tolerant for "our fixed-support RWB" and "FastIBP" 
               eta, UOTRWBIter            : parameters for UOT, is fixed in general;  
    Return:    result_mean,result_std     : the mean and variance matrix
    '''
    point_lists = array(point_lists); u_list = array(u_list); barycenter = array(barycenter); gamma_list = array(gamma_list)
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
        WB_weights,_ = WBbyFastIBP(point_lists,barycenter,u_list,w_list,epsilon)
        org_cost,_ = cost_distributions(barycenter,WB_weights,point_lists,u_list,w_list)
        cost_matrix_bb = ot.dist(barycenter,barycenter)

        ## baselibe1: use FastIBP compute noisy dataset
        time1 = time.process_time()
        noisyWB_weights,_ = WBbyFastIBP(noisy_point_lists,barycenter,u_list,w_list,epsilon)
        time11 = time.process_time()
        runtime1 = time11 - time1
        dis1 = ot.emd2(WB_weights,noisyWB_weights,cost_matrix_bb)
        cost1,_ = cost_distributions(barycenter,noisyWB_weights,point_lists,u_list,w_list)
        res_list.append(array([runtime1,dis1,cost1]))

        ## baseline2: use UOT compute noisy dataset
        for gamma in gamma_list:
            #print("----------UOOT-------gamma = ",gamma)
            time2 = time.process_time()
            UOTWB_weights,_,_,_ = WBbyRobustUOTbasedIBP_toHa(noisy_point_lists,barycenter,u_list,w_list,gamma,eta,UOTRWBIter)
            time22 = time.process_time()
            runtime2 = time22 - time2
            #print("WB_weights,UOTWB_weights = ",sum(WB_weights),sum(UOTWB_weights))
            if np.isnan(sum(UOTWB_weights)):
                print("-----------NAN------------")
                dis2 = 0
                cost2 = 0
            else:    
                dis2 = ot.emd2(WB_weights,UOTWB_weights,cost_matrix_bb)
                cost2,_ = cost_distributions(barycenter,UOTWB_weights,point_lists,u_list,w_list)
            res_list.append(array([runtime2,dis2,cost2]))

        ## baseline3: our robust  AWB
        #print("-----our RWB--------------------------------")
        time3 = time.process_time()
        myRWB_weights,_,_,_,_ = MyRobustWBbyCraftedFastIBP_toHa(noisy_point_lists,barycenter,u_list,w_list,epsilon,zeta)
        time33 = time.process_time()
        runtime3 = time33 - time3
        dis3 =  ot.emd2(WB_weights,myRWB_weights,cost_matrix_bb)
        cost3,_ = cost_distributions(barycenter,myRWB_weights,point_lists,u_list,w_list)
        res_list.append(array([runtime3,dis3,cost3]))
        AllReault_list.append(res_list)
        ## index
    AllReault_list = array(AllReault_list)
    result_mean = sum(AllReault_list,axis=0) / testtimes
    result_var = sum(AllReault_list**2,axis=0) / testtimes - result_mean**2
    result_std = result_var**(0.5)
    return result_mean,result_std



