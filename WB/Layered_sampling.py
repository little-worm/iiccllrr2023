import numpy as np
import time,random
from numpy import array,floor,ceil,log2,copy,zeros,arange,ones,append,sum
from Fixed_RWB_reasonability import cost_distributions
from ot import dist,emd2
from Fixed_RWB import MyRobustWBbyCraftedFastIBP_toHa
from Fixed_RWB_reasonability import my_index,cost_distributions
from Free_WB import MyFreeSupport_FastIBP_toHa,MyFreeSupport_RobustWBbyCraftedFastIBP_toHa
from Tools import myListsSlice


#   #--------------------------------------------------------------------------
#   # point_lists
#   point_lists = [[[1,1],[2,3],[5,6],[5,5],[4,4]],[[1,5],[6,6],[1,4],[3,5],[4,7]],[[2,3],[3,3],[5,6],[4,2],[3,4]]]
#   point_lists = np.random.random(size=(20,5,1)) * 100
#   # barycenter
#   barycenter = array([[2,2],[4,5],[6,5],[1,4]])
#   # u_list
#   N_list = array([array(pl).shape[0] for pl in point_lists])
#   u_list = [np.random.random(N) for N in N_list]
#   u_list = [u / sum(u) for u in u_list]
#   # w_list
#   w_list = array(np.random.random(len(point_lists)))
#   w_list = w_list / sum(w_list)







def myRobustWD(outlier_measure_locations:array,outlier_measure_weights:array,BC_locations:array,BC_weights:array,zeta:float)->float:
    '''
    From:      myself
    Function:  compute the robust Wasserstein distance between a noisy measure and a clear measure
    parameter: outlier_measure_locations,outlier_measure_weights: the locations and weights of the boisy measure 
               point_lists,u_list                               : the locations and weights of the clear measure;           
    Return:    res                                              : see "Function"
    '''
    outlier_measure_locations = array(outlier_measure_locations); outlier_measure_weights = array(outlier_measure_weights)
    BC_locations = array(BC_locations); BC_weights = array(BC_weights)
    aug_distMatrix = zeros((outlier_measure_locations.shape[0],BC_locations.shape[0]+1))
    aug_distMatrix[:,:outlier_measure_locations.shape[0]] = dist(outlier_measure_locations,BC_locations)
    aug_outlier_measure_weights = outlier_measure_weights / (1-zeta)
    aug_BC_weights = np.append(BC_weights, zeta/(1-zeta))
    res = emd2(aug_outlier_measure_weights,aug_BC_weights,aug_distMatrix,zeta)
    return res





#   ##------------test for ----------myRobustWD()------------------------------
#   print("##------------test for ----------myRobustWD()------------------------------")
#   outlier_measure_weights = array(u_list[0])
#   outlier_measure_locations = array(point_lists[0])
#   BC_locations = array(point_lists[1])
#   BC_weights = array(u_list[1])
#   zeta = 0.2
#   res = myRobustWD(outlier_measure_locations,outlier_measure_weights,BC_locations,BC_weights,zeta)
#   print("res = " ,res)






#   def findApproxSolutionOfFreeRWB(point_lists:array,u_list:array,w_list:array,repeatTimes:int,epsilon:float,zeta:float,tmpSampleSize = 500)->array:
#       '''
#       From:      myself
#       Function:  compute the constant approxiamte solution of free-support robust Wasserstein barycenter 
#       parameter: point_lists,u_list: the locations and weights of the input distributions;
#                  w_list            : the weights for input probability distributions    
#                  repeatTimes       : control the failure probability less than  (\delta)^(repeatTimes) 
#                  epsilon           : the additive error we can tolerant
#                  zeta              : the total weight of outlier in input distributions
#       Return:    target_points,target_weight: the locations and weights of the approxiamte solution 
#       '''
#       point_lists = array(point_lists); u_list = array(u_list); w_list = array(w_list)
#       target_value = 10000000
#       target_points = point_lists[0]; target_weight = u_list[0]
#       solution_pointsList = random.choices(point_lists, k=repeatTimes )
#       for points in solution_pointsList:
#           tmp_indexList = random.choices(arange(point_lists.shape[0]),k=tmpSampleSize)
#           tmp_point_lists,tmp_u_list,tmp_w_list = myListsSlice( [point_lists,u_list,w_list], tmp_indexList )
#           tmp_BC_weight,_,_,tmp_value,_ = MyRobustWBbyCraftedFastIBP_toHa(tmp_point_lists,points,tmp_u_list,tmp_w_list,epsilon,zeta)
#           if tmp_value < target_value:
#               target_value = tmp_value
#               target_points = points
#               target_weight = tmp_BC_weight
#               #print("target_points,target_weight = ",target_points,target_weight)
#       return target_points,target_weight






#   def findApproxSolutionOfFreeRWB(point_lists:array,u_list:array,w_list:array,repeatTimes:int,epsilon:float,zeta:float)->array:
#       '''
#       From:      myself
#       Function:  compute the constant approxiamte solution of free-support robust Wasserstein barycenter 
#       parameter: point_lists,u_list: the locations and weights of the input distributions;
#                  w_list            : the weights for input probability distributions    
#                  repeatTimes       : control the failure probability less than  (\delta)^(repeatTimes) 
#                  epsilon           : the additive error we can tolerant
#                  zeta              : the total weight of outlier in input distributions
#       Return:    target_points,target_weight: the locations and weights of the approxiamte solution 
#       '''
#       print("------------start findApproxSolutionOfFreeRWB()-----------------")
#       point_lists = array(point_lists); u_list = array(u_list); w_list = array(w_list)
#       target_value = 10000000
#       target_points = point_lists[0]; target_weight = u_list[0]
#       #   solution_pointsList = random.choices(point_lists, k=repeatTimes )
#       for points in point_lists[:repeatTimes]:
#           #   tmp_indexList = random.choices(arange(point_lists.shape[0]),k=tmpSampleSize)
#           #   tmp_point_lists,tmp_u_list,tmp_w_list = myListsSlice( [point_lists,u_list,w_list], tmp_indexList )
#           tmp_BC_weight,_,_,tmp_value,_ = MyRobustWBbyCraftedFastIBP_toHa(point_lists,points,u_list,w_list,epsilon,zeta)
#           if tmp_value < target_value:
#               target_value = tmp_value
#               target_points = points
#               target_weight = tmp_BC_weight
#       print("target_points,target_weight = ",target_points,target_weight)
#       print("------------end   findApproxSolutionOfFreeRWB()-----------------")
#       return target_points,target_weight







def findApproxSolutionOfFreeRWB(point_lists:array,u_list:array,w_list:array,repeatTimes:int,epsilon:float,zeta:float)->array:
    '''
    From:      myself
    Function:  compute the constant approxiamte solution of free-support robust Wasserstein barycenter 
    parameter: point_lists,u_list: the locations and weights of the input distributions;
               w_list            : the weights for input probability distributions    
               repeatTimes       : control the failure probability less than  (\delta)^(repeatTimes) 
               epsilon           : the additive error we can tolerant
               zeta              : the total weight of outlier in input distributions
    Return:    target_points,target_weight: the locations and weights of the approxiamte solution 
    '''
    print("------------start findApproxSolutionOfFreeRWB()-----------------")
    point_lists = array(point_lists); u_list = array(u_list); w_list = array(w_list)
    target_value = 10000000
    target_points = point_lists[0]; target_weight = u_list[0]
    for points in point_lists[:repeatTimes]:
        tmp_BC_weight,_,_,tmp_value,_ = MyRobustWBbyCraftedFastIBP_toHa(point_lists,points,u_list,w_list,epsilon,zeta)
        if tmp_value < target_value:
            target_value = tmp_value
            target_points = points
            target_weight = tmp_BC_weight
    #print("target_points,target_weight = ",target_points,target_weight)
    print("------------end   findApproxSolutionOfFreeRWB()-----------------")
    return target_points,target_weight










#   ##------------test for ----------findApproxSolutionOfFreeRWB()------------------------------
#   print("------------test for ----------findApproxSolutionOfFreeRWB()------------------------------")
#   epsilon = 1000; zeta = 0.2
#   repeatTimes = 4
#   res = findApproxSolutionOfFreeRWB(point_lists,u_list,w_list,repeatTimes,epsilon,zeta)
#   print("res = ",res)






def myLayerNum(cost:float,Radius:float,logbase:float)->int:
    '''
    From:      myself
    Function:  compute "index of layer" by the offered "cost" and "Radius" in "layered sampling" 
    Return:    see Function
    '''    
    layer_num = ceil(log2(cost/Radius) / log2(logbase) )
    if layer_num < 0:
        layer_num = 0
    return int(layer_num)



#   print("------------test for ----------myLayerNum()------------------------------")
#   cost = np.random.randint(0,100)
#   Radius = 10; logbase = 1.5
#   res = myLayerNum(cost,Radius,logbase)
#   print("res = ",res)





def layerPartition(point_lists:array,u_list:array,w_list:array,repeatTimes:int,Radius:float,logbase:float,epsilon:float,zeta:float)->list: 
    '''
    From:      myself
    Function:  Partition the dataset for "layered sampling" 
    parameter: point_lists,u_list: the locations and weights of the input distributions;
               w_list            : the weights for input probability distributions    
               repeatTimes       : control the failure probability less than  (\delta)^(repeatTimes) 
               Radius            : the radius of the innermost layer for "layered sampling"
               epsilon           : the additive error we can tolerant
               zeta              : the total weight of outlier in input distributions
    Return:    see Function 
    '''
    point_lists = array(point_lists); u_list = array(u_list); w_list = array(w_list) 
    approxSolution = findApproxSolutionOfFreeRWB(point_lists,u_list,w_list,repeatTimes,epsilon,zeta)
    robust_cost_list = array( [myRobustWD(points,u,approxSolution[0],approxSolution[1],zeta) for points,u in zip(point_lists,u_list)] )
    max_cost = max(robust_cost_list)
    total_layer_num = myLayerNum(max_cost,Radius,logbase) + 1
    layerPoints_list = [[] for i in range(total_layer_num)]
    layerWeights_list = [[] for i in range(total_layer_num)]
    layerW_list = [[] for i in range(total_layer_num)]
    for c,points,u,w in zip(robust_cost_list,point_lists,u_list,w_list):
        tmp_layerNum = myLayerNum(c,Radius,logbase)
        layerPoints_list[tmp_layerNum].append(points)
        layerWeights_list[tmp_layerNum].append(u) 
        layerW_list[tmp_layerNum].append(w)

    nonEmptyLayerIndex = [index for index in range(len(layerPoints_list)) if len(layerPoints_list[index]) > 0] 
    ##  print("nonEmptyLayerIndex = ",nonEmptyLayerIndex) 
    ##  print("===============",len(layerPoints_list),len(layerWeights_list),len(layerW_list),len(nonEmptyLayerIndex))  
    layerPoints_list,layerWeights_list,layerW_list = myListsSlice([layerPoints_list,layerWeights_list,layerW_list],nonEmptyLayerIndex)
    layerPoints_list = [array(arr) for arr in layerPoints_list]    
    layerWeights_list = [array(arr) for arr in layerWeights_list]    
    layerW_list = [array(arr) for arr in layerW_list]    
    ##  nonEmptyLayerIndex = [index for index in range(len(layerPoints_list)) if len(layerPoints_list[index]) > 0] 
    ##  print("nonEmptyLayerIndex = ",nonEmptyLayerIndex) 
    return layerPoints_list,layerWeights_list,layerW_list,approxSolution



#   epsilon = 1000; zeta = 0.2
#   repeatTimes = 4
#   Radius = 10
#   layerPoints_list,layerWeights_list,layerW_list,_ = layerPartition(point_lists,u_list,w_list,repeatTimes,Radius,logbase,epsilon,zeta)
#   #print("layerPoints_list,layerWeights_list,layerW_list = ")
#   #print(layerPoints_list,layerWeights_list,layerW_list)
#   tmp_arr = [len(ll) for ll in layerWeights_list]
#   print("tmp_arr =", tmp_arr,sum(tmp_arr))






def layeredSamplingTest(point_lists,u_list,w_list,sampleSizeList,zeta,noise_mean,noise_var,repeatTimes,Radius,logbase,epsilon,freeSupportIter,testtimes):
    '''
    From:      myself
    Function:  compare the efficiency for "layered sampling" and "uniform sampling"
    parameter: point_lists,u_list: the locations and weights of the input distributions;
               w_list            : the weights for input probability distributions
               zeta, noise_mean, noise_var: the total proportion, the mean, the variance of outliers for given distribution       
               repeatTimes       : control the failure probability less than  (\delta)^(repeatTimes) 
               Radius            : the radius of the innermost layer for "layered sampling"
               epsilon           : the additive error we can tolerant
               freeSupportIter： the number of iteration times we set for free-support RWB
    Return:    see Function 
    '''
    point_lists = array(point_lists); u_list = array(u_list); w_list = array(w_list)
    AllReault_list = []
    for tt in range(testtimes):
        print("================================================================================================== tt = ",tt)
        ## add "zeta" noise for every input distribution
        noise_dim = point_lists.shape[2]
        noisy_point_lists = np.copy(point_lists)
        for points,u in zip(noisy_point_lists,u_list):
            tmp_index = my_index(u,zeta)
            tmp_noise = np.random.normal(noise_mean,noise_var,size=(tmp_index,noise_dim))
            points[:tmp_index,:] = tmp_noise
        time2 = time.process_time()
        layerPoints_list,layerWeights_list,layerW_list,approxSolution = layerPartition(point_lists,u_list,w_list,repeatTimes,Radius,logbase,epsilon,zeta)
        time22 = time.process_time()
        res_list = []
        initBarycenter = random.choices(point_lists)[0]
        ##initBarycenter = noisy_point_lists[random.randint(0,noisy_point_lists.shape[0])]
        org_WB_weights,_,org_barycenter,_ = MyFreeSupport_FastIBP_toHa(point_lists,initBarycenter,u_list,w_list,epsilon,freeSupportIter)
        org_cost,_ = cost_distributions(org_barycenter,org_WB_weights,point_lists,u_list,w_list)
        ## baseline1: our robust  AWB on original dadaset
        time0 = time.process_time()
        initBarycenter = random.choices(noisy_point_lists)[0]
        robust_WB_weight,_,_,robust_barycenter,_,_,_ = MyFreeSupport_RobustWBbyCraftedFastIBP_toHa(noisy_point_lists,initBarycenter,u_list,w_list,epsilon,zeta,freeSupportIter)
        time00 = time.process_time()
        runtime0 = time00 - time0
        dis0 = emd2(robust_WB_weight,org_WB_weights,dist(robust_barycenter,org_barycenter))
        cost0,_ = cost_distributions(robust_barycenter,robust_WB_weight,point_lists,u_list,w_list)

        aug_sampleSizeList = copy(sampleSizeList)
        for ss in range(sampleSizeList.shape[0]*2): ## Sample
            time1 = time.process_time()
            if ss < sampleSizeList.shape[0]: ## point_lists for a layered sampling
                time1 = time1 - (time22 - time2)
                tmp = 0
                tmp_point_lists = []
                tmp_u_list = []
                tmp_w_list = []
                for pl,wel,wl in zip(layerPoints_list,layerWeights_list,layerW_list):
                    if list(pl) == list([]):
                        continue 
                    pl_index_list = arange(pl.shape[0])
                    tmp_index_list = random.choices( array(pl_index_list), wl, k = sampleSizeList[ss] )
                    tmp = myListsSlice( [pl,wel],tmp_index_list )
                    tmp1 = tmp[0]; tmp2 = tmp[1]
                    tmp3 = list(ones(sampleSizeList[ss]) * sum(wl) / aug_sampleSizeList[ss])
                    tmp_point_lists = tmp_point_lists + tmp1
                    tmp_u_list = tmp_u_list + tmp2
                    tmp_w_list = tmp_w_list + tmp3
                aug_sampleSizeList = append(aug_sampleSizeList,array(tmp_point_lists).shape[0])
            else:
                tmp_index_list = random.choices( range(point_lists.shape[0]), ones( point_lists.shape[0] ), k = aug_sampleSizeList[ss] )

                tmp_point_lists,tmp_u_list = myListsSlice( [point_lists,u_list],tmp_index_list )
                tmp_w_list = ones( array(tmp_index_list).shape[0] ) / array(tmp_index_list).shape[0]
            ## several sampling
            #print("-----our RWB--------------------------------")
            tmp_Robust_WB_weights,_,_,tmp_robust_barycenter,_,_,_ = MyFreeSupport_RobustWBbyCraftedFastIBP_toHa(tmp_point_lists,initBarycenter,tmp_u_list,tmp_w_list,epsilon,zeta,freeSupportIter)
            time11 = time.process_time()
            runtime1 = time11 - time1


            dis1 =  emd2(tmp_Robust_WB_weights,org_WB_weights,dist(tmp_robust_barycenter,org_barycenter))
            cost1,_ = cost_distributions(tmp_robust_barycenter,tmp_Robust_WB_weights,point_lists,u_list,w_list)
            print("dis1,cost1 = ",dis1,cost1)
            res_list.append(array([runtime1,dis1,cost1]))
        AllReault_list.append(res_list)
        ## index
    AllReault_list = array(AllReault_list)
    result_mean = sum(AllReault_list,axis=0) / testtimes
    t1 = copy(AllReault_list)
    result_var = sum(AllReault_list**2,axis=0) / testtimes 
    t2 = copy(AllReault_list)
    result_var = result_var - result_mean**2
    result_std = result_var**(0.5)
    E_X2 = sum(AllReault_list**2,axis=0) / testtimes 
    Ex_2 = result_mean**2
    t3  = testtimes
    result_mean = np.hstack((result_mean[:sampleSizeList.shape[0]],result_mean[sampleSizeList.shape[0]:]))
    result_std =  np.hstack((result_std[:sampleSizeList.shape[0]],result_std[sampleSizeList.shape[0]:]))
    result_var =  np.hstack((result_var[:sampleSizeList.shape[0]],result_var[sampleSizeList.shape[0]:]))

    return result_mean,result_std,result_var,layerPoints_list,layerWeights_list,layerW_list





#   testtimes = 2
#   zeta = 0.1
#   noise_mean = 30; noise_var = 30
#   epsilon = 1000
#   freeSupportIter = 2
#   repeatTimes = 2
#   Radius = 100
#   sampleSizeList = array([10,20,30,40,50,60])
#   logbase = 1.5
#   result_mean,result_std,result_var,layerPoints_list,_,_ = layeredSamplingTest(point_lists,u_list,w_list,sampleSizeList,zeta,noise_mean,noise_var,repeatTimes,Radius,logbase,epsilon,freeSupportIter,testtimes)
#   
#   print("result_mean,result_std,result_var,layerPoints_list = ")
#   print(result_mean,result_std,result_var,layerPoints_list)

