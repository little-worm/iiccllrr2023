import numpy as np 
from numpy import random ,ones,outer,array,exp,copy,sum,zeros,dot,linalg,log,sqrt,abs,diag,mean,ceil,round,floor
from ot import dist
import random,time
from Fixed_RWB import MyRobustWBbyCraftedFastIBP_toHa, WBbyRobustUOTbasedIBP_toHa
from Fixed_WB import WBbyFastIBP



def update_position_for_WB(point_lists:array,X_list:array,w_list:array)->array:
    '''
    From:      our free support WB 
    Function:  update the position of "free support WB"
    parameter: point_lists     : the location lists of input distributions;    ndarray((n_1,d),(n_2,d),...,(n_m,d))
               X_list          : the transportation plan list of WB 
               w_list          : the weight for input distributions                 ndarray(m)
    Note:      there are "m" input  distributions; "n" denotes the support size of barycenter;          
    Return:    new_BCpoint_list: update the position of "free support WB"
    '''        
    point_lists = array(point_lists); X_list = array(X_list); w_list = array(w_list)
    normal_X_list = (array( [ X/sum(X,axis=0)  for X in X_list ] ))
    new_BCpoint_lists = array([ dot(w*X.T,array(pl)) for w,pl,X in zip(w_list,point_lists,normal_X_list)])
    new_BCpoint_list = sum(new_BCpoint_lists,axis = 0)
    return new_BCpoint_list






#   #--------------------------------------------------------------------------
#   # point_lists
#   point_lists = [[[1,1],[2,3],[5,6],[5,5],[4,4]],[[1,5],[6,6],[1,4],[3,5],[4,7]],[[2,3],[3,3],[5,6],[4,2],[3,4]]]
#   # barycenter
#   barycenter = array([[2,2],[4,5],[6,5],[1,4]])
#   # u_list
#   print("point_lists = ",point_lists)
#   N_list = array([array(pl).shape[0] for pl in point_lists])
#   print("N_list = ",N_list)
#   u_list = [np.random.random(N) for N in N_list]
#   u_list = [u / sum(u) for u in u_list]
#   print("u_list = ",u_list)
#   # w_list
#   w_list = array(np.random.random(len(point_lists)))
#   w_list = w_list / sum(w_list)
#   print("w_list = ",w_list)
#   # X_list
#   X_list = array([np.random.random(size=(n,barycenter.shape[0])) for n in N_list])
#   print("X_list = ",X_list[0].shape)
#   new_BCpoint_list = update_position_for_WB(point_lists,X_list,w_list)
#   print("new_BCpoint_list = ",new_BCpoint_list)
#   #--------------------------------------------------------------------------









def MyFreeSupport_RobustWBbyCraftedFastIBP_toHa(point_lists:array,initBarycenter:array,u_list:array,w_list:array,epsilon:float,zeta:float,freeSupportIter:int)->any:
    '''
    From:      our free support robust WB 
    Function:  my free support Robust WB by CraftedFastIBP
    parameter: point_lists    : the location lists of input distributions;    ndarray((n_1,d),(n_2,d),...,(n_m,d))
               initBarycenter : the initial location list of barycenter;               ndarray(n,d)  
               u_list         : the lists of  simplex distributions function;      ndarray(n_1,n_2,...,m_m)  
               w_list         : the weight for input distributions                 ndarray(m)
               epsilon        : the additive error we can tolerant
               zeta_a,zeta_b  : the total mass of outliers for given distribution and barycenter
               freeSupportIter：the number of iteration times we set
    Note:      there are "m" input  distributions; The support size of each distributions can be different; "n" denotes the support size of barycenter;          
    Return:    robust_res_u         : probability simplex  of barycenter of RWB (without dummy point)
               extended_robust_res_u: probability simplex  of barycenter of RWB (with dummy point)
               robust_X_list        : the transportation plan list (in feasible domain) 
               barycenter           : the final position of RBC (without dummy point)
    '''    
    time0 = time.process_time()
    point_lists = array(point_lists); barycenter = array(initBarycenter); u_list = array(u_list); w_list = array(w_list)
    for i in range(freeSupportIter):
        robust_res_u,extended_robust_res_u,robust_X_list,cost,runtime = MyRobustWBbyCraftedFastIBP_toHa(point_lists,barycenter,u_list,w_list,epsilon,zeta)
        new_BCpoint_list = update_position_for_WB(point_lists,robust_X_list,w_list)
        barycenter = new_BCpoint_list[ :-1]
    time1 = time.process_time()
    runtime = time1 - time0    
    return robust_res_u,extended_robust_res_u,robust_X_list,barycenter,new_BCpoint_list,cost,runtime




#    #--------------------------------------------------------------------------
#    point_lists = [[[1,1],[2,3],[5,6],[5,5],[4,4]],[[1,5],[6,6],[1,4],[3,5],[4,7]],[[2,3],[3,3],[5,6],[4,2],[3,4]]]
#    # point_lists
#    #point_lists = array( [dot(pl,100) for pl in point_lists] )
#    # barycenter
#    barycenter = array([[2,2],[4,5],[6,5],[1,4]])
#    # u_list
#    print("point_lists = ",point_lists)
#    N_list = array([array(pl).shape[0] for pl in point_lists])
#    print("N_list = ",N_list)
#    u_list = array([np.random.random(N) for N in N_list])
#    u_list = array([u / sum(u) for u in u_list])
#    print("u_list = ----------",u_list.shape)
#    # w_list
#    w_list = array(np.random.random(array(point_lists).shape[0]))
#    w_list = w_list / sum(w_list)
#    print("w_list = ",w_list)
#    # epsilon
#    epsilon = 200
#    #test function
#    zeta = 0.1
#    freeSupportIter = 10
#    robust_res_u,extended_robust_res_u,robust_X_list,barycenter,new_BCpoint_list,cost,runtime = MyFreeSupport_RobustWBbyCraftedFastIBP_toHa(point_lists,barycenter,u_list,w_list,epsilon,zeta,freeSupportIter)
#    print("res = ",res)
#    print("-------------------------------------------------------------------")
#    #   #--------------------------------------------------------------------------















def MyFreeSupport_RobustWBbyUOT_toHa(point_lists:array,initBarycenter:array,u_list:array,w_list:array,gamma:float,eta:float,UOTRWBIter:int,freeSupportIter:int)->any:
    '''
    From:      our free support robust WB 
    Function:  my free support Robust WB by CraftedFastIBP
    parameter: point_lists: the location lists of input distributions;    ndarray((n_1,d),(n_2,d),...,(n_m,d))
               initBarycenter: the initial location list of barycenter;               ndarray(n,d)  
               u_list: the lists of  simplex distributions function;      ndarray(n_1,n_2,...,m_m)  
               w_list: the weight for input distributions                 ndarray(m)
               gamma, eta, UOTRWBIter: parameters of UOT
               freeSupportIter： the number of iteration times we set
    Note:      there are "m" input  distributions; The support size of each distributions can be different; "n" denotes the support size of barycenter;          
    Return:    robust_res_u: probability simplex  of barycenter of RWB (without dummy point)
               robust_X_list: the transportation plan list (in feasible domain) 
               barycenter: the final position of RWB 
    '''    
    time0 = time.process_time()
    point_lists = array(point_lists); barycenter = array(initBarycenter); u_list = array(u_list); w_list = array(w_list)
    for i in range(freeSupportIter):
        #print("barycenter = ",barycenter)
        robust_res_u,robust_X_list,cost,runtime = WBbyRobustUOTbasedIBP_toHa(point_lists,barycenter,u_list,w_list,gamma,eta,UOTRWBIter)
        barycenter = update_position_for_WB(point_lists,robust_X_list,w_list)
    time1 = time.process_time()
    runtime = time1 - time0    
    return robust_res_u,robust_X_list,barycenter,cost,runtime




#    point_lists = [[[1,1],[2,3],[5,6],[5,5],[4,4]],[[1,5],[6,6],[1,4],[3,5],[4,7]],[[2,3],[3,3],[5,6],[4,2],[3,4]]]
#    #--------------------------------------------------------------------------
#    # point_lists
#    #point_lists = array( [dot(pl,100) for pl in point_lists] )
#    
#    # barycenter
#    barycenter = array([[2,2],[4,5],[6,5],[1,4]])
#    # u_list
#    print("point_lists = ",point_lists)
#    N_list = array([array(pl).shape[0] for pl in point_lists])
#    print("N_list = ",N_list)
#    u_list = array([np.random.random(N) for N in N_list])
#    u_list = array([u / sum(u) for u in u_list])
#    print("u_list = ----------",u_list.shape)
#    # w_list
#    w_list = array(np.random.random(array(point_lists).shape[0]))
#    w_list = w_list / sum(w_list)
#    print("w_list = ",w_list)
#    # epsilon
#    epsilon = 200
#    #test function
#    zeta = 0.1
#    gamma = 64
#    eta = 1
#    UOTRWBIter = 10
#    freeSupportIter = 10
#    initBarycenter = barycenter
#    res = MyFreeSupport_RobustWBbyUOT_toHa(point_lists,initBarycenter,u_list,w_list,gamma,eta,UOTRWBIter,freeSupportIter)
#    #print("res = ",res)
#    print("-------------------------------------------------------------------")
#    #   #--------------------------------------------------------------------------










def MyFreeSupport_FastIBP_toHa(point_lists:array,initBarycenter:array,u_list:array,w_list:array,epsilon:float,freeSupportIter:int)->any:
    '''
    From:      our free support robust WB 
    Function:  my free support Robust WB by CraftedFastIBP
    parameter: point_lists: the location lists of input distributions;    ndarray((n_1,d),(n_2,d),...,(n_m,d))
               initBarycenter: the initial location list of barycenter;               ndarray(n,d)  
               u_list: the lists of  simplex distributions function;      ndarray(n_1,n_2,...,m_m)  
               w_list: the weight for input distributions                 ndarray(m)
               gamma, eta, UOTRWBIter: parameters of UOT
               freeSupportIter： the number of iteration times we set for free-support RWB
    Note:      there are "m" input  distributions; The support size of each distributions can be different; "n" denotes the support size of barycenter;          
    Return:    res_u: probability simplex  of barycenter of RWB (without dummy point)
               X_list: the transportation plan list (in feasible domain) 
               barycenter: the final position of RWB ========================================================================================
    '''    
    time0 = time.process_time()
    point_lists = array(point_lists); barycenter = array(initBarycenter); u_list = array(u_list); w_list = array(w_list)
    for i in range(freeSupportIter):
        #print("barycenter = ",barycenter)
        res_u,X_list = WBbyFastIBP(point_lists,barycenter,u_list,w_list,epsilon)
        barycenter = update_position_for_WB(point_lists,X_list,w_list)
    time1 = time.process_time()
    runtime = time1 - time0    
    return res_u,X_list,barycenter,runtime




point_lists = [[[1,1],[2,3],[5,6],[5,5],[4,4]],[[1,5],[6,6],[1,4],[3,5],[4,7]],[[2,3],[3,3],[5,6],[4,2],[3,4]]]
#--------------------------------------------------------------------------
# point_lists
#point_lists = array( [dot(pl,100) for pl in point_lists] )

# barycenter
barycenter = array([[2,2],[4,5],[6,5],[1,4]])
# u_list
print("point_lists = ",point_lists)
N_list = array([array(pl).shape[0] for pl in point_lists])
print("N_list = ",N_list)
u_list = array([np.random.random(N) for N in N_list])
u_list = array([u / sum(u) for u in u_list])
print("u_list = ----------",u_list.shape)
# w_list
w_list = array(np.random.random(array(point_lists).shape[0]))
w_list = w_list / sum(w_list)
print("w_list = ",w_list)
# epsilon
epsilon = 200
#test function
zeta = 0.1
gamma = 64
eta = 1
UOTRWBIter = 10
freeSupportIter = 10
initBarycenter = barycenter
res_u,X_list,barycenter,runtime = MyFreeSupport_FastIBP_toHa(point_lists,initBarycenter,u_list,w_list,epsilon,freeSupportIter)
print("res = ",res_u,X_list,barycenter,runtime)
print("-------------------------------------------------------------------")
#   #--------------------------------------------------------------------------

