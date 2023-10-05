from typing_extensions import runtime
import numpy as np 
from numpy import random ,ones,outer,array,exp,copy,sum,zeros,dot,linalg,log,sqrt,abs,diag,mean,ceil,round,floor
from ot import dist
import random
import time
from Fixed_WB import rBcBfun_ofWB, RoundToFeasibleSolution_ofWB, Bfun_ofOT
myError = 0.1; MAXcou = 8


def wx2Dlog(M):
    '''
    From:      worm     
    Function:  compute "log(M)" elementwise for 2D array "M". This function is designed for the case "M[i].shape != M[j].shape"  
    parameter: M  : a 2D array
    Return:    res: the "log(M)"          
    '''
    M = copy(M)
    res = array([log(m) for m in M])
    return res








def CraftedFastIBP(costMatrix_list:array,u_list:array,w_list:array,epsilon:int,eta:int,zeta:int)->array:    
    '''
    From:     Algorithm 1 of "Fixed-Support Wasserstein Barycenters: Computational Hardness and Fast Algorithm" 
              designed for our "crafted RWB"; that is, we fixed the weight of outlier of barycenter.  e.g., b[-1] = zeta_ofBC = zeta / (1-zeta)
    Function: CraftedFastIBP
    parameter:costMatrix_list: cost matrix list between  "input distributions and barycenter" (行数代表输入的support的size，列数代表barycenter的support的size)
              u_list         : input simplex list of input distributions ;      ndarray(n_1,n_2,...,m_m)  
              w_list         : the weights for input probability distributions;                ndarray(m)
              epsilon        : the additive error we can tolerant
              eta            : on entropic regularization term for Sinkhorn algorithm
              zeta         : the total weight of outlier in input distributions
    Return:   res_B_list     :  the transportation plan list (maybe not in feasible domain)                                         
    '''
    costMatrix_list = array(costMatrix_list); u_list = array(u_list); w_list = array(w_list)
    zeta_ofBC = zeta / (1-zeta)
    N_list = array([array(arr).shape[0] for arr in u_list])
    theta = 1
    lamb1_list = array([zeros(N) for N in N_list])
    lamb2_list = array([zeros(N) for N in N_list])
    m = w_list.shape[0]
    n = array(costMatrix_list[0]).shape[1]
    tau1_list = zeros((m,n))
    tau2_list = zeros((m,n))
    E = 10000000
    epsilon = max([epsilon,myError])
    print("----------epsilon,eta--in--CraftedFastIBP-- = ",epsilon,eta)
    #---------------------------------------------------------------------------
    cou = 0
    while E > epsilon and cou < MAXcou:
        cou = cou +1
        #Step1
        lamb0_list = (1-theta)*lamb1_list + theta*lamb2_list
        tau0_list = (1-theta)*tau1_list + theta*tau2_list
        #Step2
        R,C,_ = rBcBfun_ofWB(lamb0_list,tau0_list,costMatrix_list,eta)
        R_normal = array([i/sum(i) for i in R])
        lamb22_list = lamb2_list - (R_normal-u_list) / (4*theta)
        C_normal = array([i/sum(i) for i in C ])
        w_list_square = array([w**2 for w in w_list])
        beta = dot(w_list,tau2_list)*4*theta + dot(w_list_square,C_normal)
        beta = beta / sum(w_list_square)
        oneBeta = ones(m)
        tau22_list = array([ -1*(beta*w + c_normal*w)/(4*theta) + t for w,c_normal,t in zip(w_list,C_normal,tau2_list) ])
        #Step3
        lamb3_list = lamb0_list + theta*lamb22_list - theta*lamb2_list
        tau3_list = tau0_list + theta*tau22_list - theta*tau2_list
        #Step4
        phi1 = 0
        for (l,t,w,u,CM) in zip(lamb1_list,tau1_list,w_list,u_list,costMatrix_list):
            B = Bfun_ofOT(l,t,CM,eta)
            B_norm1 = linalg.norm(B,ord=1)
            phi1 = phi1 + w*(log(B_norm1)-dot(l,u)) 

        phi3 = 0
        for (l,t,w,u,CM) in zip(lamb3_list,tau3_list,w_list,u_list,costMatrix_list):
            B = Bfun_ofOT(l,t,CM,eta)
            B_norm1 = linalg.norm(B,ord=1)
            phi3 = phi3 + w*(log(B_norm1)-dot(l,u)) 

        if phi1 < phi3:
            lamb4_list = lamb1_list
            tau4_list = tau1_list
        else:
            lamb4_list = lamb3_list
            tau4_list = tau3_list

        #Step5a 
        _,C,_ = rBcBfun_ofWB(lamb4_list,tau4_list,costMatrix_list,eta)
        #Step5b 
        tmp = dot(w_list,log(C))
        tau5_list = array([t+tmp-log(c) for t,c in zip(tau4_list,C)])

        #-------------------fix weight of baryceneter for outlier-----------------------------
        tau5_list[:,-1] = array([t[-1]+log(zeta_ofBC)-log(c[-1]) for t,c in zip(tau4_list,C)])
        #-------------------fix weight of baryceneter for outlier-----------------------------

        lamb5_list = lamb4_list
        #Step6a
        R,_,_ = rBcBfun_ofWB(lamb5_list,tau5_list,costMatrix_list,eta)
        #Step6b
        lamb_list = array([l+log(u)-log(r) for l,u,r in zip(lamb5_list,u_list,R) ])
        tau_list = tau5_list
        #Step7a
        _,C,_ = rBcBfun_ofWB(lamb_list,tau_list,costMatrix_list,eta)
        #Step7b
        tmp = dot(w_list,log(C))
        tau11_list = array([t+tmp-log(c) for t,c in zip(tau_list,C)])
        #-------------------fix weight of baryceneter for outlier-----------------------------
        tau11_list[:,-1] = array([t[-1]+log(zeta_ofBC)-log(c[-1]) for t,c in zip(tau_list,C)])
        #-------------------fix weight of baryceneter for outlier-----------------------------
        lamb11_list = lamb_list
        #Step8
        theta = theta*(sqrt(theta**2 + 4) - theta) / 2
        lamb1_list = lamb11_list
        tau1_list = tau11_list
        lamb2_list = lamb22_list
        tau2_list = tau22_list
        _,tmp_C,_ = rBcBfun_ofWB(lamb_list,tau_list,costMatrix_list,eta)
        tmp = dot(w_list,tmp_C)
        e_list = array([w*linalg.norm(c-tmp,ord=1) for w,c in zip(w_list,tmp_C) ]) 
        E = sum(e_list)
        res_B_list = array([Bfun_ofOT(l,t,CM,eta) for l,t,CM in zip(lamb_list,tau_list,costMatrix_list)])
        #print("CraftedFastIBP---------------------cou = ",cou)
    return res_B_list







#    #-------------------------------------------------------------------------
#    #costMatrix_list = array([np.random.random(size=(N,n)) for N in N_list]) * 100
#    point_lists = [[[1,1],[2,3],[5,6],[5,5],[4,4]],[[1,5],[6,6],[1,4],[3,5],[4,7]],[[2,3],[3,3],[5,6],[4,2],[3,4]]]
#    barycenter = array([[2,2],[4,5],[6,5]])
#    m = array(point_lists).shape[0] # the number of input distributionds
#    n = barycenter.shape[0] # the support size of barycenter 
#    N_list = array([array(arr).shape[0] for arr in point_lists ])
#    costMatrix_list = array([dist(array(pl),barycenter) for pl in point_lists])
#    print("costMatrix_list = ",costMatrix_list)
#    #print("point_lists = ",point_lists.shape)
#    w_list = array(np.random.random(m))
#    w_list = w_list / sum(w_list)
#    print("w_list = ",w_list)
#    u_list = array([np.random.random(N) for N in N_list])
#    u_list = array([u / sum(u) for u in u_list])
#    print("u_list = ",u_list)
#    epsilon = 0.05
#    eta = 0.1
#    zeta = 0.1
#    res = CraftedFastIBP(costMatrix_list,u_list,w_list,epsilon,eta,zeta)
#    print("res = ",res)
#    print("----------------------------------------------")
#    #--------------------------------------------------------------------------






def MyRobustWBbyCraftedFastIBP_toHa(point_lists:array,barycenter:array,u_list:array,w_list:array,epsilon:int,zeta:int)->any:
    '''
    From:      our robust WB 
    Function:  WB by CraftedFastIBP
    parameter: point_lists: the location lists of input distributions;    ndarray((n_1,d),(n_2,d),...,(n_m,d))
               barycenter: the location list of barycenter;               ndarray(n,d)  
               u_list: input simplex list of input distributions ;      ndarray(n_1,n_2,...,m_m)  
               w_list: the weights for input probability distributions;                ndarray(m)
               epsilon: the additive error we can tolerant
               zeta: the total mass of outliers for given distribution
    Note:      there are "m" input  distributions; The support size of each distributions can be different; "n" denotes the support size of barycenter;          
    Return:    robust_res_u: probability simplex  of barycenter of RWB (without dummy point)
               extended_robust_res_u: probability simplex  of barycenter of RWB (with dummy point)
               robust_X_list: the transportation plan list (in feasible domain) 
    '''
    time0 = time.process_time()
    point_lists = array(point_lists); barycenter = array(barycenter); u_list = array(u_list); w_list = array(w_list)
    costMatrix_list = array([dist(array(pl),barycenter) for pl in point_lists])
    robust_costMatrix_list = []
    for costMatrix in costMatrix_list:
        cM_shape = costMatrix.shape
        row_Addzeros = zeros(cM_shape[1])
        #column_Addzeros = zeros(cM_shape[0] + 1)
        column_Addzeros = zeros(cM_shape[0])
        #costMatrix = np.row_stack((costMatrix,row_Addzeros))
        costMatrix = np.column_stack((costMatrix,column_Addzeros))
        robust_costMatrix_list.append(costMatrix)
    robust_costMatrix_list = array(robust_costMatrix_list)    
    N_list = array([array(pl).shape[0] for pl in point_lists])
    my_n = int(mean(N_list))
    eta = epsilon / (4*log(my_n)) # n denotes the support size of barycenter #or the support size of input distribution
    #print("eta-++++++++++++++++++++++++++++++++++++++++++++++++++++-in--WBbyFastIBP = ",eta)
    epsilon1 = epsilon / (4*max([linalg.norm(CM,ord=np.Inf) for CM in costMatrix_list]))
    #-------------------------------------------------------
    #Step1
    u1_list = array([ (1-epsilon1/4)*array(u) for u in u_list ])
    u1_list = array([u1 + epsilon1 / (4*array(u1).shape[0]) for u1 in u1_list])
    robust_u1_list = u1_list / (1-zeta)
    #Step2
    robust_X1_list = CraftedFastIBP(robust_costMatrix_list,robust_u1_list,w_list,epsilon1/2,eta,zeta)
    #Step3
    robust_X_list = RoundToFeasibleSolution_ofWB(robust_X1_list,robust_u1_list,w_list) #u1_list or u_list ????
    #Step4
    robust_res_u_list = array([w*sum(X,axis=0) for w,X in zip(w_list,robust_X_list)])
    extended_robust_res_u = sum(robust_res_u_list,axis=0)
    robust_res_u = extended_robust_res_u[:-1]
    robust_res_u = robust_res_u / sum(robust_res_u)
    #----------compute cost of RWB----------
    cost_list = array([ w*sum(CM*X) for w,CM,X in zip(w_list,robust_costMatrix_list,robust_X_list) ])  
    cost = sum(cost_list)
    time1 = time.process_time()
    runtime = time1 - time0
    return robust_res_u,extended_robust_res_u,robust_X_list,cost,runtime





#    #--------------------------------------------------------------------------
#    # point_lists
#    point_lists = array([[[1,1],[2,3],[5,6],[5,5],[4,4]],[[1,5],[6,6],[1,4],[3,5],[4,7]],[[2,3],[3,3],[5,6],[4,2],[8,8]]])
#    # barycenter
#    barycenter = array([[2,2],[4,5],[6,5],[1,4]])
#    # u_list
#    #print("point_lists = ",point_lists)
#    N_list = array([array(pl).shape[0] for pl in point_lists])
#    print("N_list = ",N_list)
#    u_list = array([np.random.random(N) for N in N_list])
#    u_list = array([u / sum(u) for u in u_list])
#    print("u_list = ----------",u_list.shape)
#    # w_list
#    w_list = array(np.random.random(point_lists.shape[0]))
#    w_list = w_list / sum(w_list)
#    print("w_list = ",w_list)
#    # epsilon
#    epsilon = 200
#    #test function
#    zeta = 0.1
#    robust_res_u,extended_robust_res_u,robust_X_list,cost,runtime = MyRobustWBbyCraftedFastIBP_toHa(point_lists,barycenter,u_list,w_list,epsilon,zeta)
#    print("extended_robust_res_u = ",extended_robust_res_u)
#    print("-------------------------------------------------------------------")
#    #   #--------------------------------------------------------------------------







def CraftedDualIBP(costMatrix_list:array,u_list:array,w_list:array,epsilon:int,eta:int,zeta:int)->array:    
    '''
    From:     Algorithm 1 of "On the Complexity of Approximating Wasserstein Barycenters" 
              designed for our "crafted RWB"; that is, we fixed the weight of outlier of barycenter.  e.g., b[-1] = zeta_ofBC = zeta / (1-zeta)
    Function: CraftedDualIBP
    parameter:costMatrix_list: cost matrix list between  "input distributions and barycenter" (行数代表输入的support的size，列数代表barycenter的support的size)
              u_list: input simplex list of input distributions ;      ndarray(n_1,n_2,...,m_m)  
              w_list: the weights for input probability distributions;                ndarray(m)
              epsilon: the additive error we can tolerant
              eta: on entropic regularization term for Sinkhorn algorithm
              zeta: the total mass of outliers for given distribution
    Return:   res_B_list:  the transportation plan list (maybe not in feasible domain)                                         
    '''
    costMatrix_list = array(costMatrix_list); u_list = array(u_list); w_list = array(w_list)
    zeta_ofBC = zeta / (1-zeta)
    N_list = array([array(arr).shape[0] for arr in u_list])
    lamb_list = array([zeros(N) for N in N_list])
    m = w_list.shape[0]
    n = array(costMatrix_list[0]).shape[1]
    tau_list = zeros((m,n))
    K_list = array([exp((-1)*Cl/eta) for Cl in costMatrix_list])
    cond1 = 10000000
    cond2 = 10000000
    print("----------epsilon,eta--in--CraftedDualIBP-- = ",epsilon,eta)
    #---------------------------------------------------------------------------
    while cond1 > epsilon or cond2 > epsilon:
        for l in range(m):
            tmp_Kev = dot( K_list[l], exp(tau_list[l]))
            tmp_Kev = tmp_Kev.reshape((tmp_Kev.shape[0],1))
            tmp_lamb = log(u_list[l]) - log(tmp_Kev.T) # may have error log(K_list[l]*exp(tau_list[l]))  
            lamb_list[l] =  tmp_lamb[0]
 
        Log_KTeu_list = array([log( dot(K.T,array([list(exp(lamb))]).T) ).T[0] for K,lamb in zip(K_list,lamb_list)])
        sum_weighted_Log_KTeu_list = sum(dot(Log_KTeu_list.T,w_list))# may error
        for l in range(m):
            tau_list[l] = sum_weighted_Log_KTeu_list - Log_KTeu_list[l] 

        #-------------------fix weight of baryceneter for outlier-----------------------------
        tau_list[:,-1] = array([log(zeta_ofBC)-c[-1] for c in Log_KTeu_list])
        #-------------------fix weight of baryceneter for outlier-----------------------------

        R,C,_ = rBcBfun_ofWB(lamb_list,tau_list,costMatrix_list,eta)
        qBar = dot(C.T,w_list) 
        cond1_list = array([linalg.norm(c-qBar,ord=1) for c in C])
        cond1 = dot(cond1_list,w_list)
        cond2_list = array([linalg.norm(r-u,ord=1) for r,u in zip(R,u_list)])
        cond2 = dot(cond2_list,w_list)
    res_B_list = array([Bfun_ofOT(l,t,CM,eta) for l,t,CM in zip(lamb_list,tau_list,costMatrix_list)])
    return res_B_list




#   #nan出现的时候，epsilon，n，调节
#   #-------------------------------------------------------------------------
#   point_lists = [[[1,1],[2,3],[5,6],[5,5],[4,4]],[[1,5],[6,6],[1,4],[3,5],[4,7]],[[2,3],[3,3],[5,6],[4,2],[3,4]]]
#   barycenter = array([[2,2],[4,5],[6,5]])
#   m = array(point_lists).shape[0] # the number of input distributionds
#   n = barycenter.shape[0] # the support size of barycenter 
#   N_list = array([array(arr).shape[0] for arr in point_lists ])
#   costMatrix_list = array([dist(array(pl),barycenter) for pl in point_lists])
#   print("costMatrix_list = ",costMatrix_list)
#   #print("point_lists = ",point_lists.shape)
#   w_list = array(np.random.random(m))
#   w_list = w_list / sum(w_list)
#   print("w_list = ",w_list)
#   u_list = array([np.random.random(N) for N in N_list])
#   u_list = array([u / sum(u) for u in u_list])
#   print("u_list = ",u_list)
#   epsilon = 100
#   eta = 0.1
#   zeta = 0.1
#   res = CraftedDualIBP(costMatrix_list,u_list,w_list,epsilon,eta,zeta)    
#   print("res = ",res)
#   print("----------------------------------------------")
#   #----------------------------------------------------------------------



def MyRobustWBbyCraftedDualIBP_toHa(point_lists:array,barycenter:array,u_list:array,w_list:array,epsilon:int,zeta:int)->any:
    '''
    From:      our robust WB 
    Function:  my robust WB by Crafted DualIBP 
    parameter: point_lists: the location lists of input distributions;    ndarray((n_1,d),(n_2,d),...,(n_m,d))
               barycenter: the location list of barycenter;               ndarray(n,d)  
               u_list: input simplex list of input distributions ;      ndarray(n_1,n_2,...,m_m)  
               w_list: the weights for input probability distributions;                ndarray(m)
               epsilon: the additive error we can tolerant
               zeta: the total mass of outliers for given distribution and barycenter
    Note:      there are "m" input  distributions; The support size of each distributions can be different; "n" denotes the support size of barycenter;          
    Return:    robust_res_u: probability simplex  of barycenter of RWB (without dummy point)
               extended_robust_res_u: probability simplex  of barycenter of RWB (with dummy point)
               robust_X_list: the transportation plan list (in feasible domain) 
    '''
    time0= time.process_time()
    point_lists = array(point_lists); barycenter = array(barycenter); u_list = array(u_list); w_list = array(w_list)
    costMatrix_list = array([dist(array(pl),barycenter) for pl in point_lists])
    robust_costMatrix_list = []
    for costMatrix in costMatrix_list:
        cM_shape = costMatrix.shape
        row_Addzeros = zeros(cM_shape[1])
        #column_Addzeros = zeros(cM_shape[0] + 1)
        column_Addzeros = zeros(cM_shape[0])
        #costMatrix = np.row_stack((costMatrix,row_Addzeros))
        costMatrix = np.column_stack((costMatrix,column_Addzeros))
        robust_costMatrix_list.append(costMatrix)
    robust_costMatrix_list = array(robust_costMatrix_list)    
    N_list = array([array(pl).shape[0] for pl in point_lists])
    my_n = int(mean(N_list))
    eta = epsilon / (4*log(my_n)) # n denotes the support size of barycenter #or the support size of input distribution
    epsilon1 = epsilon / (4*max([linalg.norm(CM,ord=np.Inf) for CM in costMatrix_list]))
    #-------------------------------------------------------
    #Step1
    u1_list = array([ (1-epsilon1/4)*array(u) for u in u_list ])
    u1_list = array([u1 + epsilon1 / (4*array(u1).shape[0]) for u1 in u1_list])
    robust_u1_list = u1_list / (1-zeta)
    #Step2
    robust_X1_list = CraftedDualIBP(robust_costMatrix_list,robust_u1_list,w_list,epsilon1/2,eta,zeta)
    #Step3
    robust_X_list = RoundToFeasibleSolution_ofWB(robust_X1_list,robust_u1_list,w_list) #u1_list or u_list ????
    #Step4
    robust_res_u_list = array([w*sum(X,axis=0) for w,X in zip(w_list,robust_X_list)])
    extended_robust_res_u = sum(robust_res_u_list,axis=0)
    robust_res_u = extended_robust_res_u[:-1]
    robust_res_u = robust_res_u / sum(robust_res_u)
    #----------compute cost of RWB----------
    cost_list = array([ w*sum(CM*X) for w,CM,X in zip(w_list,robust_costMatrix_list,robust_X_list) ])  
    cost = sum(cost_list)
    time1 = time.process_time()
    runtime = time1 - time0
    return robust_res_u,extended_robust_res_u,robust_X_list,cost,runtime





#    #--------------------------------------------------------------------------
#    # point_lists
#    point_lists = array([[[1,1],[2,3],[5,6],[5,5],[4,4]],[[1,5],[6,6],[1,4],[3,5],[4,7]],[[2,3],[3,3],[5,6],[4,2],[8,8]]])
#    # barycenter
#    barycenter = array([[2,2],[4,5],[6,5],[1,4]])
#    # u_list
#    #   print("point_lists = ",point_lists)
#    #   
#    m = array(point_lists).shape[0]
#    N_list= array([array(pl).shape[0] for pl in point_lists]) # the support size list of input distributions
#    n = barycenter.shape[0] # the support size of barycenter 
#    costMatrix_list = array([np.random.random(size=(N,n)) for N in N_list]) * 100
#    point_lists = array([np.random.random(size=(N,2)) for N in N_list]) * 100
#    
#    N_list = array([array(pl).shape[0] for pl in point_lists])
#    print("N_list = ",N_list)
#    u_list = array([np.random.random(N) for N in N_list])
#    u_list = array([u / sum(u) for u in u_list])
#    print("u_list = -----------------------",u_list)
#    # w_list
#    w_list = array(np.random.random(point_lists.shape[0]))
#    w_list = w_list / sum(w_list)
#    print("w_list = ",w_list)
#    # epsilon
#    epsilon = 1000000
#    #test function
#    zeta = 0.1
#    robust_res_u,extended_robust_res_u,robust_X_list,cost,runtime = MyRobustWBbyCraftedDualIBP_toHa(point_lists,barycenter,u_list,w_list,epsilon,zeta)
#    print("robust_res_u = ",robust_res_u)
#    print("-------------------------------------------------------------------")
#    #--------------------------------------------------------------------------









def RobustUOTbasedIBP(costMatrix_list:array,u_list:array,w_list:array,gamma:int,eta:int,UOTRWBIter:int)->array:    
    '''
    From:     Algorithm 1 of "On Robust Optimal Transport: Computational Complexity and Barycenter Computation" 
    Function: RobustUOTbasedIBP
              parameter:
              costMatrix_list: cost matrix list between  "input distributions and barycenter" (行数代表输入的support的size，列数代表barycenter的support的size)
              u_list: input simplex list of input distributions ;      ndarray(n_1,n_2,...,m_m)  
              w_list: the weights for input probability distributions;                ndarray(m)
              gamma: the \tau in paper to control the marginal violation
              eta: on entropic regularization term for Sinkhorn algorithm
              UOTRWBIter: prespecified iterative times
    Return:   res_B_list:  the transportation plan list (maybe not in feasible domain)                                         
    '''
    costMatrix_list = array(costMatrix_list); u_list = array(u_list); w_list = array(w_list)
    N_list = array([array(arr).shape[0] for arr in u_list])
    lamb_list = array([zeros(N) for N in N_list])
    m = w_list.shape[0]
    n = array(costMatrix_list[0]).shape[1]
    tau_list = zeros((m,n))
    t = 0
    #---------------------------------------------------------------------------
    while t < UOTRWBIter:
        A,B,_ = rBcBfun_ofWB(lamb_list,tau_list,costMatrix_list,eta)
        lamb_list = gamma*eta/(gamma+eta) * (lamb_list/eta + wx2Dlog(u_list) - wx2Dlog(A) )
        tmp_avg = dot(w_list, tau_list/eta - log(B))
        tau_list = eta * ( tau_list/eta - log(B) - tmp_avg )
        t = t + 1
    res_B_list = array([Bfun_ofOT(l,t,CM,eta) for l,t,CM in zip(lamb_list,tau_list,costMatrix_list)])
    res_B_list = array( [B/sum(B) for B in res_B_list] )
    return res_B_list




#   #nan出现的时候，epsilon，n，调节
#   #-------------------------------------------------------------------------
#   point_lists = array([[[1,1],[2,3],[5,6],[5,5],[4,4]],[[1,5],[6,6],[1,4],[3,5],[4,7]]])
#   barycenter = array([[2,2],[4,5],[6,5]])
#   costMatrix_list = array([dist(array(pl),barycenter) for pl in point_lists])
#   m = array(point_lists).shape[0] # the number of input distributionds
#   N_list= array([array(pl).shape[0] for pl in point_lists]) # the support size list of input distributions
#   n = barycenter.shape[0] # the support size of barycenter 
#   print("costMatrix_list = ",costMatrix_list)
#   print("point_lists = ",point_lists.shape)
#   w_list = array(np.random.random(m))
#   w_list = w_list / sum(w_list)
#   print("w_list = ",w_list)
#   u_list = array([np.random.random(N) for N in N_list])
#   u_list = array([u / sum(u) for u in u_list])
#   print("u_list = ",u_list)
#   gamma = 1000
#   eta = 0.1
#   UOTRWBIter = 5
#   res = RobustUOTbasedIBP(costMatrix_list,u_list,w_list,gamma,eta,UOTRWBIter)
#   print("res = ",res)
#   print("----------------------------------------------")
#   #--------------------------------------------------------------------------




def WBbyRobustUOTbasedIBP_toHa(point_lists:array,barycenter:array,u_list:array,w_list:array,gamma:int,eta:int,UOTRWBIter:int)->any:
    '''
    From:      On Robust Optimal Transport: Computational Complexity and Barycenter Computation
    Function:  WB by CraftedDualIBP
    parameter: point_lists: the location lists of input distributions;    ndarray((n_1,d),(n_2,d),...,(n_m,d))
               barycenter: the location list of barycenter;               ndarray(n,d)  
               u_list: input simplex list of input distributions ;      ndarray(n_1,n_2,...,m_m)  
               w_list: the weights for input probability distributions;                ndarray(m)
               gamma: the \tau in paper to control the marginal violation
               eta: on entropic regularization term for Sinkhorn algorithm
               UOTRWBIter: prespecified iterative times
    Note:      there are "m" input  distributions; The support size of each distributions can be different; "n" denotes the support size of barycenter;          
    Return:    res_u: probability simplex  of barycenter of WB        
               X_list: the transportation plan list (maybe not in feasible domain)                 
    '''
    time0 = time.process_time()
    point_lists = array(point_lists); barycenter = array(barycenter); u_list = array(u_list); w_list = array(w_list)
    costMatrix_list = array([dist(array(pl),barycenter) for pl in point_lists])
    #-------------------------------------------------------
    #Step1
    X_list = RobustUOTbasedIBP(costMatrix_list,u_list,w_list,gamma,eta,UOTRWBIter)
    #Step4
    res_u_list = array([w*sum(X,axis=0) for w,X in zip(w_list,X_list)])
    res_u = sum(res_u_list,axis=0)
    res_u = res_u / sum(res_u)
    #----------compute cost of RWB----------
    cost_list = array([ w*sum(CM*X) for w,CM,X in zip(w_list,costMatrix_list,X_list) ])  
    cost = sum(cost_list)
    time1 = time.process_time()
    runtime = time1 - time0
    return res_u,X_list,cost,runtime









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
#   gamma = 1000
#   eta = 0.1
#   UOTRWBIter = 5
#   #test function
#   res_u,X_list,cost,runtime = WBbyRobustUOTbasedIBP_toHa(point_lists,barycenter,u_list,w_list,gamma,eta,UOTRWBIter)
#   print("res_u,X_list,cost,runtime = ",res_u,X_list,cost,runtime)
#   #--------------------------------------------------------------------------
#   
