# -*- coding: utf-8 -*-
import numpy as np 
from numpy import random ,ones,outer,array,exp,copy,sum,zeros,dot,linalg,log,sqrt,abs,diag,mean,ceil,round,floor
from ot import dist
import random
myError = 0.1



def ROUND_OT(F:array,Ur:array,Uc:array)->array:
    '''
    From:      Algorithm 2 of "Near-linear time approximation algorithms for optimal transport via Sinkhorn iteration" 
    Function:  projection to feasible coupling for OT; we make sure that  "Ur,Uc" are the margin of feasible coupling "G", where "G" is the projection of "F"
    parameter: F    : denotes the "coupling" (maybe not in feasible domain)
               Ur,Uc: denote  the marginal probability simplex 
    Return:    G    : the projection of "F" in feasible domain; it is a "feasible coupling" for OT            
    '''
    F = array(F); Ur = array(Ur); Uc = array(Uc)
    Fr = sum(F,axis=1)
    x = Ur/Fr
    x = array([min(1,i) for i in x])
    X = diag(x)
 
    F1 = dot(X,F)
    F1c = sum(F1,axis=0)
    y = Uc/F1c
    y = array([min(1,i) for i in y])
    Y = diag(y)
    F11 = dot(F1,Y)
    ERRr = Ur - sum(F11,axis=1)
    ERRc = Uc - sum(F11,axis=0)
    G = F11 + outer(ERRr,ERRc) / sum(abs(ERRr))
    return G





#-----------test----------------------------------------------------------------
#F = np.random.random(size=(5,3))
#Ur = np.random.random(size=(5))
#Uc = np.random.random(size=(3))
#print(ROUND_OT(F,Ur,Uc))
#-----------test----------------------------------------------------------------







def RoundToFeasibleSolution_ofWB(B_list:array,p_list:array,w_list:array)->array:
    '''
    From:      Algorithm 4 of "On the Complexity of Approximating Wasserstein Barycenter" ;
               /https://www.researchgate.net/profile/Alexey-Kroshnin/publication/330673052_On_the_Complexity_of_Approximating_Wasserstein_Barycenter/links/5c51d4a8a6fdccd6b5d4ed31/On-the-Complexity-of-Approximating-Wasserstein-Barycenter.pdf#cite.altschuler2017near-linear
    Function:  projection to feasible coupling list for WB, we make sure that  "p_list" are the margin of the projection of "B_list"
    parameter: B_list                : denotes the "coupling list" (maybe not in feasible domain)
               p_list                : denote  the "marginal probability simplex list"
               w_list                : the weights for input probability distributions
    Return:    feasible_coupling_list: "feasible coupling list" for WB        
    '''
    B_list = array(B_list); p_list = array(p_list); w_list = array(w_list)
    feasible_coupling_list = []
    Bc_arr = array([w*sum(B,axis=0) for w,B in zip(w_list,B_list)])
    Bc_arr_sum = sum(Bc_arr,axis=0)

    scale_arr = array([w*sum(B) for w,B in zip(w_list,B_list)] ) 
    scale_arr_sum = sum(scale_arr)

    q = Bc_arr_sum / scale_arr_sum

    for B,p in zip(B_list,p_list):
        res = ROUND_OT(B,p,q)
        feasible_coupling_list.append(res)

    return array(feasible_coupling_list)




#-----------test----------------------------------------------------------------
#B_list = np.random.random(size=(2,5,3))
#p_list = np.random.random(size=(2,5))
#w_list = np.random.random(size=(2))
#print(RoundToFeasibleSolution_ofWB(B_list,p_list,w_list))
#-----------test----------------------------------------------------------------









def Bfun_ofOT(lamb:array,tau:array,costMatrix:array,eta:int)->array:
    '''
    From: "B"  function for Algorithm 1 in "Fixed-Support Wasserstein Barycenters:costMatrixomputational Hardness and Fast Algorithm" ;
    Function:  compute primal variable (coupling) from dual variable "lamb,tau" for OT
    parameter: lamb      : dual variable
               tau       : dual variable
               costMatrix: cost matrix
               eta       : on entropic regularization term for Sinkhorn algorithm
    Return:    res       : primal variable (coupling)   
    '''
    #print("---------------Bfun_ofOT()---------------")
    lamb = array(lamb); tau = array(tau); costMatrix = array(costMatrix)
    costMatrix_shape = costMatrix.shape
    oneLamb = ones(costMatrix_shape[1])
    lambMatrix = outer(lamb,oneLamb)
    oneTau = ones(costMatrix_shape[0])
    tauMatrix = outer(oneTau,tau)
    M = lambMatrix + tauMatrix - costMatrix / eta
    res = exp(M)
    return res






#-----------test----------------------------------------------------------------
#eta = 0.1
#lamb = np.random.random(size=(5))
#tau = np.random.random(size=(3))
#costMatrix = np.random.random(size=(5,3))
#Bfun_ofOT(lamb,tau,costMatrix,eta)
#-----------test----------------------------------------------------------------











def rBcBfun_ofWB(lamb_list:array,tau_list:array,costMatrix_list:array,eta:int)->array:
    '''
    From:     "Fixed-Support Wasserstein Barycenters: Computational Hardness and Fast Algorithm" 
    Function: compute : "r_k = c(B_k(lamb,tau))" and "c_k = c(B_k(lamb,tau))" for all k in [m]
    parameter:lamb_list      : dual variable list for "input distributions"
              tau_list       : dual variable list  for barycenter
              costMatrix_list: cost matrix list between  "input distributions and barycenter"
              eta            : on entropic regularization term for Sinkhorn algorithm
    Return:   R              : "r_k = c(B_k(lamb,tau))" for all k in [m];      the column sum of every "B" in "B_list"
              C              : "c_k = c(B_k(lamb,tau))" for all k in [m];      the row sum of every "B" in "B_list"
              B_list         : primal variable (coupling) list
    '''
    lamb_list = array(lamb_list); tau_list = array(tau_list); costMatrix_list = array(costMatrix_list)
    B_list = array([Bfun_ofOT(l,t,c,eta) for l,t,c in zip(lamb_list,tau_list,costMatrix_list)])
    R = array([sum(i,axis=1) for i in B_list])
    C = array([sum(i,axis=0) for i in B_list])
    return R,C,B_list 








def FastIBP(costMatrix_list:array,u_list:array,w_list:array,epsilon:int,eta:int)->array:    
    '''
    From:     Algorithm 1 of "Fixed-Support Wasserstein Barycenters: Computational Hardness and Fast Algorithm" 
    Function: FastIBP
    parameter:costMatrix_list: cost matrix list between  "input distributions and barycenter" (行数代表输入的support的size，列数代表barycenter的support的size)
              u_list         : input simplex list of input distributions
              w_list         : the weights for input probability distributions
              epsilon        : the additive error we can tolerant
              eta            : on entropic regularization term for Sinkhorn algorithm
    Return:   res_B_list     : the coupling list (maybe not in feasible domain)                                         
    '''
    costMatrix_list = array(costMatrix_list); u_list = array(u_list); w_list = array(w_list)
    res_B_list = array([0])
    N_list = array([array(arr).shape[0] for arr in u_list])
    theta = 1
    lamb1_list = array([zeros(N) for N in N_list])
    lamb2_list = array([zeros(N) for N in N_list])
    m = w_list.shape[0]
    n = array(costMatrix_list[0]).shape[1]
    tau1_list = array(zeros((m,n)))
    tau2_list = array(zeros((m,n)))
    E = 10000000000
    print("epsilon,eta--in--FastIBP-- = ",epsilon,eta)
    epsilon = max([epsilon,myError])
    #---------------------------------------------------------------------------
    while E > epsilon:
        #Step1
        lamb0_list = (1-theta)*lamb1_list + theta*lamb2_list
        tau0_list = (1-theta)*tau1_list + theta*tau2_list
        #Step2
        R,C,B_list = rBcBfun_ofWB(lamb0_list,tau0_list,costMatrix_list,eta)


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
        _,C ,B_list= rBcBfun_ofWB(lamb4_list,tau4_list,costMatrix_list,eta)


        #Step5b 
        tmp = dot(w_list,log(C))
        tau5_list = array([t+tmp-log(c) for t,c in zip(tau4_list,C)])
        lamb5_list = lamb4_list
        #Step6a
        R,_,B_list = rBcBfun_ofWB(lamb5_list,tau5_list,costMatrix_list,eta)

 
        #Step6b
        lamb_list = array([l+log(u)-log(r) for l,u,r in zip(lamb5_list,u_list,R) ])
        tau_list = tau5_list

        #Step7a
        _,C,B_list = rBcBfun_ofWB(lamb_list,tau_list,costMatrix_list,eta)

  
        #Step7b
        tmp = dot(w_list,log(C))
        tau11_list = array([t+tmp-log(c) for t,c in zip(tau_list,C)])
        lamb11_list = lamb_list
        #Step8
        theta = theta*(sqrt(theta**2 + 4) - theta) / 2


        lamb1_list = lamb11_list
        tau1_list = tau11_list
        lamb2_list = lamb22_list
        tau2_list = tau22_list
        _,tmp_C,B_list = rBcBfun_ofWB(lamb_list,tau_list,costMatrix_list,eta)


        tmp = dot(w_list,tmp_C)
        e_list = array([w*linalg.norm(c-tmp,ord=1) for w,c in zip(w_list,tmp_C) ]) 
        E = sum(e_list)
        res_B_list = array([Bfun_ofOT(l,t,CM,eta) for l,t,CM in zip(lamb_list,tau_list,costMatrix_list)])


        # delete the following---------------
        tmp_cost_list = array([w*sum(B*cM) for w,B,cM in zip(w_list,res_B_list,costMatrix_list) ])
        tmp_cost = sum(tmp_cost_list)
        #print("tmp_cost 666=========================== ",tmp_cost)
        # delete ----------------------------
    #print("res_B_list = ",res_B_list)
    return res_B_list






   
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
#   epsilon = 0.05
#   eta = 0.1
#   res = FastIBP(costMatrix_list,u_list,w_list,epsilon,eta)
#   print("res = ",res)
#   print("----------------------------------------------")
#   #--------------------------------------------------------------------------










def WBbyFastIBP(point_lists:array,barycenter:array,u_list:array,w_list:array,epsilon:int)->array:
    '''
    From:      Algorithm 2 of"Fixed-Support Wasserstein Barycenters: Computational Hardness and Fast Algorithm" 
    Function:  WB by FastIBP
    parameter: point_lists: the location lists of input distributions;    ndarray((n_1,d),(n_2,d),...,(n_m,d))
               barycenter : the location list of barycenter;               ndarray(n,d)  
               u_list     : input simplex list of input distributions ;      ndarray(n_1,n_2,...,m_m)  
               w_list     : the weights for input probability distributions                ndarray(m)
               epsilon    : the additive error we can tolerant
    Note:      there are "m" input  distributions; The support size of each distributions can be different; "n" denotes the support size of barycenter;          
    Return:    res_u      : simplex of barycenter of WB   
               X_list     : the coupling list (in feasible domain)                                  
    '''
    point_lists = array(point_lists); barycenter = array(barycenter); u_list = array(u_list); w_list = array(w_list)
    costMatrix_list = array([dist(array(pl),barycenter) for pl in point_lists])
    N_list = array([array(pl).shape[0] for pl in point_lists])
    my_n = int(mean(N_list))
    eta = epsilon / (4*log(my_n)) # n denotes the support size of barycenter #or the support size of input distribution
    epsilon1 = epsilon / (4*max([linalg.norm(CM,ord=np.Inf) for CM in costMatrix_list]))
    #-------------------------------------------------------
    #Step1
    u1_list = array([ (1-epsilon1/4)*array(u) for u in u_list ])
    u1_list = array([u1 + epsilon1 / (4*array(u1).shape[0]) for u1 in u1_list])
    #Step2
    X1_list = FastIBP(costMatrix_list,u1_list,w_list,epsilon1/2,eta)
    #Step3
    X_list = RoundToFeasibleSolution_ofWB(X1_list,u1_list,w_list) #u1_list or u_list ????
    #Step4
    res_u_list = array([w*sum(X,axis=0) for w,X in zip(w_list,X_list)])
    res_u = sum(res_u_list,axis=0)
    return res_u,X_list






#   #--------------------------------------------------------------------------
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
#   # epsilon
#   epsilon = 2
#   #test function
#   res = WBbyFastIBP(point_lists,barycenter,u_list,w_list,epsilon)
#   print("res = ",res)
#   #--------------------------------------------------------------------------









def DualIBP(costMatrix_list:array,u_list:array,w_list:array,epsilon:int,eta:int)->array:    
    '''
    From:     Algorithm 1 of "On the Complexity of Approximating Wasserstein Barycenters" 
    Function: DualIBP
    parameter:costMatrix_list: cost matrix list between  "input distributions and barycenter" (行数代表输入的support的size，列数代表barycenter的support的size)
              u_list         : input simplex list of input distributions
              w_list         : the weights for input probability distributions
              epsilon        : the additive error we can tolerant
              eta            : on entropic regularization term for Sinkhorn algorithm
    Return:   res_B_list     :  the coupling list (maybe not in feasible domain)                                         
    '''
    costMatrix_list = array(costMatrix_list); u_list = array(u_list); w_list = array(w_list)
    N_list = array([array(arr).shape[0] for arr in u_list])
    lamb_list = array([zeros(N) for N in N_list])
    m = w_list.shape[0]
    n = array(costMatrix_list[0]).shape[1]
    tau_list = zeros((m,n))
    K_list = array([exp((-1)*Cl/eta) for Cl in costMatrix_list])
    cond1 = 10000000
    cond2 = 10000000
    print("epsilon,eta--in--DualIBP-- = ",epsilon,eta)
    #---------------------------------------------------------------------------
    while cond1 > epsilon or cond2 >epsilon:
        for l in range(m):
            tmp_Kev = dot( K_list[l], array([list(exp(tau_list[l]))]).T )
            tmp_lamb = log(u_list[l]) - log(tmp_Kev.T) # may have error log(K_list[l]*exp(tau_list[l]))  
            lamb_list[l] =  tmp_lamb[0]
 
        Log_KTeu_list = array([log( dot(K.T,array([list(exp(lamb))]).T) ).T[0] for K,lamb in zip(K_list,lamb_list)])
        sum_weighted_Log_KTeu_list = sum(dot(Log_KTeu_list.T,w_list))# may error
        for l in range(m):
            tau_list[l] = sum_weighted_Log_KTeu_list - Log_KTeu_list[l] 
#       
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
#   epsilon = 1
#   eta = 0.1
#   res = DualIBP(costMatrix_list,u_list,w_list,epsilon,eta)
#   print("res = ",res)
#   print("----------------------------------------------")
#   #--------------------------------------------------------------------------







def WBbyDualIBP(point_lists:array,barycenter:array,u_list:array,w_list:array,epsilon:int)->array:
    '''
    From:      Algorithm 2 of"On the Complexity of Approximating Wasserstein Barycenters" 
    Function:  WB by DualIBP
    parameter: point_lists: the location lists of input distributions;    ndarray((n_1,d),(n_2,d),...,(n_m,d))
               barycenter: the location list of barycenter;               ndarray(n,d)  
               u_list: input simplex list of input distributions;      ndarray(n_1,n_2,...,m_m)  
               w_list: the weights for input probability distributions              ndarray(m)
               epsilon: the additive error we can tolerant
    Note:      there are "m" input  distributions; The support size of each distributions can be different; "n" denotes the support size of barycenter;          
    Return:    res_u: simplex distributions function of barycenter of WB    
               X_list: the coupling list (in feasible domain)                                
    '''
    point_lists = array(point_lists); barycenter = array(barycenter); u_list = array(u_list); w_list = array(w_list)
    costMatrix_list = array([dist(array(pl),barycenter) for pl in point_lists])
    N_list = array([array(pl).shape[0] for pl in point_lists])
    my_n = int(mean(N_list))
    eta = epsilon / (4*log(my_n)) # n denotes the support size of barycenter #or the support size of input distribution
    epsilon1 = epsilon / (4*max([linalg.norm(CM,ord=np.Inf) for CM in costMatrix_list]))
    #-------------------------------------------------------
    #Step1
    u1_list = array([ (1-epsilon1/4)*array(u) for u in u_list ])
    u1_list = array([u1 + epsilon1 / (4*array(u1).shape[0]) for u1 in u1_list])
    #Step2
    X1_list = DualIBP(costMatrix_list,u1_list,w_list,epsilon1/2,eta)
    #Step3
    X_list = RoundToFeasibleSolution_ofWB(X1_list,u1_list,w_list) #u1_list or u_list ????
    #Step4
    res_u_list = array([w*sum(X,axis=0) for w,X in zip(w_list,X_list)])
    res_u = sum(res_u_list,axis=0)
    return res_u,X_list






#    #--------------------------------------------------------------------------
#    point_lists = array([[[1,1],[2,3],[5,6],[5,5],[4,4]],[[1,5],[6,6],[1,4],[3,5],[4,7]],[[2,3],[3,3],[5,6],[4,2],[3,4]]] )*10
#    barycenter = array([[1,1],[2,3],[5,6],[5,5],[4,4]])*10
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
#    # epsilon
#    epsilon = 20
#    #test function
#    res = WBbyDualIBP(point_lists,barycenter,u_list,w_list,epsilon)
#    print("res = ",res)
#    #--------------------------------------------------------------------------




