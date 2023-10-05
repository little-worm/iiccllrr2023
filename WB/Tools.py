import os
from numpy import array
def my_mkdir(folder_PathName):
    if not os.path.exists(folder_PathName):
        os.makedirs(folder_PathName)
        #print("folder_PathName = ",os.path.abspath(folder_PathName)) 


#   folder_PathName = '/home/worm/wormCode_2023_0201/_Neurips_2023/DataTest/tmp'
#   my_mkdir(folder_PathName)





def myListsSlice(arr_list:list,index_list:array)->list:
    '''
    From       : myself 
    Function   : return the elements of several arrays according to a index_list 
    arr_list   : the list of arrays; the arrays have the same 0-th dim 
    index_list : the index we specified
    '''
    index_list = array(index_list)
    tmp_arr_list = [[] for i in range(len(arr_list))]
    for index in index_list:
        for arr_num in range(len(arr_list)):
            tmp_arr_list[arr_num].append( arr_list[arr_num][index] )
    #res = [array(arr) for arr in tmp_arr_list]   
    #print("tmp_arr_list = ",tmp_arr_list)     
    return tmp_arr_list        


#   ## -----------------test for ----myListsSlice()------------------------------
#   print("-----------------test for ----myListsSlice()------------------------------")
#   arr1 = array( [i for i in range(10)] )
#   arr2 = arr1**2
#   arr3 = arr1**3
#   print(arr1,arr2)
#   arr_list = [arr1,arr2,arr3]
#   index_list = [2,4,6,8]
#   res = myListsSlice(arr_list,index_list)
#   print("res = ",res)





 
def search_files(path, suffix):
    '''
    Function: find all the files has the same specified "suffix"
    '''    
    filelist = []
    for root, subDirs, files in os.walk(path):
        for fileName in files:
            if fileName.endswith(suffix):
                filelist.append(os.path.join(root, fileName))
    return filelist

#   path = '/home/worm/wormCode_2023_0201/ICLR_2023/iiccllrr2023/MNIST/results_free_support_RWB_layered_sampling/MNIST_0_3000_60_10000_outlier_0.1_40_40_shift_0_40_100_R300_1.1_tt10_09-14_23:56'
#   suffix = '-ave'
#   res = search_files(path, suffix)
#   print("res = ",res)


