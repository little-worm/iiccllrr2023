import torch
import torchvision
from functools import reduce
import numpy as np,array
from collections import Counter
import itertools  # cartesian product
from sklearn.cluster import KMeans
import ot
import random
import time
import sys
import os
sys.path.append(os.path.abspath( os.path.dirname(__file__) + '/..' ))
#print("sys.path = ",sys.path)
cfd = os.path.dirname(__file__)




def loadTenKindsOfGrayImageData(outputPointer=None):
    #---------------------------------------------------------------------------------------------------
    #Functions: load ten kinds of script gray images in Minst database, they are classified by their indexes.
    #---------------------------------------------------------------------------------------------------
    batch_size_train = 60000 # batch_size_train = 60000
    batch_size_test = 10000 # batch_size_test = 10000


    train_loader = torch.utils.data.DataLoader(
      torchvision.datasets.MNIST(root= cfd + "../../data", train=True, download=True,
                                 transform=torchvision.transforms.Compose([
                                   torchvision.transforms.ToTensor()
                                  #   ,torchvision.transforms.Normalize((0.1307,), (0.3081,))
                                 ])),
      batch_size=batch_size_train, shuffle=True)

    test_loader = torch.utils.data.DataLoader(
      torchvision.datasets.MNIST(root=cfd + "../../data", train=False, download=True,
                                 transform=torchvision.transforms.Compose([
                                   torchvision.transforms.ToTensor(),
                                   torchvision.transforms.Normalize(
                                     (0.1307,), (0.3081,))
                                 ])),
      batch_size=batch_size_test, shuffle=True)


    #------------------------------------------------------------------------
    #sort "train_set" into 10 sublist,  
    #the sublist "tenKindsOfGrayImageData[label]" contains all the image datas labeled as "label"
    # Return tenKindsOfGrayImageData
    #---------------------According to labels, we sort 60000 gray images (in train_loader) into 10 class--------------------------------
    examples = enumerate(train_loader)
    batch_idx, (example_data, example_targets) = next(examples)
    example_data = example_data.reshape(batch_size_train,784)
    example_data = example_data
    example_data = example_data.tolist()

    tenKindsOfGrayImageData = [[],[],[],[],[],[],[],[],[],[]]
    [ tenKindsOfGrayImageData[label].append(data) for data,label in zip(example_data,example_targets)]

    #print("example_data = ",np.array(example_data).shape,file=outputPointer)

    #print("np.array(tenKindsOfGrayImageData[0]).shape = ",np.array(tenKindsOfGrayImageData[0]).shape,file=outputPointer)
    #print("np.array(tenKindsOfGrayImageData[1]).shape = ",np.array(tenKindsOfGrayImageData[1]).shape,file=outputPointer)
    #print("example_targets = ",example_targets)
    #example_targets.shape
    return tenKindsOfGrayImageData












def orginalDatesetForLabel_i(tenKindsOfGrayImageData, label=0, normDataSize=27, noiseDataSize=3, outputPointer=None):
    #-----------------------------------------------------------------------------------------------
    #Function: For "Label = i ",we return 3000 images.  2700 normal + 300 noise
    #---------------------------------------------------------------------------------------------------
    #Parameters:     normDataSize =  2700    the number of normal images of label = i
    #                noiseDataSize =  300    the number of noise images 
    #print("label = ",label,file=outputPointer)
    labelList = [i for i in range(10)]
    dataNoiseSet0 = [] 
    dataNoiseSet = []# select noise image datas from this set
    #Sample normal images data -----------------------------------------------------
    dataNormal = random.choices(tenKindsOfGrayImageData[label], weights=np.ones(len(tenKindsOfGrayImageData[label]),np.integer) , k=normDataSize)

    #Sample noise images data -----------------------------------------------------
    labelList.remove(label) #Remove the normal label
    #print(labelList)
    [dataNoiseSet0.append(tenKindsOfGrayImageData[i]) for i in labelList]
    #print(len(dataNoiseSet0[0]))

    for i in labelList:
        dataNoiseSet = dataNoiseSet + tenKindsOfGrayImageData[i]
    #    print(len(dataNoiseSet))

    dataNoise = random.choices(population=dataNoiseSet, weights=np.ones(len(dataNoiseSet)), k=noiseDataSize)
    #print("len(dataNoise) = ",len(dataNoise),file=outputPointer)
    OrginalSetForLabel_i = dataNoise + dataNormal # dataNoise and dataNormal are our orginal sample set (2700+300)
    #print("len(OrginalSetForLabel_i) = ",len(OrginalSetForLabel_i),file=outputPointer)
    return OrginalSetForLabel_i









def turnGrayImages_1D_to_2D(oneDimList0, outputPointer=None):
    #---Turn grey images into 2D structures
    oneDimList = np.copy(oneDimList0)
    aix = [i for i in range(28)]
    locationForImages = [i for i in itertools.product(aix,aix)]
    oneDimList = np.array(oneDimList).reshape(len(oneDimList),784,1)
    #print(oneDimList.shape)
    twoDimGrayImages = []

    #print(len(oneDimList))

    for i in range(len(oneDimList)):
        #    tmp = [list(l1) + list(l2) for l1,l2 in zip(locationForImages,oneDimList[i]) ]
        tmp = [list(l1)  for l1,l2 in zip(locationForImages,oneDimList[i]) if l2 > 0.000000000001]

    #    print(len(tmp))
        if len(tmp)>60:
            twoDimGrayImages.append(tmp)
    #print(np.array(twoDimGrayImages).shape)
    #print(len(twoDimGrayImages[0]))
    return twoDimGrayImages
    
    


def turnGrayImages_1D_to_3D(oneDimList0, grayScaleFor3DMNIST, outputPointer=None):
    #---Turn grey images into 3D structures
    oneDimList = np.copy(oneDimList0)
    aix = [i for i in range(28)]
    locationForImages = [i for i in itertools.product(aix,aix)]
    oneDimList = np.array(oneDimList).reshape(len(oneDimList),784,1) * grayScaleFor3DMNIST 
    #print(oneDimList.shape)
    threeDimGrayImages = []
    #print(len(oneDimList))
    for i in range(len(oneDimList)):
        tmp = [ (list(l1)+list(l2)) for l1,l2 in zip(locationForImages,oneDimList[i]) ] 
        #   print("tmp = ",tmp[0:10])
        #   tmp = [list(l1)  for l1,l2 in zip(locationForImages,oneDimList[i]) if l2 > 0.0001]
        threeDimGrayImages.append(tmp)
        #threeDimGrayImages = array(threeDimGrayImages)    
    #   print("threeDimGrayImages = ",len(threeDimGrayImages),len(threeDimGrayImages[0]),len(threeDimGrayImages[1]))
    return threeDimGrayImages
    
    








def turn23DimImagesToDistribution(twoDimImagesDatabase, kClusters = 50, outputPointer=None):
    #twoDimImagesDatabase = np.copy(twoDimImagesDatabase0)
    location_A = []  # the location of distribution
    weights_A = []  # the weight(value) of distribution
    for i in range(len(twoDimImagesDatabase)):
        tmpKmeans = KMeans(n_clusters= kClusters, n_init=1).fit(twoDimImagesDatabase[i])
        tmpKcenter = tmpKmeans.cluster_centers_
        location_A.append(tmpKcenter)

        label = tmpKmeans.labels_
        labelDictionary = Counter(label)
    #    print(labelDictionary)
        labelDistribution = [ labelDictionary.get(j) / len(twoDimImagesDatabase[i]) for j in range(kClusters)] 
    #    print((labelDistribution))
        weights_A.append(labelDistribution)
    return location_A,weights_A
















if __name__ == '__main__': 
    print("Hello worm!!!")
    ##  
    # ----------------Test---------------loadTenKindsOfGrayImageData()-----------------
    print("----------------Test---------------loadTenKindsOfGrayImageData()-----------------")
    test_tenKindsOfGrayImageData = loadTenKindsOfGrayImageData()
    print(type(test_tenKindsOfGrayImageData))
  



    # --------------Test------------------orginalDatesetForLabel_i()--------------------------------
    print("--------------Test------------------orginalDatesetForLabel_i()-----------------------")
    testOrginalSetForLabel_i = orginalDatesetForLabel_i(test_tenKindsOfGrayImageData, label=0, normDataSize=270, noiseDataSize=30)
    print(np.array(testOrginalSetForLabel_i).shape)
    #print("111111111111 = ",max(testOrginalSetForLabel_i[0]),min(testOrginalSetForLabel_i[0]))






#    # -----------Test-----turnGrayImages_1D_to_2D()--------------------------------
#    print("-----------Test-----turnGrayImages_1D_to_2D()-------------------------")
#    oneDimList = np.copy(testOrginalSetForLabel_i)
#    test_twoDimGrayImages = turnGrayImages_1D_to_2D(oneDimList)
#    print("test_twoDimGrayImages = ",type(test_twoDimGrayImages))
#    print("test_twoDimGrayImages = ",len(test_twoDimGrayImages[0]))
#    print("test_twoDimGrayImages = ",len(test_twoDimGrayImages[1]))
#    test_index = random.randint(0,30)
#    print(len(test_twoDimGrayImages[test_index]))





    # -----------Test-----turnGrayImages_1D_to_3D()--------------------------------

    print("-----------Test-----turnGrayImages_1D_to_3D()-------------------------")
    grayScaleFor3DMNIST = 28
    oneDimList = np.copy(testOrginalSetForLabel_i)
    test_twoDimGrayImages = turnGrayImages_1D_to_3D(oneDimList,grayScaleFor3DMNIST)
    print("test_twoDimGrayImages = ",type(test_twoDimGrayImages))
    print("test_twoDimGrayImages = ",len(test_twoDimGrayImages[0]))
    print("test_twoDimGrayImages = ",len(test_twoDimGrayImages[1]))
    test_index = random.randint(0,30)
    print(len(test_twoDimGrayImages[test_index]))




    # ----------Test---------------turn2DimImagesToDistribution()-------------------------
    print("----------Test---------------turn2DimImagesToDistribution()--------------------")
    kClusters = 50
    testLocation_A,testWeights_A = turn23DimImagesToDistribution(test_twoDimGrayImages, kClusters)  
    print(len(testLocation_A))    
    print(len(testLocation_A[0])) 



