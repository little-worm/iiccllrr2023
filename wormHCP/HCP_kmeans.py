import itertools  # cartesian product
#import matplotlib.pyplot as plt
import numpy as np
import os
from sklearn.cluster import KMeans
from collections import Counter
#from nilearn import plotting 
#NII_DIR='./nii_dir'    #nii文件所在root目录
import nibabel as nibaptitu
#pwd = os.path.dirname(__file__)
pwd =  os.path.dirname(os.path.realpath('__file__'))
pwd = pwd + '/'
father_path=os.path.abspath(os.path.dirname(pwd)+os.path.sep+".")
father_path = father_path + '/' 
grader_father_path=os.path.abspath(os.path.dirname(pwd)+os.path.sep+"..")
grader_father_path = grader_father_path + '/'
print("pwd                = ",pwd)
print("father_path        = ",father_path)
print("grader_father_path = ",grader_father_path)

 
import nibabel as nib
 
def read_nii_file1(nii_path):
    '''
    根据nii文件路径读取nii图像
    '''
    nii_image=nib.load(nii_path)
    return nii_image




def getOrginalDataserForBrains(imgPath):
    orginalDataset = []
    filenames=os.listdir(imgPath)
    filenames = [tmpFile for tmpFile in filenames if '.nii.gz' in tmpFile]
    #print("filenames = ",filenames)
    for tmpFile in filenames:
        example_filename = imgPath + tmpFile  
        nii_image1=read_nii_file1(example_filename)
        tmpBrainList = nii_image1.get_fdata()
    #    print("-----------------------------")
        print("tmpBrainList = ",tmpBrainList.shape[-1]) 
        for i in range(tmpBrainList.shape[-1]):
            orginalDataset.append(tmpBrainList[:,:,:,i])

    orginalDataset = np.array(orginalDataset)
    #print(orginalDataset.shape)
    return orginalDataset









def turn_3DvolumnBrain_to_distribution(brainList=None,shrink=300,kClusters=300): 
    location_weights_A = []
    cou = 0
    for brain in brainList:
        cou = cou+1
        print("cou = ",cou)
        brain = np.array(brain)
        print("brain.shape = ",brain.shape)
        aix1_size, aix2_size, aix3_size = brain.shape 
        #print("max(brain.ravel()) = ",max(brain.ravel()))
        #print("min(brain.ravel()) = ",min(brain.ravel()))
        aix1_list = [i for i in range(aix1_size)]
        aix2_list = [i for i in range(aix2_size)]
        aix3_list = [i for i in range(aix3_size)]
        #print("aix1_size, aix2_size, aix3_size  = ",aix1_size, aix2_size, aix3_size  )
        greyValueBrain = [[i[0],i[1],i[2],value/shrink] for i,value in zip(itertools.product(aix1_list,aix2_list,aix3_list),brain.ravel())]
        #print("greyValueBrain = ",greyValueBrain)
        greyValueBrain = np.array(greyValueBrain)
        greyValueBrain = greyValueBrain.reshape(aix1_size*aix2_size*aix3_size,4)
        #print("greyValueBrain = ",greyValueBrain.shape)
        tmpPcdCloud = [tmp for tmp in greyValueBrain if tmp[3]>0.0001]
        #print("tmpPcdCloud = ",np.array(tmpPcdCloud).shape)
        tmpKmeans = KMeans(n_clusters= kClusters,n_init=1).fit(tmpPcdCloud)
        tmpKcenter = tmpKmeans.cluster_centers_
        #print("tmpKcenter = ",tmpKcenter)
        
        label = tmpKmeans.labels_
        labelDictionary = Counter(label)
        #print(labelDictionary)
        labelDistribution = [ labelDictionary.get(j) / len(tmpPcdCloud) for j in range(kClusters)] 
        
        labelDistribution = np.array(labelDistribution).reshape((kClusters,1))
        #print("labelDistribution = ",(labelDistribution))
        
        location_weights_Ai = np.hstack((np.array(tmpKcenter), np.array(labelDistribution)))
        location_weights_A.append(location_weights_Ai)
    return location_weights_A





kClusters=2


imgPath = pwd+'../wormHCP/fourDimBrains/'
orginalDataset = getOrginalDataserForBrains(imgPath)

#orginalDataset = orginalDataset[:2]

location_weights_A = turn_3DvolumnBrain_to_distribution(brainList=orginalDataset,shrink=300,kClusters=kClusters)

numbersOf3dDShapes = len(orginalDataset)
tmpMatrix = np.array(location_weights_A).reshape(kClusters*numbersOf3dDShapes,5)
np.savetxt(pwd+'kCluster'+str(kClusters) + 'Kmeans.brainmatrix_2023_0830',tmpMatrix,fmt='%f',delimiter=' ',newline='\r\n')

## test1 = [i for i in greyValueBrain if i[3]>1 ]
## print(test1)






