import sys,os
import numpy as np
from numpy import array,ones,sum
sys.path.append(os.path.abspath( os.path.dirname(__file__) + '/..' ))
sys.path.append(os.path.abspath( os.path.dirname(__file__) + '/../../../WB' ))
#   print("sys.path = ",sys.path)
import Plot as myPltPlus
from Tools import search_files


pwd = os.path.dirname(__file__) 



#------------------plot-------Ensemble_clustering_1000_001-----

wormSupTitle = "Layered sampling" ##########
fileName_ave = search_files(pwd,'-ave')[0]
print("fileName_ave = ",fileName_ave)
fileName_std = search_files(pwd,'-std')[0]
myaveResult = np.loadtxt(fileName_ave)
mystdResult = np.loadtxt(fileName_std)
myaveResult = myaveResult.T; mystdResult = mystdResult.T
imgsPath = pwd
xLabelNames=[10,20,30,40,50,60]   ##########
ylimList=[[0,600],[0,3],[1013,1017]]
myPltPlus.wormPlot_AverageResult_Variance_Matrix(imgsPath,myaveResult,mystdResult, xLabelNames, wormSupTitle,ylimList,outputPointer=pwd)


