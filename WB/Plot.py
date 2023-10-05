
import numpy as np
import matplotlib.pyplot as plt
from pylab import *                                 #支持中文
#mpl.rcParams['font.sans-serif'] = ['SimHei']
plt.rcParams['font.size'] = 24
import time
import os 
from Tools import my_mkdir





def wormPlot_AverageResult_Variance_Matrix(imgsPath,myaveResult,myvarResult, xLabelNames=[10,20,30,40,50,60], wormSupTitle="Worm",ylimList=None,outputPointer=None):
    # example data
    mystdResult = [np.sqrt(i) for i in myvarResult]
    tmpImgsFolderName = '/imgs'
    my_mkdir(imgsPath+tmpImgsFolderName)
  
    #print("imgsPath = ",imgsPath,file=outputPointer)
    x = np.arange(6)

    # Now switch to a more OO interface to exercise more features.
    fig, (ax_l, ax_c, ax_r) = plt.subplots(nrows=1, ncols=3, sharex=True, figsize=(40, 10))

    #ax_l.set_title('all errorbars')
    ax_l.plot(x,myaveResult[0], marker='o', mec='r', mfc='w', color="r", label='our coreset')
    ax_l.errorbar(x, myaveResult[0], yerr=mystdResult[0],elinewidth=1.5,capsize=3,barsabove=True)
    ax_l.plot(x,myaveResult[3], marker='*', ms=10, color="b", label='uniform sampling')
    ax_l.errorbar(x, myaveResult[3], yerr=mystdResult[3],elinewidth=1.5,capsize=3,barsabove=True)
    ax_l.set_xlabel("$\Gamma$")
    ax_l.set_ylabel("running time")  
    ax_l.legend()

    #ax_c.set_title('all errorbars')
    bar_width = 0.35  # the width of the bars
    ax_c.bar(x,myaveResult[1],bar_width,align="center",color="r",label="our coreset",alpha=0.5)
    ax_c.errorbar(x, myaveResult[1], yerr=mystdResult[1],elinewidth=1.5,capsize=3,barsabove=True,fmt="o")
    #ax_c.errorbar(x, myaveResult[1], yerr=mystdResult[1],xerr=None,fmt="o")

    ax_c.bar(x + bar_width, myaveResult[4], bar_width, color="b", label='uniform sampling')
    ax_c.errorbar(x+ bar_width, myaveResult[4], yerr=mystdResult[4],elinewidth=1.5,capsize=3,barsabove=True,fmt="o")
    ax_c.set_xlabel("$\Gamma$")
    ax_c.set_ylabel("WD")
    ax_c.legend()

    #ax_r.set_title('all errorbars')
    bar_width = 0.35  # the width of the bars
    ax_r.bar(x,myaveResult[2],bar_width,align="center",color="r",label="our coreset",alpha=0.5)
    ax_r.errorbar(x, myaveResult[2], yerr=mystdResult[2],elinewidth=1.5,capsize=3,barsabove=True,fmt="o")
  
    ax_r.bar(x + bar_width, myaveResult[5], bar_width, color="b", label='uniform sampling')
    ax_r.errorbar(x+ bar_width, myaveResult[5], yerr=mystdResult[5],elinewidth=1.5,capsize=3,barsabove=True,fmt="o")
    
    ax_r.set_xlabel("$\Gamma$")
    ax_r.set_ylabel("cost")
    ax_r.legend()
    if ylimList != None:
        if len(ylimList) > 0:
            ax_l.set_ylim(ylimList[0][0],ylimList[0][1]) 
        if len(ylimList) > 1:
            ax_c.set_ylim(ylimList[1][0],ylimList[1][1])
        if len(ylimList) > 2: 
            ax_r.set_ylim(ylimList[2][0],ylimList[2][1])
 
    plt.xticks(x, xLabelNames, rotation=0)

    #fig.suptitle(wormSupTitle,y=-0.00001)
    #fig.suptitle(wormSupTitle)
    timeStamp = time.strftime("%Y_%m_%d_%H", time.localtime())
    plt.savefig(imgsPath + tmpImgsFolderName + '/'+str(wormSupTitle)+'.png')
#    print("os.getcwd() = ",os.getcwd()) 
    print("imgsPath = ",imgsPath)
    plt.show()
    