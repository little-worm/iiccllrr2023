# iiccllrr2023

# Conda: CSDN链接：https://blog.csdn.net/lx_ros/article/details/123597208
- conda --version : 查看版本
- conda update conda : 更新至最新版本 
- conda update anaconda : 更新anaconda 
- conda create --name your-env :创建环境 
- conda create --name your-env your-pkg ;  创建环境并同时安装指定包: 
  conda create --name worm python=3.5
- conda activate your-env : 激活环境 
- conda info --envs : 查看已经创建的环境 
- conda remove --name ENVNAME --all :完整的删除一个环境 
  conda remove -p /var/www/myblog/felixvenv --all
- conda create --clone ENVNAME --name NEWENV : 复制1个环境 
- conda env export --name ENVNAME > envname.yml
  conda env create -f=/path/to/environment.yml -n your-env-name 
  :将环境导出到yaml文件，用于创建新的环境 
- conda env list : 查看服务器上所有的 conda 环境  




输入如下指令查看CUDA版本：nvidia-smi





- sudo aptitude update 
- conda install pytorch torchvision torchaudio pytorch-cuda=11.7 -c pytorch -c nvidia
- pip3 install -U scikit-learn
- pip install POT
- pip install nilearn
- conda install -c conda-forge matplotlib






#python 路径设置
- import os
- import sys
- cfd = os.path.dirname(__file__)
- print("cfd = ",cfd)
- sys.path.append(os.path.abspath(os.path.join(cfd)))
- sys.path.append(os.path.abspath(os.path.join(cfd+'..')))
- sys.path.append(os.path.abspath(os.path.join(cfd+'../..')))
sys.path.append(os.path.abspath( os.path.dirname(__file__) + '/..' ))

- print("sys.path = ",sys.path)




HCP dataset download: 
https://github.com/jokedurnez/HCP_download/tree/master