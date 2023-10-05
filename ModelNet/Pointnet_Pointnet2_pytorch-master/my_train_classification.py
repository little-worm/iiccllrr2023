import os,sys,torch,argparse
import numpy as np
from ModelNetDataLoader import ModelNetDataLoader
from sklearn.cluster import KMeans
from collections import Counter

cfd = os.path.dirname(os.path.abspath(__file__))
sys.path.append(os.path.join(cfd, 'models'))

def parse_args():
    '''PARAMETERS'''
    parser = argparse.ArgumentParser('training')
    parser.add_argument('--use_cpu', action='store_true', default=False, help='use cpu mode')
    parser.add_argument('--gpu', type=str, default='0', help='specify gpu device')
    parser.add_argument('--batch_size', type=int, default=1000, help='batch size in training')
    parser.add_argument('--model', default='pointnet_cls', help='model name [default: pointnet_cls]')
    parser.add_argument('--num_category', default=40, type=int, choices=[10, 40],  help='training on ModelNet10/40')
    parser.add_argument('--epoch', default=200, type=int, help='number of epoch in training')
    parser.add_argument('--learning_rate', default=0.001, type=float, help='learning rate in training')
    parser.add_argument('--num_point', type=int, default=1024, help='Point Number')
    parser.add_argument('--optimizer', type=str, default='Adam', help='optimizer for training')
    parser.add_argument('--log_dir', type=str, default=None, help='experiment root')
    parser.add_argument('--decay_rate', type=float, default=1e-4, help='decay rate')
    parser.add_argument('--use_normals', action='store_true', default=False, help='use normals')
    parser.add_argument('--process_data', action='store_true', default=False, help='save data offline')
    parser.add_argument('--use_uniform_sample', action='store_true', default=False, help='use uniform sampiling')
    return parser.parse_args()




def turnShapesToDistribution(shapeLists, kClusters = 60):
    #---------------------------------------------------------------------------------
    #Function:turn shapes into distributions 
    #parameters:shapeLists: point cloud list from ModelNet40
    #           kClusters: we group 3D-points in each .pcd file into kClusters=2000 clusters
    #---------------------------------------------------------------------------------
    location_weights_A = []
    for shape in shapeLists:
        tmpKmeans = KMeans(n_clusters= kClusters,n_init=1).fit(shape)
        tmpKcenter = tmpKmeans.cluster_centers_        
        label = tmpKmeans.labels_
        labelDictionary = Counter(label)
        labelDistribution = [ labelDictionary.get(j) / len(shape) for j in range(kClusters)] 
        labelDistribution = np.array(labelDistribution).reshape((kClusters,1))   
        location_weights_Ai = np.hstack((np.array(tmpKcenter), np.array(labelDistribution)))
        location_weights_A.append(location_weights_Ai)
    return np.array(location_weights_A)




args = parse_args()
data_path = '../../../data/modelnet40_normal_resampled/'
train_dataset = ModelNetDataLoader(root=data_path, args=args, split='train', process_data=args.process_data)
test_dataset = ModelNetDataLoader(root=data_path, args=args, split='test', process_data=args.process_data)
trainDataLoader = torch.utils.data.DataLoader(train_dataset, batch_size=args.batch_size, shuffle=False, num_workers=10, drop_last=True)
testDataLoader = torch.utils.data.DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False, num_workers=10)


mylabel = 8
myData = []
for data,label in trainDataLoader:
    tmpData = [d for d,l in zip(data,label) if l==mylabel]
    myData = myData + tmpData
    print(len(myData))

for data,label in testDataLoader:
    tmpData = [d for d,l in zip(data,label) if l==mylabel]
    print(len(tmpData))
    myData = myData + tmpData
    print(len(myData))



kClusters = 60
location_weights_A = turnShapesToDistribution(myData, kClusters)
numbersOf3dDShapes = len(myData)
tmpMatrix = np.array(location_weights_A).reshape(kClusters*numbersOf3dDShapes,4)
np.savetxt(cfd+ str(mylabel) + 'Kmeans.matrix_2023_0922',tmpMatrix,fmt='%f',delimiter=' ',newline='\r\n')


