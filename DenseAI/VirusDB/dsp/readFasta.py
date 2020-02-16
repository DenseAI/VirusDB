# -*- coding:utf-8 -*-

import numpy as np
# import scipy.io as sio
import os
import h5py

def readFasta(dataSet:str):

    # AcNmb, Seq, numberOfClusters, clusterNames, pointsPerCluster

    if os.path.exists(dataSet) == False:
        return None, None, None, None, None

    if dataSet.endswith('.mat'):
        data = h5py.File(dataSet, 'r')

        # < KeysViewHDF5['#refs#', 'AcNmb', 'Seq', 'clusterNames', 'numberOfClusters', 'pointsPerCluster'] >
        AcNmb = data['AcNmb']
        obj_list = []
        for ii in range(AcNmb.shape[0]):
            obj = data[(AcNmb[ii][0])]
            str = "".join(chr(i) for i in obj[:])
            obj_list.append(str)
        AcNmb = obj_list
        #print("AcNmb: ", AcNmb)

        Seq = data['Seq']
        seq_obj_list = []
        for ii in range(Seq.shape[0]):
            obj = data[(Seq[ii][0])]
            str = "".join(chr(i) for i in obj[:])
            #print(str)
            seq_obj_list.append(str)
        Seq = seq_obj_list
        print("Seq: ", len(Seq))

        numberOfClusters = data['numberOfClusters']
        numberOfClusters = numberOfClusters[()][0][0]


        clusterNames = data['clusterNames']
        obj_list = []
        for ii in range(clusterNames.shape[0]):
            obj = data[(clusterNames[ii][0])]
            str = "".join(chr(i) for i in obj[:])
            obj_list.append(str)
        clusterNames = obj_list

        pointsPerCluster = data['pointsPerCluster']
        obj_list = []
        for ii in range(pointsPerCluster.shape[0]):
            obj = data[(pointsPerCluster[ii][0])]
            num = obj[()][0][0]
            obj_list.append(num)
        pointsPerCluster = obj_list

        return AcNmb, Seq, numberOfClusters, clusterNames, pointsPerCluster
    else:
        print()




# if __name__ == '__main__':
#     # dataset = '/home/huanghaiping/Research/Software/MLDSP-master/DataBase/Primates.mat'
#     # AcNmb, Seq, numberOfClusters, clusterNames, pointsPerCluster = readFasta(dataset)
#     # # print(AcNmb)
#     # # print(Seq)
#     # print(numberOfClusters)
#     # print(clusterNames)
#     # print(pointsPerCluster)
#
#     #from skbio.core.distance import DistanceMatrix
#     from skbio.stats.distance import DissimilarityMatrix
#     import numpy as np
#
#     data = np.array([[0.0, 0.5, 1.0],
#                      [0.5, 0.0, 0.75],
#                      [1.0, 0.75, 0.0]])
#     ids = ["a", "b", "c"]
#     dm_from_np = DissimilarityMatrix(data, ids)
#     print(dm_from_np)
#
#     dm_from_np = np.corrcoef(data)
#     print(dm_from_np)
#     D = (1 - dm_from_np)/2;
#     print(D)
