#! /usr/bin/env python
# coding=utf-8
#================================================================
#
#   Editor      : PyCharm
#   File name   : test.py
#   Author      : Jingyi Wang
#   Created date: 2020/8/21 11:09
#   Description : 
#
#================================================================

from AP import ap
from SpectralClustering import spectralClustering
from DBSCAN import dbscan
import numpy as np
from sklearn.metrics import adjusted_mutual_info_score, adjusted_rand_score

datasets=['iris','wine','wdbc','seeds','libras_movement','ionosphere','ecoli','parkinsons','dermatology','balance_scale']
gt_n_clusters=[3,3,2,3,7,15,2,8,2,6,3]
    
for i in range(len(datasets)):
    dataset = datasets[i]
    gb_path = 'newdataset/' + dataset + '.gb0'
    gb = np.loadtxt(gb_path)
    # gt = np.loadtxt('finaldata/' + dataset + '_new.txt')[:, 2]
    ap(dataset, gb)
    # dbscan(dataset, gb)
    # spectralClustering(dataset, gt_n_clusters[i])

# print(dataset, ', pred_n_clusters=%d , gt_n_clusters=%d' % (n_clusters, gt_n_clusters[i][j]))
# np.savetxt('result\\AP\\' + dataset + '.txt', pred, fmt='%d')
