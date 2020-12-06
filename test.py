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

# !!!jain.txt不用离群点！！！
datasets = [['A', 'Aggregation', 'B', 'Flame', 'S3']
    , ['five_cluster', 'four_cluster', 'Spiral2', 'three_cluster', 'ThreeCircles', 'two_cluster', 'Twomoons']
    , ['circle', 'cth', 'db', 'db3', 'E6', 'fc1', 'line', 'ls', 'sk', 'sn', 'Spiral3']]
gt_n_clusters = [[5, 7, 5, 2, 15]
    , [5, 4, 2, 3, 3, 2, 2]
    , [3, 4, 4, 4, 7, 5, 4, 6, 3, 5, 3]]
# i = 0
# j = 0

i=2
for j in range(10,11):
    dataset = datasets[i][j]
    gb_path = '/data/wangjingyi/G2-DBCANN/dataset' \
              + str(i + 1) + '/' + dataset + '.gb0'
    gb = np.loadtxt(gb_path)
    # gt = np.loadtxt('finaldata/' + dataset + '_new.txt')[:, 2]
    # ap(dataset, gb)
    # dbscan(dataset, gb)
    spectralClustering(dataset, gt_n_clusters[i][j])

# print(dataset, ', pred_n_clusters=%d , gt_n_clusters=%d' % (n_clusters, gt_n_clusters[i][j]))
# np.savetxt('result\\AP\\' + dataset + '.txt', pred, fmt='%d')
