#! /usr/bin/env python
# coding=utf-8
#================================================================
#
#   Editor      : PyCharm
#   File name   : SpectralClustering.py
#   Author      : Jingyi Wang
#   Created date: 2020/8/20 10:43
#   Description : 
#
#================================================================

from sklearn.cluster import SpectralClustering
import numpy as np
from sklearn.metrics import adjusted_mutual_info_score, adjusted_rand_score
import pandas as pd
import os

# 参数说明 https://www.cnblogs.com/pinard/p/6235920.html
def spectralClustering(dataset, n_cluster, g):
    data_gt = np.loadtxt('finaldata/' + dataset + '_new.txt')
    data=data_gt[:, :-1]
    gt = data_gt[:, -1]

    s = SpectralClustering(n_clusters=n_cluster, gamma=g).fit(data)
    # s = SpectralClustering(affinity='nearest_neighbors',n_clusters=n_cluster,n_neighbors=g).fit(data)
    pred = s.labels_+1
    np.savetxt('pred/SpectralClustering/' + dataset + '_pred.txt', pred)

    ari = adjusted_rand_score(gt, pred)
    ami = adjusted_mutual_info_score(gt, pred)
    return ari, ami

if __name__ == '__main__':
    datasets = [['A', 'Aggregation', 'B', 'Flame', 'S3']
        , ['five_cluster', 'four_cluster', 'Spiral2', 'three_cluster', 'ThreeCircles', 'two_cluster', 'Twomoons']
        , ['circle', 'cth', 'db', 'db3', 'E6', 'fc1', 'line', 'ls', 'sk', 'sn', 'Spiral3']
        , ['Jain']]
    gt_n_clusters = [[5, 7, 5, 2, 15]
        , [5, 4, 2, 3, 3, 2, 2]
        , [3, 4, 4, 4, 7, 5, 4, 6, 3, 5, 3]
        , [2]]

    # gs=[0.01,0.05,0.1,0.5,1,2,4,8,16,32]
    gs=[0.45]
    bestrm=-1
    for g in gs:
        rm_all,amiari=[],[]
        for i in range(len(datasets)):
            for j in range(len(datasets[i])):
                dataset = datasets[i][j]
                gb_path = '/data/wangjingyi/G2-DBCANN/dataset' + str(i + 1) + '/' + dataset + '.gb0'
                #gb = np.loadtxt(gb_path)
                ari,ami=spectralClustering(dataset, gt_n_clusters[i][j],g)
                print(dataset.center(15, ' ') + ' n_clusters=%d ami=%.4f ari=%.4f' % (gt_n_clusters[i][j], ari, ami))
                rm_all.append((ari + ami) / 2)
                amiari.append([ami,ari])
        print(amiari)
        amiari=np.array(amiari)
        # print('ami')
        # for aaaaa in range(amiari.shape[0]):
        #     print(amiari[aaaaa][0])
        # print('ari')
        # for aaaaa in range(amiari.shape[0]):
        #     print(amiari[aaaaa][1])
        np.savetxt('pred/SpectralClustering/amiari.txt',amiari)
        print('ami_ave=%.4f'%np.mean(amiari[:,0]),'  ari_ave=%.4f'%np.mean(amiari[:,1]))
        rm = sum(rm_all) / len(rm_all)
        if rm >bestrm:
            bestrm=rm
            bestg=g
        print('****** g',g,'  rm',rm,'\n')
    print('bestrm',bestrm,'  g',bestg,'\n\n')
