#! /usr/bin/env python
# coding=utf-8
#================================================================
#
#   Editor      : PyCharm
#   File name   : AP.py
#   Author      : Jingyi Wang
#   Created date: 2020/8/20 10:27
#   Description : AP(Affinity Propagation)
#
#================================================================
import os
from sklearn.cluster import AffinityPropagation
import numpy as np
from sklearn.metrics import adjusted_mutual_info_score, adjusted_rand_score
import pandas as pd
'''
AffinityPropagation(damping=0.5, max_iter=200, convergence_iter=15, copy=True, preference=None, affinity=’euclidean’, verbose=False)
damping : float, optional, default: 0.5,阻尼系数,默认值0.5
max_iter : int, optional, default: 200,最大迭代次数,默认值是200
convergence_iter : int, optional, default: 15,在停止收敛的估计集群数量上没有变化的迭代次数。默认15
copy : boolean, optional, default: True,布尔值,可选,默认为true,即允许对输入数据的复制
preference : array-like, shape (n_samples,) or float, optional,近似数组,每个点的偏好 - 具有较大偏好值的点更可能被选为聚类的中心点。 簇的数量，即集群的数量受输入偏好值的影响。 如果该项未作为参数，则选择输入相似度的中位数作为偏好
affinity : string, optional, default=``euclidean``目前支持计算预欧几里得距离。 即点之间的负平方欧氏距离。
verbose : boolean, optional, default: False
'''
# A 0.8 35
# Aggregation 0.9
#
def ap(dataset, gb0, d, p):
    gt = np.loadtxt('finaldata/' + dataset + '_new.txt')[:, -1]
    similarity = -gb0

    af = AffinityPropagation(affinity='precomputed', damping=d, preference=p)\
        .fit(similarity)
    n_clusters = len(af.cluster_centers_indices_)
    pred = af.labels_+1
    np.savetxt('pred/AP/' + dataset + '_pred.txt', pred)

    ari = adjusted_rand_score(gt, pred)
    ami = adjusted_mutual_info_score(gt, pred)
    return n_clusters, ari, ami

if __name__ == '__main__':
    datasets = [['A', 'Aggregation', 'B', 'Flame', 'S3']
        , ['five_cluster', 'four_cluster', 'Spiral2', 'three_cluster', 'ThreeCircles', 'two_cluster', 'Twomoons']
        , ['circle', 'cth', 'db', 'db3', 'E6', 'fc1', 'line', 'ls', 'sk', 'sn', 'Spiral3']
        , ['Jain']]
    gt_n_clusters = [[5, 7, 5, 2, 15]
        , [5, 4, 2, 3, 3, 2, 2]
        , [3, 4, 4, 4, 7, 5, 4, 6, 3, 5, 3]
        , [2]]
    print(datasets)
    print(gt_n_clusters)

    # ds=[0.7,0.8,0.9]
    # ps=[-1,-10,-50,-100,-500,-1000,-3000,-5000]
    ds=[0.95]
    ps=[-5000]
    bestrm=-1
    for d in ds:
        for p in ps:
            rm_all,amiari=[],[]
            for i in range(len(datasets)):
                for j in range(len(datasets[i])):
                    dataset = datasets[i][j]
                    if i != 3:
                        gb_path = '/data/wangjingyi/G2-DBCANN/dataset' + str(i + 1) + '/' + dataset + '.gb0'
                    else:
                        gb_path = '/data/wangjingyi/G2-DBCANN/dataset' + str(i) + '/' + dataset + '.gbd'
                    gb = np.loadtxt(gb_path)
                    n_clusters, ari, ami = ap(dataset, gb, d, p)
                    print(dataset.center(15, ' ') + ' n_clusters=%d ami=%.4f ari=%.4f' % (n_clusters, ari, ami))
                    rm_all.append((ari + ami) / 2)
                    amiari.append([ami,ari])
            rm = sum(rm_all) / len(rm_all)
            if rm>bestrm:
                bestrm=rm
                bestdp=(d,p)
            amiari=np.array(amiari)
            # print('ami')
            # for aaaaa in range(amiari.shape[0]):
            #     print(amiari[aaaaa][0])
            # print('ari')
            # for aaaaa in range(amiari.shape[0]):
            #     print(amiari[aaaaa][1])
            np.savetxt('pred/AP/amiari.txt',np.array(amiari))
            print('ami_ave=%.4f'%np.mean(amiari[:,0]),'  ari_ave=%.4f'%np.mean(amiari[:,1]))
            print('****** d=',d,'  p=',p,'  rm=',rm,'\n')
    print('bestrm',bestrm,'  bestdp',bestdp)

# A 0.9 -18.5~-24 n_clusters=5 ami=0.9460 ari=0.9480
# Aggregation 0.86 -280 n_clusters=7 ami=0.7680 ari=0.8934
# B 0.84 -60 n_clusters=5 ami=0.8691 ari=0.8757
# Flame 0.5~0.8 -350 n_clusters=2 ami=1.0000 ari=1.0000
# S3 0.9 -10000000.00 n_clusters=15 ami=0.7410 ari=0.8015

# five_cluster 0.73 -250~-400 n_clusters=5 ami=0.9930 ari=0.9829
# four_cluster 0.63 -250 n_clusters=4 ami=0.9682 ari=0.9571
# Spiral2 0.9 -3000 n_clusters=2 ami=1.0000 ari=1.0000
# three_cluster 0.58 -75 n_clusters=3 ami=1.0000 ari=1.0000
# ThreeCircles 0.98 -5000 n_clusters=3 ami=1.0000 ari=1.0000
# two_cluster 0.7 -200 n_clusters=4 ami=0.8291 ari=0.7825
# Twomoons 0.85 -800 n_clusters=2 ami=1.0000 ari=1.0000

# circle 0.95 -800000.00 n_clusters=3 ami=0.9961 ari=0.9940
# cth 0.95~0.97 -10000~-12500 n_clusters=4 ami=1.0000 ari=1.0000
# db 0.8~0.85 -40000~-50000 n_clusters=4 ami=1.0000 ari=1.0000
# db3 0.8~0.9 -10000~-40000 n_clusters=4 ami=1.0000 ari=1.0000
# E6 0.95 -21000.00 n_clusters=7 ami=0.9979 ari=0.9962
# fc1 0.7 -4000 n_clusters=5 ami=0.9960 ari=0.9918
# line 0.8 -13000~-15000 n_clusters=4 ami=0.9956 ari=0.9945
# ls 0.9 -50000 n_clusters=6 ami=0.9990 ari=0.9985
# sk 0.9 -50000 n_clusters=3 ami=0.9468 ari=0.8939
# sn 0.7 -1000 n_clusters=5 ami=1.0000 ari=1.0000
# Spiral3 0.8 -1000 n_clusters=3 ami=1.0000 ari=1.0000

# Jain 0.8 -2000~-5000 n_clusters=2 ami=1.0000 ari=1.0000