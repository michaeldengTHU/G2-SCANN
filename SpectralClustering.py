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
    data_gt = np.loadtxt('newdataset/dataset/' + dataset + '_new.txt')
    data=data_gt[:, :-1]
    gt = data_gt[:, -1]

    s = SpectralClustering(n_clusters=n_cluster, gamma=g).fit(data)
    # s = SpectralClustering(affinity='nearest_neighbors',n_clusters=n_cluster,n_neighbors=g).fit(data)
    pred = s.labels_+1
    np.savetxt('newdataset/SpectralClustering/' + dataset + '_pred.txt', pred)

    ari = adjusted_rand_score(gt, pred)
    ami = adjusted_mutual_info_score(gt, pred)
    return ari, ami

if __name__ == '__main__':
    # datasets = [['A', 'Aggregation', 'B', 'Flame', 'S3']
    #     , ['five_cluster', 'four_cluster', 'Spiral2', 'three_cluster', 'ThreeCircles', 'two_cluster', 'Twomoons']
    #     , ['circle', 'cth', 'db', 'db3', 'E6', 'fc1', 'line', 'ls', 'sk', 'sn', 'Spiral3']
    #     , ['Jain']]
    # gt_n_clusters = [[5, 7, 5, 2, 15]
    #     , [5, 4, 2, 3, 3, 2, 2]
    #     , [3, 4, 4, 4, 7, 5, 4, 6, 3, 5, 3]
    #     , [2]]
    datasets=['iris','wdbc','seeds','libras_movement','ionosphere','dermatology']
    gt_n_clusters=[3,2,3,15,2,6]

    # gs=[0.01,0.05,0.1,0.5,1,2,4,8,16,32]
    gs=[0.45]
    results=[]
    bestrm=-1
    for g in gs:
        rm_all,amiari=[],[]
        for i in range(len(datasets)):
            dataset = datasets[i]
            ari,ami=spectralClustering(dataset, gt_n_clusters[i],g)
            print(dataset.center(15, ' ') + ' n_clusters=%d ami=%.4f ari=%.4f' % (gt_n_clusters[i], ami, ari))
            rm_all.append((ari + ami) / 2)
            amiari.append([ami,ari])
            results.append([dataset,g,gt_n_clusters[i],ami,ari])
            '''
            for j in range(len(datasets[i])):
                dataset = datasets[i][j]
                gb_path = '/data/wangjingyi/G2-DBCANN/dataset' + str(i + 1) + '/' + dataset + '.gb0'
                #gb = np.loadtxt(gb_path)
                ari,ami=spectralClustering(dataset, gt_n_clusters[i][j],g)
                print(dataset.center(15, ' ') + ' n_clusters=%d ami=%.4f ari=%.4f' % (gt_n_clusters[i][j], ari, ami))
                rm_all.append((ari + ami) / 2)
            '''
        print(amiari)
        amiari=np.array(amiari)
        print('ami')
        for aaaaa in range(amiari.shape[0]):
            print(amiari[aaaaa][0])
        print('ari')
        for aaaaa in range(amiari.shape[0]):
            print(amiari[aaaaa][1])
        np.savetxt('newdataset/SpectralClustering/amiari.txt',amiari)
        print('ami_ave=%.4f'%np.mean(amiari[:,0]),'  ari_ave=%.4f'%np.mean(amiari[:,1]))
        rm = sum(rm_all) / len(rm_all)
        if rm >bestrm:
            bestrm=rm
            bestg=g
        print('****** g',g,'  rm',rm,'\n')
    print('bestrm',bestrm,'  g',bestg,'\n\n')
    '''
    csv_path='newdataset/SpectralClustering/para.csv'
    if os.path.exists(csv_path):
        df=pd.read_csv(csv_path,index_col=0)
        newdf=pd.DataFrame(np.array(results),columns=['dataset','gamma','n_clusters','ami','ari'])
        df=df.append(newdf,ignore_index=True)
        df.to_csv(csv_path)
    else:
        df=pd.DataFrame(np.array(results),columns=['dataset','gamma','n_clusters','ami','ari'])
        df.to_csv(csv_path)
    '''
# A 25.00 n_clusters=5 ami=0.9552 ari=0.9490
# Aggregation 0.30 n_clusters=7 ami=0.9949 ari=0.9914
# B 150.00 n_clusters=5 ami=0.9779 ari=0.9682
# Flame 0.50 n_clusters=2 ami=1.0000 ari=1.0000
# S3 !!!n_neighbors!!! 53.000 n_clusters=15 ami=0.7372 ari=0.8029

# five_cluster 0.9~2 n_clusters=5 ami=0.9930 ari=0.9829
# four_cluster 5~40 n_clusters=4 ami=0.9947 ari=0.9915
# Spiral2 5.00 n_clusters=2 ami=1.0000 ari=1.0000
# three_cluster 0.01~30 n_clusters=3 ami=1.0000 ari=1.0000
# ThreeCircles 10.00~11 n_clusters=3 ami=1.0000 ari=1.0000
# two_cluster 0.01~30 n_clusters=2 ami=1.0000 ari=1.0000
# Twomoons 15.00 n_clusters=3 ami=1.0000 ari=1.0000

# circle 0.01~0.02 n_clusters=3 ami=0.9961 ari=0.9940
# cth 0.01~0.1 n_clusters=4 ami=1.0000 ari=1.0000
# db 0.05 n_clusters=4 ami=1.0000 ari=1.0000
# db3 0.1 n_clusters=4 ami=0.7861 ari=0.8494
# E6 5.00 n_clusters=7 ami=0.9996 ari=0.9994
# fc1 0.01 n_clusters=5 ami=1.0000 ari=1.0000
# line 0.003~0.05 n_clusters=4 ami=0.9956 ari=0.9945
# ls 0.01~0.1 n_clusters=6 ami=0.9990 ari=0.9985
# sk 0.01~0.1 n_clusters=3 ami=1.0000 ari=1.0000
# sn 0.01~0.1 n_clusters=5 ami=1.0000 ari=1.0000
# Spiral3 0.5~10 n_clusters=3 ami=1.0000 ari=1.0000

# Jain 0.5~1 n_clusters=2 ami=1.0000 ari=1.0000