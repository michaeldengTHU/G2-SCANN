#! /usr/bin/env python
# coding=utf-8
#================================================================
#
#   Editor      : PyCharm
#   File name   : DBSCAN.py
#   Author      : Jingyi Wang
#   Created date: 2020/8/20 10:52
#   Description : 
#
#================================================================
from sklearn.cluster import DBSCAN
import numpy as np
from sklearn.metrics import adjusted_mutual_info_score, adjusted_rand_score
import pandas as pd
import os

#参数 https://www.cnblogs.com/pinard/p/6217852.html
def dbscan(dataset, gb0, e, m):
    data = np.loadtxt('newdataset/dataset/' + dataset + '_new.txt')[:, :-1]
    gt = np.loadtxt('newdataset/dataset/' + dataset + '_new.txt')[:, -1]
    similarity=gb0

    # db=DBSCAN(eps=e,min_samples=m).fit(data)
    db=DBSCAN(metric='precomputed',eps=e,min_samples=m).fit(similarity)
    # n_clusters = len(db.core_sample_indices_)
    pred = db.labels_ + 1
    lspred = list(set(pred))
    if 0 in lspred:
        n_clusters = len(lspred) - 1
    else:
        n_clusters = len(lspred)
    # np.savetxt('pred/DBSCAN/' + dataset + '_pred.txt', pred)
    np.savetxt('newdataset/DBSCAN/' + dataset + '_pred.txt', pred)

    ari = adjusted_rand_score(gt, pred)
    ami = adjusted_mutual_info_score(gt, pred)
    return n_clusters,ari,ami

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

    # es=[0.01,0.05,0.1,0.2,0.5,1,2,4,8,16,24]
    # ms=[1,2,4,8,16]
    es=[4]
    ms=[16]
    results=[]
    bestrm=-1
    for e in es:
        for m in ms:
            rm_all,amiari=[],[]
            for i in range(len(datasets)):
                dataset = datasets[i]
                gb0=np.loadtxt('newdataset/dataset/'+dataset+'.gb0')
                n_clusters,ari,ami=dbscan(dataset,gb0,e,m)
                print(dataset.center(15, ' ') + ' n_clusters=%d ami=%.4f ari=%.4f' % (n_clusters, ari, ami))
                rm_all.append((ari + ami) / 2)
                amiari.append([ami,ari])
                results.append([dataset,e,m,n_clusters,ami,ari])
            '''
                for j in range(len(datasets[i])):
                    dataset = datasets[i][j]
                    if i!=3:
                        gb_path = '/data/wangjingyi/G2-DBCANN/dataset' + str(i + 1) + '/' + dataset + '.gb0'
                    else:
                        gb_path = '/data/wangjingyi/G2-DBCANN/dataset' + str(i) + '/' + dataset + '.gbd'
                    gb = np.loadtxt(gb_path)
                    n_clusters,ari,ami=dbscan(dataset, gb,e,m)
                    print(dataset.center(15, ' ') + ' n_clusters=%d ami=%.4f ari=%.4f' % (n_clusters, ari, ami))
                    rm_all.append((ari + ami) / 2)
            '''
            rm = sum(rm_all) / len(rm_all)
            if rm>bestrm:
                bestrm=rm
                bestem=(e,m)
            print(amiari)
            amiari=np.array(amiari)
            print('ami')
            for aaaaa in range(amiari.shape[0]):
                print(amiari[aaaaa][0])
            print('ari')
            for aaaaa in range(amiari.shape[0]):
                print(amiari[aaaaa][1])
            np.savetxt('newdataset/DBSCAN/amiari.txt',np.array(amiari))
            print('ami_ave=%.4f'%np.mean(amiari[:,0]),'  ari_ave=%.4f'%np.mean(amiari[:,1]))
            print('****** e',e,'  m',m,'  rm',rm,'\n')
    print('bestrm',bestrm,'  bestem',bestem)
    '''
    csv_path='newdataset/DBSCAN/para_gb0.csv'
    if os.path.exists(csv_path):
        df=pd.read_csv(csv_path,index_col=0)
        newdf=pd.DataFrame(np.array(results),columns=['dataset','eps','min_samples','n_clusters','ami','ari'])
        df=df.append(newdf,ignore_index=True)
        df.to_csv(csv_path)
    else:
        df=pd.DataFrame(np.array(results),columns=['dataset','eps','min_samples','n_clusters','ami','ari'])
        df.to_csv(csv_path)
    '''
# A 0.75 700 n_clusters=106 ami=0.3027 ari=0.5110
# Aggregation 1.2 14 n_clusters=6 ami=0.0434 ari=0.1606

# linux:
# B 0.04 22 n_clusters=5 ami=0.9743 ari=0.9497
# E6 2 8 n_clusters=7 ami=0.9940 ari=0.9900
# Spiral3 2 1 n_clusters=3 ami=1.0000 ari=1.0000
# S3 30000 32 n_clusters=15 ami=0.3567 ari=0.6404
