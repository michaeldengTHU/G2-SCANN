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
    gt = np.loadtxt('newdataset/dataset/' + dataset + '_new.txt')[:, -1]
    similarity = -gb0

    af = AffinityPropagation(affinity='precomputed', damping=d, preference=p)\
        .fit(similarity)
    n_clusters = len(af.cluster_centers_indices_)
    pred = af.labels_+1
    np.savetxt('newdataset/AP/' + dataset + '_pred.txt', pred)

    ari = adjusted_rand_score(gt, pred)
    ami = adjusted_mutual_info_score(gt, pred)
    return n_clusters, ari, ami

if __name__ == '__main__':
    datasets=['iris','wdbc','seeds','libras_movement','ionosphere','dermatology']
    gt_n_clusters=[3,2,3,15,2,6]
    print(datasets)
    print(gt_n_clusters)

    # ds=[0.7,0.8,0.9]
    # ps=[-1,-10,-50,-100,-500,-1000,-3000,-5000]
    ds=[0.68]
    ps=[-60]
    results=[]
    bestrm=-1
    for d in ds:
        for p in ps:
            rm_all,amiari=[],[]
            for i in range(len(datasets)):
                dataset=datasets[i]
                gb=np.loadtxt('newdataset/dataset/'+dataset+'.gb0')
                n_clusters, ari, ami = ap(dataset, gb, d, p)
                print(dataset.center(15, ' ') + ' n_clusters=%d ami=%.4f ari=%.4f' % (n_clusters, ami, ari))
                results.append([dataset,d,p,n_clusters,ami,ari])
                rm_all.append((ari + ami) / 2)
                amiari.append([ami,ari])
                '''
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
                '''
            rm = sum(rm_all) / len(rm_all)
            if rm>bestrm:
                bestrm=rm
                bestdp=(d,p)
            amiari=np.array(amiari)
            print('ami')
            for aaaaa in range(amiari.shape[0]):
                print(amiari[aaaaa][0])
            print('ari')
            for aaaaa in range(amiari.shape[0]):
                print(amiari[aaaaa][1])

            np.savetxt('newdataset/AP/amiari.txt',np.array(amiari))
            print('ami_ave=%.4f'%np.mean(amiari[:,0]),'  ari_ave=%.4f'%np.mean(amiari[:,1]))
            print('****** d=',d,'  p=',p,'  rm=',rm,'\n')
    print('bestrm',bestrm,'  bestdp',bestdp)
    '''
    csv_path='newdataset/AP/para.csv'
    if os.path.exists(csv_path):
        df=pd.read_csv(csv_path,index_col=0)
        newdf=pd.DataFrame(np.array(results),columns=['dataset','damping','preference','n_clusters','ami','ari'])
        df=df.append(newdf,ignore_index=True)
        df.to_csv(csv_path)
    else:
        df=pd.DataFrame(np.array(results),columns=['dataset','damping','preference','n_clusters','ami','ari'])
        df.to_csv(csv_path)
    '''
