#! /usr/bin/env python
# coding=utf-8
#================================================================
#
#   Editor      : PyCharm
#   File name   : trytry.py
#   Author      : Jingyi Wang
#   Created date: 2020/8/20 14:58
#   Description : 
#
#================================================================
import re
import json
import numpy as np
import fileinput
from heapq import *
import heapq
import sys
from scipy.special import comb, perm
import itertools
import pandas as pd
import os

def mydijkstra_old(G, S):
    '''
    该有向图共有N个顶点，顶点的索引是0、1、2、N-1。权重都是整数。在图中共有M条辺。
    具体使用的命令行为 python Dijkstra.py <inputfile> outputfile
    inputfile：共有M+1行，每个数字都是正整数用空格隔开
        第一行：定点数、边数、开始节点
        其他行 图上一条边的开始节点索引、图上一条边的结束节点索引、该边的权重
    outputfile：每一行为从开始节点到能到达的所有节点的相关内容，每个数字都是正整数并用空格隔开
        节点索引、从开始节点到该节点的最短路径、从开始节点到该节点的路径树上的父节点索引
    '''

    '''
    line_index = 0
    for line in fileinput.input():
        if line_index == 0:
            node_num,edge_num,start = int(line.split(' ')[0]),int(line.split(' ')[1]),int(line.split(' ')[2])
            edge = [[sys.maxsize for j in range(node_num)] for i in range(node_num)] # 链接矩阵
            # 每一行是以该行节点为头，每一列是以该列为尾
        else:
            starting,ending,weight = int(line.split(' ')[0]),int(line.split(' ')[1]),int(line.split(' ')[2])
            edge[starting][ending] = weight    
        line_index +=1
    '''
    edge = G
    start = S
    node_num = edge.shape[0]

    parent = [start] * node_num  # 每个节点的父节点
    visit = [False] * node_num  # 每个节点是否被访问过
    visit[start] = True
    pre_node_index = start
    only_distance = edge[start]

    distance = [[only_distance[i], i] for i in range(len(only_distance))]
    heapify(distance)
    while len(distance) > 0:
        min_distance, min_distance_node_index = heapq.heappop(distance)
        visit[min_distance_node_index] = True
        for i in range(node_num):
            if visit[i]: continue
            temp = min_distance + edge[min_distance_node_index][i]
            if only_distance[i] > temp:
                only_distance[i] = temp
                parent[i] = min_distance_node_index
                distance.append([only_distance[i], i])
                print('new edge', min_distance_node_index, i)
        heapify(distance)

    for i in range(len(only_distance)):
        if only_distance[i] != sys.maxsize:
            print(i, only_distance[i], parent[i])


def mydijkstra(G, S):
    edge = G
    start = S
    node_num = edge.shape[0]

    parent = [start] * node_num  # 每个节点的父节点
    visit = [False] * node_num  # 每个节点是否被访问过
    visit[start] = True
    pre_node_index = start
    only_distance = edge[start]

    distance = [[only_distance[i], i] for i in range(len(only_distance))]
    heapify(distance)
    while len(distance) > 0:
        min_distance, min_distance_node_index = heapq.heappop(distance)
        visit[min_distance_node_index] = True
        for i in range(node_num):
            if visit[i]: continue
            temp = min_distance + edge[min_distance_node_index][i]
            if only_distance[i] > temp:
                only_distance[i] = temp
                parent[i] = min_distance_node_index
                distance.append([only_distance[i], i])
        heapify(distance)

    # for i in range(len(only_distance)):
        # if only_distance[i] != 999999:
        # print(i,only_distance[i],parent[i])
    return only_distance, parent


def reorder():
    datasets = [['A', 'Aggregation', 'B', 'Flame', 'S3']
        , ['five_cluster', 'four_cluster', 'Spiral2', 'three_cluster', 'ThreeCircles', 'two_cluster', 'Twomoons']
        , ['circle', 'cth', 'db', 'db3', 'E6', 'fc1', 'line', 'ls', 'sk', 'sn', 'Spiral3']
        , ['Jain']]
    gt_n_clusters = [[5, 7, 5, 2, 15]
        , [5, 4, 2, 3, 3, 2, 2]
        , [3, 4, 4, 4, 7, 5, 4, 6, 3, 5, 3]
        , [2]]

    o1=[1,1,1,1,2,2,2,2,3,3,2,2,3,1,3,4,2,3,3,3,3,3,3,3]
    o2=[1,3,2,5,6,4,2,1,10,6,3,7,11,4,5,1,5,9,7,1,3,4,8,2]
    for i in range(len(o1)):
        dataset=datasets[o1[i]-1][o2[i]-1]
        print(dataset)


def forSNNDPClog():
    lines=open('SNNDPC_code/SNNDPC_newdataset_para.log','r').readlines()
    lines_k=[]
    d={}
    for line in lines:
        if re.search("k=",line):
            lines_k.append(line)
    for line in lines_k:
        line_split=line.strip().split(' ')
        dataset=line_split[0]
        if dataset=="waveform" or dataset=="waveform_noise":
            continue
        k=int(line_split[-4].split('=')[-1])
        ami=float(line_split[-2].split('=')[-1])
        ari=float(line_split[-1].split('=')[-1])
        if k in d.keys():
            d[k][dataset]={}
            d[k][dataset]['ami']=ami
            d[k][dataset]['ari']=ari
        else:
            d[k]={}
            d[k][dataset]={}
            d[k][dataset]['ami']=ami
            d[k][dataset]['ari']=ari
    json_str = json.dumps(d)
    with open('SNNDPC_code/SNNDPC_newdataset_para.json', 'w') as json_file:
        json_file.write(json_str)
    
    best_k=0
    best_rm=0
    for k in d.keys():
        rm_all=[]
        for dataset in d[k].keys():
            # print(dataset,d[k][dataset])
            rm_all.append((d[k][dataset]['ami']+d[k][dataset]['ami'])/2)
        rm_mean=np.mean(rm_all)
        print('k=%d'%k, 'rm_mean=%.4f'%rm_mean)
        if rm_mean>best_rm:
            best_rm=rm_mean
            best_k=k
    print('best_k=%d'%best_k, 'best_rm=%.4f'%best_rm)
    '''
    k=3 rm_mean=0.1357
    k=5 rm_mean=0.1970
    k=7 rm_mean=0.1757
    k=10 rm_mean=0.2231
    k=12 rm_mean=0.2211
    k=15 rm_mean=0.2414
    k=17 rm_mean=0.2340
    k=20 rm_mean=0.2503
    k=22 rm_mean=0.1565
    k=25 rm_mean=0.1437
    k=27 rm_mean=0.1427
    k=30 rm_mean=0.2387
    k=33 rm_mean=0.2672
    k=35 rm_mean=0.2320
    k=40 rm_mean=0.2084
    k=45 rm_mean=0.2277
    k=50 rm_mean=0.1687
    k=60 rm_mean=0.1844
    best_k=33 best_rm=0.2672
    '''
    print('forSNNDPClog done')


def comb():
    from scipy.special import comb, perm
    import itertools
    import numpy as np
    total=int(perm(9,4)+perm(9,5)+perm(9,6)+perm(9,7)+perm(9,8)+perm(9,9))
    print('total',total)
    wrong_comb=[[1,4,7],[2,5,8],[3,6,9],[1,2,3],[4,5,6],[7,8,9],[1,5,9],[3,5,7]]
    for i in range(8):
        wrong_comb.append(wrong_comb[i][::-1])

    def iswrong(l):
        for i in range(len(l)-1):
            li,li1=l[i],l[i+1]
            for j in range(len(wrong_comb)):
                if li==wrong_comb[j][0] and li1==wrong_comb[j][2] and not wrong_comb[j][1] in l[:i]:
                    return True
        return False

    sum_wrong=0
    for i in range(4,9):
        wrong=0
        totali=list(itertools.permutations(np.arange(1,10).tolist(),i))
        for p in totali:
            if iswrong(list(p)):
                wrong+=1
        print('length=%d wrong=%d'%(i,wrong))
        sum_wrong+=wrong
        if i==8:
            sum_wrong+=wrong
    print('total-sum_wrong',total-sum_wrong)


def forg2mannlog():
    if not os.path.exists('OLD/G2-MANNv2_newdataset_para.csv'):
        lines=open('OLD/G2-MANNv2_newdataset_para_waveform.log','r').readlines()
        lines_k=[]
        d=[]
        for line in lines:
            if re.search("n_clusters=",line):
                lines_k.append(line)
        for line in lines_k:
            line_split=line.strip().split(' ')
            dataset=line_split[0]
            if dataset=="waveform" or dataset=="waveform_noise":
                continue
            ami=float(line_split[-2].split('=')[-1])
            ari=float(line_split[-1].split('=')[-1])
            d.append([dataset,ami,ari])
        df=pd.DataFrame(np.array(d),columns=['dataset','ami','ari'])
        df.to_csv('OLD/G2-MANNv2_newdataset_para.csv')
    else:
        df=pd.read_csv('OLD/G2-MANNv2_newdataset_para.csv',index_col=0)
        d={}
        for index, row in df.iterrows():
            gamma=row['wjy_gamma_para']
            dataset=row['dataset']
            ami=row['ami']
            ari=row['ari']
            if gamma in d.keys():
                d[gamma][dataset]={}
                d[gamma][dataset]['ami']=ami
                d[gamma][dataset]['ari']=ari
            else:
                d[gamma]={}
                d[gamma][dataset]={}
                d[gamma][dataset]['ami']=ami
                d[gamma][dataset]['ari']=ari
        json_str = json.dumps(d)
        with open('OLD/G2-MANNv2_newdataset_para.json', 'w') as json_file:
            json_file.write(json_str)
        
        best_gamma=0
        best_rm=0
        for gamma in d.keys():
            rm_all=[]
            for dataset in d[gamma].keys():
                rm_all.append((d[gamma][dataset]['ami']+d[gamma][dataset]['ami'])/2)
            rm_mean=np.mean(rm_all)
            print('gamma=%.2f'%gamma, 'rm_mean=%.4f'%rm_mean)
            if rm_mean>best_rm:
                best_rm=rm_mean
                best_gamma=gamma
        print('best_gamma=%.2f'%best_gamma, 'best_rm=%.4f'%best_rm)
    '''
    gamma=0.25 rm_mean=0.3724
    gamma=0.26 rm_mean=0.3745
    gamma=0.27 rm_mean=0.3726
    gamma=0.28 rm_mean=0.3724
    gamma=0.29 rm_mean=0.3739
    gamma=0.30 rm_mean=0.3737
    gamma=0.33 rm_mean=0.3538
    gamma=0.35 rm_mean=0.3540
    best_gamma=0.26 best_rm=0.3745
    '''
    print('forg2mannlog done')

if __name__ == '__main__':
    # m=999999
    # G=np.array(
    #     [[0,1,4,7],
    #      [1,0,2,m],
    #      [4,2,0,3],
    #      [7,m,3,0]
    #     ]
    # )
    # mydijkstra(G,0)

    # forSNNDPClog()
    forg2mannlog()

    # comb()
    