import sys
import numpy as np
import GCAFCN as fcn
from sklearn.metrics import adjusted_mutual_info_score, adjusted_rand_score
import argparse

def g2mann(dataset,nclust):
    if dataset!='Jain':
        gb_dist = np.loadtxt('../synthetic_dataset/' + dataset + '.gb0')
    else:
        gb_dist = np.loadtxt('../synthetic_dataset/' + dataset + '.gbd')

    data0 = np.loadtxt('../synthetic_dataset/'+dataset+'.txt')
    ND0 = data0.shape[0]                            # number of sample data
    DIM = data0.shape[1]                            # dimension of input sample vector
    if (DIM == 3):                                  # 2D sample vector + label
       data0 = np.delete(data0, 2, axis=1)          # have removal of label column
       DIM = DIM - 1
    
    classes0 = -2*np.ones((ND0, 1), dtype=np.int)   # classes for each sample data /default: -2

    # --------------------------------------------------------------------------
    # Use the DE-OUTLIER algorithm to remove/mark outlier points
    if dataset!='Jain':
        cl, type, no = fcn.deoutlier(data0, MinPts=nclust)
        cl = cl.transpose()
        #cl = classes0
        data = data0.copy()
        for i in range(ND0-1, -1, -1):
           if (cl[i] == -1):
               classes0[i] = cl[i]                   # this is a outlier point
               data = np.delete(data, i, axis=0)
        ND = data.shape[0]
    else:
        ND=ND0
        data=data0

    # --------------------------------------------------------------------------
    # Compute pairwise Euclidean distance matrix
    dist, pwd = fcn.dist_m(data)

    # Find the maximum number of nearest natural neighbors
    dynk, max_nnn = fcn.NaN(dist)        # find maximum number of natural neighbors (max_nnn = 100)

    # Find the SPL adjacency matrix that has nearest natural neighbors
    """
    weight = np.mat(np.zeros((ND, ND)))
    for i in range(ND):
       for j in np.argsort(dist[i, :]).tolist()[1:dynk+1]:   # j: index
           weight[i, j] = dist[i, j]                         # Only retain distance values of natural neighbors of this data point; otherwise all zeros
    
    gb_dist = fcn.graph_dist(data, weight)                   # Evaluation of graph-based SPL adjacency matrix
    np.savetxt('./'+fname+'.gb0', gb_dist)
    """

    # gb_dist = np.loadtxt('./'+fname+'.gb0')                  # gbd: without de-outlier; gb0: with dynamic natural neighbor
    print('Computation/load of graph-based SPL was done ...')

    # Find the G2-SPL adjacency matrix with epsilon-natural neighbors
    sigma = fcn.epsilon(data, nclust)     # an empirical formula used to find epsilon(-neighborhood)
    gb_rbf = np.mat(np.zeros((ND, ND)))
    for i in range(ND):
        for j in np.argsort(gb_dist[i, :]).tolist()[0:max_nnn]:
            if i in np.argsort(gb_dist[j, :]).tolist()[0:max_nnn]:
                gb_rbf[i, j] = np.exp(-np.power(gb_dist[i, j], 2) / (2 * (np.power(sigma, 2))))
    # -----------------------------------------------------------------------------

    # #############################################################################
    # SPL-weighted Local Degree (SLD)
    # #############################################################################

    # Compute graph-based local degree
    rho = fcn.graphdensity(gb_rbf)

    # Compute the SLD/gamma values
    gamma, ordrho, parent, nchild = fcn.gamma_find(rho, gb_dist)

    # Sort the gamma values in descending order (gamma_sorted) and save the index (ordgamma)
    gamma_sorted = np.array(sorted(gamma, reverse=True))
    ordgamma = np.lexsort((gamma_sorted, gamma))[::-1]

    # ---------------------------------------------------------------------------------------
    # G2-SCANN Clustering Algorithm
    nclustMin = nclustMax = nclust                          # given number of clusters
    twonclust = 2*nclust
    for nclust in range(nclustMin, nclustMax + 1):
        classes = -2*np.ones((ND, 1), dtype=np.int)         # classes for each data point without any outlier point
        classn8 = -2*np.ones((ND, 1), dtype=np.int)
        tree = -2*np.ones((twonclust, ND), dtype=np.int)    # node types of tree-like cluster

        
        # Remove the lone trees that belong to the 2*nclust largest gamma (SLD) values but contain
        # disproportionately few data points or vertices.
        # -considering the twonclust largest gamma values as cluster heads
        for i in range(twonclust):
            classn8[ordgamma[i]] = i + 1  # -2: not handled (default); 1, ..., twonclust: cluster heads; -8: lone-tree
        #print('A: gamma_sorted =', gamma_sorted[0:twonclust, ])
        # -propagating class messages from cluster heads to other nodes for each tree-like cluster
        for i in range(ND):
            if (classn8[ordrho[i]] == -2):  # data points that are not handled
                classn8[ordrho[i]] = classn8[parent[ordrho[i]]]
        # -finding the number of tree-like clusters
        for i in range(twonclust):
            cnt = 0
            for j in range(ND):
                if (classn8[ordrho[j]] == i + 1):
                    cnt += 1
            tree[i, 1] = cnt  # the number of nodes in each tree-like cluster
            #prod[i, ] = gamma_sorted[i, ]*cnt
        #print('B: the number of tree-like clusters =', ND, tree[:, 1].transpose())
        #print('C: ordgamma =', ordgamma[0:twonclust, ])
        # -justifying tree-like clusters containing disproportionately few data points as lone trees or isolated islands
        for i in range(twonclust):
            if (tree[i, 1] < 0.027*(ND/nclust)):   # 99.73% (3sigma) 0.027
                classes[ordgamma[i]] = -1          # root vertex label of lone tree is assigned as outliers
                ordgamma[i] = -8                   # confirmed as lone trees
        for i in range(twonclust-1, -1, -1):
            if (ordgamma[i] == -8):
                ordgamma = np.delete(ordgamma, i, axis=0)   # delete ordgamma index of lone tree
                tree = np.delete(tree, i, axis=0)  # delete lone tree rows


        # Classify the nclust root vertices with the largest gamma values
        peak_root = np.zeros((nclust, DIM + 1))             # gamma peak #: 0, 1, 2, ..., nclust-1
        for i in range(nclust):
            classes[ordgamma[i]] = i + 1                    # 1, 2, ..., nclust; -1 outlier; -2 others
            tree[i, 0] = ordgamma[i]                        # gamma peak indices for each tree-like cluster
            for j in range(DIM):
                peak_root[i, j] = data[ordgamma[i], j]
            peak_root[i, DIM] = ordgamma[i]

        # Classify the other data points
        # print (ND, ND0, len(ordrho), len(parent), len(classes))
        for i in range(ND):
            if (classes[ordrho[i]] == -2):  # data points that are not handled
                classes[ordrho[i]] = classes[parent[ordrho[i]]]  # class message propagation from gamma peaks

        bestcl = classes

    classes = bestcl.copy()
    nclust = int(np.max(bestcl[:, 0]))
    cnt = 0
    # print(dataset,ND0,ND,len(classes))

    if dataset != 'Jain':
        for i in range(ND0):
           if (classes0[i] != -1):
               classes0[i] = classes[cnt]
               cnt += 1

    return classes, nclust

if __name__ == '__main__':
    all_datasets = ['A', 'Aggregation', 'B', 'Flame', 'S3'
                ,'five_cluster', 'four_cluster', 'Spiral2', 'three_cluster', 'ThreeCircles', 'two_cluster', 'Twomoons'
                ,'circle', 'cth', 'db4', 'db3', 'E6', 'fc1', 'line', 'ls', 'sk', 'sn', 'Spiral3'
                ,'Jain']
    all_gt_n_clusters = [5, 7, 5, 2, 15
                    ,5, 4, 2, 3, 3, 2, 2
                    ,3, 4, 4, 4, 7, 5, 4, 6, 3, 5, 3
                    ,2]

    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset',type=str,default='A',help='input the dataset name')
    args = parser.parse_args()
    datasets=[args.dataset]
    try:
        gt_n_clusters=[all_gt_n_clusters[all_datasets.index(args.dataset)]]
    except Exception as e:
        print('There is no dataset named',args.dataset)
        sys.exit(0)

    amiari=[]
    for i in range(len(datasets)):
        dataset = datasets[i]
        if i != 3:
            pred,n_cluster=g2mann(dataset,gt_n_clusters[i])
        else:
            pred,n_cluster=g2mann(dataset,gt_n_clusters[i])
        if dataset!='face':
            gt=np.loadtxt('../synthetic_dataset/' + dataset + '_new.txt')[:,-1]
        else:
            gt=[]
            for i in range(1,41):
                gt+=[i]*10
        pred=pred.reshape((-1))
        ari = adjusted_rand_score(gt, pred)
        ami = adjusted_mutual_info_score(gt, pred)
        print(dataset.center(15, ' ')  + ' n_clusters=%d ami=%.4f ari=%.4f' % (n_cluster, ami, ari))
        amiari.append([ami,ari])
    # np.savetxt('pred/G2-MANN/amiari.txt',np.array(amiari))
    