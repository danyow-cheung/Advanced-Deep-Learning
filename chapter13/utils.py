import numpy as np 
from scipy.optimize import linear_sum_assignment

def unsupervised_labels(y,yp,n_classes,n_clusters):
    '''
    Linear assignment algorithm
    Arguments:
        y(tensor)           Ground truth labels
        yp(tensor)          Predicted clusters
        n_classes(int)      Number of classes
        n_clusters(int)     Number of clusters
    '''
    assert n_classes == n_clusters

    # initialize count matrix 
    c = np.zeros([n_clusters,n_classes])
    # populate count matrix 
    for i in range(len(y)):
        c[int(yp[i]),int(y[i])]+=1 
    # optimal permuation using Hungarian Algo
    # the higher the count,the lower the cost 
    # so we use -c for linear assignment 
    row,col = linear_sum_assignment(-c)
    # compute accuracy 
    accuracy = c[row,col].sum()/c.sum()
    return accuracy * 100 
