import numpy as np
import pickle as pkl
import networkx as nx
import scipy.sparse as sp
import random as rn
from scipy.sparse.linalg.eigen.arpack import eigsh
import sys
import os

def one_hot(num_vecs, num_classes):
    ret_init =  np.random.rand(num_vecs,num_classes)
    ret = np.zeros_like(ret_init)
    ret[np.arange(len(ret_init)), ret_init.argmax(1)] = 1.
    return ret

def save(elems,test_index,dataset_str):
    with open("gcn/data/ind.{}.test.index".format(dataset_str),mode='w') as file:
        file.write('\n'.join(str(x) for x in test_index))

    names = ['x', 'y', 'tx', 'ty', 'allx', 'ally', 'graph']
    if len(elems) != len(names):
        raise ValueError("Must provide structure like [x, y, tx, ty, allx, ally, graph]")

    for i in range(len(names)):
        with open("gcn/data/ind.{}.{}".format(dataset_str, names[i]), 'wb') as f:
            pkl.dump(elems[i],f)

def data_gen(filename=None,num_x=100, num_tx=1000,num_features=2000, feature_probability = 0.1, num_classes=5):
    num_allx = num_x+num_tx

    graph = nx.fast_gnp_random_graph(num_allx,min(0.1,1/num_allx))
    if not nx.is_connected(graph):
        last = None
        for component in nx.connected_components(graph):
            if last is not None:
                graph.add_edge(last,rn.sample(component,1)[0])

            last = rn.sample(component,1)[0]

    test_index = list(range(num_x,num_allx))

    graph = nx.convert.to_dict_of_lists(graph)

    x = sp.rand(num_x,num_features,density=feature_probability,format='csr').astype(bool).astype(float)
    tx = sp.rand(num_tx,num_features,density=feature_probability,format='csr').astype(bool).astype(float)
    allx = x

    y = one_hot(num_x,num_classes)
    ty = one_hot(num_tx,num_classes)
    ally = y

    if None != filename:
        save([x,y,tx,ty,allx,ally,graph],test_index,filename)


data_gen("random_100_1000_2000_0p1_5")
# d = DataGen()