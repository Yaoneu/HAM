import random
import numpy as np
from collections import defaultdict
import torch.nn.functional as F
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.neighbors import KDTree
import torch


def anchor_split(anchor_list, train_ratio):
    train_list = random.sample(anchor_list, int(train_ratio * len(anchor_list)))
    test_list = []
    for an in anchor_list:
        if an not in train_list:
            test_list.append(an)

    return train_list, test_list

def read_edge(graphfile, directed):
    edge = []
    nodedict = {}
    i = 0
    f = open(graphfile, 'r', encoding='utf-8')
    while 1:
        l=f.readline().rstrip('\n')
        if l=='':
            break
        src, dst = l.split()
        if src not in nodedict.keys():
            nodedict[src] = i
            i = i + 1
        if dst not in nodedict.keys():
            nodedict[dst] = i
            i = i + 1
        if directed:
            edge.append([nodedict[src], nodedict[dst]])
        else:
            edge.append([nodedict[src], nodedict[dst]])
            edge.append([nodedict[dst], nodedict[src]])

    return nodedict, edge


def read_attr(nodedict1, nodedict2, attributefile):
    if attributefile != []:
        attributed = True
        attr1 = {}
        attr2 = {}
        f = open(attributefile[0], 'r', encoding='utf-8')
        while 1:
            l = f.readline().rstrip('\n')
            if l == '':
                break
            node, vec = l.split()
            vec = list(map(float, vec.split(',')))
            attr1[node] = vec
        f.close()

        f = open(attributefile[1], 'r', encoding='utf-8')
        while 1:
            l = f.readline().rstrip('\n')
            if l == '':
                break
            node, vec = l.split()
            vec = list(map(float, vec.split(',')))
            attr2[node] = vec
        f.close()

        attr_matrix1 = [[]] * len(nodedict1.keys())
        attr_matrix2 = [[]] * len(nodedict2.keys())

        for key in nodedict1.keys():
            i = nodedict1[key]
            attr_matrix1[i] = attr1[key]
        for key in nodedict2.keys():
            i = nodedict2[key]
            attr_matrix2[i] = attr2[key]
        I1 = np.array(attr_matrix1, dtype=np.float32)
        I2 = np.array(attr_matrix2, dtype=np.float32)
    else:
        I1 = []
        I2 = []
        attributed = False
    return I1, I2, attributed


def read_anchor(nodedict1, nodedict2, anchorfile):
    anchor = []
    f = open(anchorfile, 'r', encoding='utf-8')
    while 1:
        l = f.readline().rstrip('\n')
        if l == '':
            break
        src, dst = l.split()
        i = nodedict1[src]
        j = nodedict2[dst]
        anchor.append([i, j])
    return anchor


def edge_to_dict(edgelist):
    edge_dict = defaultdict(list)
    for edge in edgelist:
        edge_dict[edge[0]].append(edge[1])

    return edge_dict

def edge_augmentation_AMN(edge1, edge2, train_list, attr_matrix1, attr_matrix2):
    nodesize1 = len(attr_matrix1)
    attr_dim = len(attr_matrix1[0])
    anchor_matrix1 = np.zeros([len(attr_matrix1), len(train_list)], dtype=np.float32)
    anchor_matrix2 = np.zeros([len(attr_matrix2), len(train_list)], dtype=np.float32)
    vec1 = np.concatenate((attr_matrix1, anchor_matrix1), axis=1)
    vec2 = np.concatenate((attr_matrix2, anchor_matrix2), axis=1)
    new_edge = []
    an1 = [x[0] for x in train_list]
    an2 = [x[1] for x in train_list]

    for e in edge1:
        new_edge.append(e)
        if e[0] in an1:
            index1 = an1.index(e[0])
            new_edge.append([an2[index1] + nodesize1, e[1]])
            if vec1[e[0]][attr_dim + index1] == 0:
                vec1[e[0]][attr_dim + index1] = 1
            else:
                vec1[e[0]][attr_dim + index1] = 2
        if e[1] in an1:
            index1 = an1.index(e[1])
            new_edge.append([e[0], an2[index1] + nodesize1])
            if vec1[e[1]][attr_dim + index1] == 0:
                vec1[e[1]][attr_dim + index1] = 1
            else:
                vec1[e[1]][attr_dim + index1] = 2

    for e in edge2:
        new_edge.append([e[0] + nodesize1, e[1] + nodesize1])
        if e[0] in an2:
            index1 = an2.index(e[0])
            new_edge.append([an1[index1], e[1] + nodesize1])
            if vec2[e[0]][attr_dim + index1] == 0:
                vec2[e[0]][attr_dim + index1] = 1
            else:
                vec2[e[0]][attr_dim + index1] = 2
        if e[1] in an2:
            index1 = an2.index(e[1])
            new_edge.append([e[0] + nodesize1, an1[index1]])
            if vec2[e[1]][attr_dim + index1] == 0:
                vec2[e[1]][attr_dim + index1] = 1
            else:
                vec2[e[1]][attr_dim + index1] = 2

    print('ori length=', len(new_edge))
    new_edge = list(set([tuple(t) for t in new_edge]))
    print('now', len(new_edge))

    return new_edge, vec1, vec2


def evaluate(alignment_matrix, true_matrix, k):
    nodenum1 = true_matrix.shape[0]
    # nodenum2=true_matrix.shape[1]
    pre = [0] * k

    for i in range(nodenum1):
        if np.sum(true_matrix[i]) > 0:
            true_node = np.argwhere(true_matrix[i] == 1)[0][0]
            sort1 = np.argsort(-alignment_matrix[i])
            for j in range(len(sort1)):
                if sort1[j] == true_node and j < k:
                    for p in range(j, k):
                        pre[p] = pre[p] + 1
            sort2 = np.argsort(-alignment_matrix.T[true_node])
            for j in range(len(sort2)):
                if sort2[j] == i and j < k:
                    for p in range(j, k):
                        pre[p] = pre[p] + 1

    pre_n = np.array(pre) / (2 * np.sum(true_matrix))
    return pre_n


def test_kdtree(model, test_list, vec, edge, nodesize1, nodesize2, k=30):
    model.eval()
    with torch.no_grad():
        v, _, _ = model(vec, edge)
        v1, v2 = v.split([nodesize1, nodesize2], dim=0)
        v1 = F.normalize(v1, p=2, dim=1, eps=1e-12, out=None)
        v2 = F.normalize(v2, p=2, dim=1, eps=1e-12, out=None)

        an1 = [x[0] for x in test_list]
        an2 = [x[1] for x in test_list]
        emb1 = v1[an1].cpu().detach().numpy()
        emb2 = v2[an2].cpu().detach().numpy()
        v1 = v1.cpu().detach().numpy()
        v2 = v2.cpu().detach().numpy()

        kd_tree1 = KDTree(v2, metric="euclidean")
        kd_tree2 = KDTree(v1, metric="euclidean")

        ind1 = kd_tree1.query(emb1, k=k, return_distance=False)
        ind2 = kd_tree2.query(emb2, k=k, return_distance=False)

        pre = [0] * k
        for i in range(len(test_list)):
            # sn1 node candidates
            candidates = ind1[i]
            for j in range(len(candidates)):
                if candidates[j] == an2[i] and j <= k:
                    for p in range(j, k):
                        pre[p] = pre[p] + 1
            # sn2 node candidates
            candidates = ind2[i]
            for j in range(len(candidates)):
                if candidates[j] == an1[i] and j <= k:
                    for p in range(j, k):
                        pre[p] = pre[p] + 1
    pre = np.array(pre) / (2 * len(test_list))
    return v1, v2, pre

