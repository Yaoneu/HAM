from utils import *
import time
import math
import run_AMN


def get_emb_AMN(graphfile1, graphfile2, anchorfile, attributefile,
                training_ratio, num_GAT_blocks, d=100, K=10, epoch=500,
                directed1=True, directed2=True, w1=1., w2=0.1):
    t1 = time.time()
    # loading information from datasets
    nodedict1, edge1 = read_edge(graphfile1, directed1)
    nodedict2, edge2 = read_edge(graphfile2, directed2)
    anchor = read_anchor(nodedict1, nodedict2, anchorfile)
    attr_matrix1, attr_matrix2, attributed = read_attr(nodedict1, nodedict2, attributefile)
    node_size1 = len(nodedict1.keys())
    node_size2 = len(nodedict2.keys())
    print('SN1 node number:', node_size1, 'SN2 node number:', node_size2)

    # preprocess
    print('negative sampling......')
    sampling_table1, sampling_table2 = gen_sampling_table(edge1, edge2, node_size1, node_size2)
    train_list, test_list = anchor_split(anchor, training_ratio)

    if not attributed:
        attr_matrix1 = np.ones([node_size1, 1], dtype=np.float32)
        attr_matrix2 = np.ones([node_size2, 1], dtype=np.float32)

    # network fusion component proposed in AMN
    edge, vec1, vec2 = edge_augmentation_AMN(edge1, edge2, train_list, attr_matrix1, attr_matrix2)
    vec_dim = vec1.shape[1]
    vec = np.concatenate([vec1, vec2], axis=0)

    del edge1, edge2, vec1, vec2

    # use AMN model to generate node representations
    sn1_emb, sn2_emb = run_AMN.run_AMN(edge, vec, vec_dim, d, epoch, K, num_GAT_blocks,
                                       sampling_table1, sampling_table2,
                                       node_size1, node_size2, test_list,
                                       w1=w1, w2=w2)

    print('running time is', time.time() - t1)
    return sn1_emb, sn2_emb


def gen_sampling_table(edge1, edge2, numNodes1, numNodes2):
    # negative sampling used in structure learning
    t = time.time()
    table_size = 1e8
    power = 0.75

    edgedict1 = edge_to_dict(edge1)
    node_degree1 = np.zeros(numNodes1)  # out degree

    for key in edgedict1.keys():
        node_degree1[key] = len(edgedict1[key])

    norm1 = sum([math.pow(node_degree1[i], power) for i in range(numNodes1)])

    sampling_table1 = np.zeros(int(table_size), dtype=np.uint32)

    p = 0
    i = 0
    for j in range(numNodes1):
        p += float(math.pow(node_degree1[j], power)) / norm1
        while i < table_size and float(i) / table_size < p:
            sampling_table1[i] = j
            i += 1

    edgedict2 = edge_to_dict(edge2)
    node_degree2 = np.zeros(numNodes2)
    for key in edgedict2.keys():
        node_degree2[key] = len(edgedict2[key])

    norm2 = sum([math.pow(node_degree2[i], power) for i in range(numNodes2)])

    sampling_table2 = np.zeros(int(table_size), dtype=np.uint32)

    p = 0
    i = 0
    for j in range(numNodes2):
        p += float(math.pow(node_degree2[j], power)) / norm2
        while i < table_size and float(i) / table_size < p:
            sampling_table2[i] = j
            i += 1

    print('negative sampling finished, time=', time.time() - t)
    return sampling_table1.tolist(), sampling_table2.tolist()


if __name__ == '__main__':
    # datasets
    graphfile1 = 'data/douban/douban1.txt'
    graphfile2 = 'data/douban/douban2.txt'
    anchorfile = 'data/douban/douban-anchor.txt'
    attributefile = ['data/douban/douban-attr1.txt',
                     'data/douban/douban-attr2.txt']

    d = 100  # the dimension of node embeddings
    tr = 0.1  # training ratio, i.e., the proportion of anchor nodes used in training process
    K = 10  # the number of negative samples
    epoch = 300  # training epochs
    num_GAT_blocks = 2  # the number of GAT layers
    w1 = 0.9  # structure loss weight
    w2 = 1. - w1  # attribute loss weight
    print('training ratio=', tr, 'embedding dim=', d, 'w1=', w1)
    get_emb_AMN(graphfile1, graphfile2, anchorfile, attributefile, tr, num_GAT_blocks=num_GAT_blocks,
                d=d, K=K, epoch=epoch, directed1=False, directed2=False, w1=w1, w2=w2)