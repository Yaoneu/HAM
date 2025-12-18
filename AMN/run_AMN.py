from utils import *
import time
from torch.utils.data import DataLoader
import AMN


def run_AMN(edge, vec, vec_dim, d, epoch, K, num_GAT_blocks, sampling_table1, sampling_table2,
            nodesize1, nodesize2, test_list, batch_size=100, w1=1., w2=0.1):
    device = 'cpu'
    torch.manual_seed(123)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(123)
        device = 'cuda:%d' % 0

    best_pre = []
    max_pre_30, max_epoch = 0, 0
    t = time.time()
    # AMN model
    model = AMN.AMN(vec_dim, d, num_GAT_blocks).to(device)

    edge_dict = edge_to_dict(edge)
    dataset = torch.tensor([i for i in range(len(vec))]).to(device)
    edge = torch.tensor(edge).T.to(device)
    vec = torch.from_numpy(vec).to(torch.float32).to(device)

    data_loader = DataLoader(dataset=dataset, batch_size=batch_size, shuffle=True)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

    sn1_emb = []
    sn2_emb = []

    for e in range(epoch + 1):
        epoch_time = time.time()
        model.train()
        # total_loss = torch.FloatTensor(0).to(device)
        for i, data in enumerate(data_loader):
            emb, emb_recon, emb_context = model(vec, edge)

            # embedding look_up
            n1, n2, sign = embedding_look_up_stru(data, K, sampling_table1, nodesize1,
                                                  sampling_table2, device, edge_dict)
            stru_loss = model.stru_loss(emb[n1], emb[n2], emb_context[n2], sign)

            attr_loss = model.emb_loss(vec[data], emb_recon[data])
            total_loss = w1 * stru_loss + w2 * attr_loss

            optimizer.zero_grad()
            total_loss.backward()
            optimizer.step()

        if e % 50 == 0:
            print('epoch:{}, epoch time:{}'.format(e, time.time() - epoch_time))
            if e > 0:
                v1, v2, pre = test_kdtree(model, test_list, vec, edge, nodesize1, nodesize2)
                print("Epoch:{}, Test_Hits:{}, Time:{}".format(
                    e, pre, time.time() - t))
                if sum(pre) >= max_pre_30:
                    best_pre = pre
                    max_epoch = e
                    max_pre_30 = sum(pre)
                    sn1_emb = v1
                    sn2_emb = v2
    print('best hits:', best_pre, 'max epoch=', max_epoch)

    return sn1_emb, sn2_emb


def embedding_look_up_stru(data, negative_ratio, sampling_table1, nodesize1,
                           sampling_table2, device, edge_dict):
    # find negative samples in each batch
    data = data.tolist()
    n1 = []
    n2 = []
    sign = []
    for i in range(len(data)):
        node = data[i]
        neighbor = edge_dict[node]
        n1 = n1 + [node] * (len(neighbor) * (negative_ratio + 1))
        n2 = n2 + neighbor
        sign = sign + [1.] * len(neighbor)

        if node < nodesize1:
            num_intra = len(np.argwhere(np.array(neighbor) < nodesize1))
            num_inter = len(neighbor) - num_intra

            neg_intra = random.sample(sampling_table1, negative_ratio * num_intra)
            neg_inter = random.sample(sampling_table2, negative_ratio * num_inter)

            n2 = n2 + neg_intra + [x + nodesize1 for x in neg_inter]

        else:
            num_inter = len(np.argwhere(np.array(neighbor) < nodesize1))
            num_intra = len(neighbor) - num_inter

            neg_intra = random.sample(sampling_table2, negative_ratio * num_intra)
            neg_inter = random.sample(sampling_table1, negative_ratio * num_inter)

            n2 = n2 + [x + nodesize1 for x in neg_intra] + neg_inter

        sign = sign + [-1.] * (len(neg_intra) + len(neg_inter))

    n1 = torch.tensor(n1).to(device)
    n2 = torch.tensor(n2).to(device)
    sign = torch.tensor(sign).to(device)
    return n1, n2, sign
