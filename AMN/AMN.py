import torch
import torch.nn as nn
from torch_geometric.nn import MessagePassing
from torch_geometric.nn import GATConv


class AMN(MessagePassing):
    def __init__(self, in_dim, out_dim, num_GAT_blocks):
        super(AMN, self).__init__(aggr='add')
        self.num_GAT_blocks = num_GAT_blocks
        self.GATs = []
        for i in range(num_GAT_blocks):
            self.GATs.append(GATConv(out_dim, out_dim, bias=False))
        self.GATs = nn.ModuleList(self.GATs)

        # attention
        self.layer_weight = torch.empty(in_dim, out_dim)
        torch.nn.init.xavier_uniform_(self.layer_weight)
        self.layer_weight = nn.Parameter(self.layer_weight, requires_grad=True)

        self.attention = torch.empty(1, in_dim + out_dim)
        torch.nn.init.xavier_uniform_(self.attention)
        self.attention = nn.Parameter(self.attention, requires_grad=True)
        self.linear = nn.Linear(in_dim + out_dim, out_dim, bias=False)

        # reconstruction
        self.recon_layer = nn.Linear(out_dim, in_dim, bias=True)

        # content vectors
        self.con_layer = nn.Linear(out_dim, out_dim, bias=True)

        self.reset_parameters()

    def reset_parameters(self):
        self.linear.reset_parameters()
        for i in range(self.num_GAT_blocks):
            self.GATs[i].reset_parameters()
        self.recon_layer.reset_parameters()
        self.con_layer.reset_parameters()

    def forward(self, x, edge):
        relu = nn.ReLU()
        tanh = nn.Tanh()
        # x is a vertical stack of vec1 and vec2, edge = edge1 + edge2 (edge2 are re-numbered)

        u0 = torch.matmul(x, self.layer_weight)
        outputs = [u0]
        emb_input = u0
        for i in range(self.num_GAT_blocks):
            GAT_output_i = tanh(self.GATs[i](emb_input, edge))
            outputs.append(GAT_output_i)
            emb_input = GAT_output_i

        w = []
        for i in range(len(outputs)):
            w_i = torch.exp(relu(self.linear(self.attention * torch.cat((x, outputs[i]), dim=1))))
            w.append(w_i)
            if i == 0:
                sum_w = w_i
            else:
                sum_w = sum_w + w_i

        for i in range(len(outputs)):
            w[i] = w[i] / sum_w
            if i == 0:
                u = w[i] * outputs[i]
            else:
                u = u + w[i] * outputs[i]

        u_recon = self.recon_layer(u)
        u_context = relu(self.con_layer(u))

        return u, u_recon, u_context

    def stru_loss(self, emb1, emb2, emb2_context, sign):
        logsigmoid = nn.LogSigmoid()
        loss1 = -torch.mean(logsigmoid(torch.mul(
            sign, torch.sum(torch.mul(emb1, emb2), dim=1))))

        loss2 = -torch.mean(logsigmoid(torch.mul(
            sign, torch.sum(torch.mul(emb1, emb2_context), dim=1))))

        return loss1 + loss2

    def emb_loss(self, emb, emb_neighbor):
        align_emb_loss = torch.mean(torch.norm(emb - emb_neighbor, p=2, dim=1))

        return align_emb_loss
