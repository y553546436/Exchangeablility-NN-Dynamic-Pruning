import torch
import torch.nn.functional as F
from torch_geometric.nn import MessagePassing, global_add_pool, global_mean_pool, global_max_pool, GlobalAttention
from torch_geometric.utils import degree
from ogb.graphproppred.mol_encoder import AtomEncoder,BondEncoder
from config import DEVICE, early_terminate_it
from utils.confidences import StatsTestConfidence
from copy import deepcopy
from models.linear_module import MyLinear

class GCNConv(MessagePassing):
    def __init__(self, emb_dim):
        super(GCNConv, self).__init__(aggr='add')

        self.linear = torch.nn.Linear(emb_dim, emb_dim, bias=False)
        self.root_emb = torch.nn.Embedding(1, emb_dim)
        self.bond_encoder = BondEncoder(emb_dim = emb_dim)

    def forward(self, x, edge_index, edge_attr):
        x = self.linear(x)
        x = F.relu(x)
        edge_embedding = self.bond_encoder(edge_attr)

        row, col = edge_index

        deg = degree(row, x.size(0), dtype = x.dtype) + 1
        deg_inv_sqrt = deg.pow(-0.5)
        deg_inv_sqrt[deg_inv_sqrt == float('inf')] = 0

        norm = deg_inv_sqrt[row] * deg_inv_sqrt[col]

        return self.propagate(edge_index, x=x, edge_attr = edge_embedding, norm=norm) + (x + self.root_emb.weight) * 1./deg.view(-1,1)

    def my_eval(self, confidence):
        self.linear = MyLinear(self.linear, confidence)

    def gather_flops(self):
        return self.linear.gather_flops()

    def message(self, x_j, edge_attr, norm):
        return norm.view(-1, 1) * (x_j + edge_attr)

    def update(self, aggr_out):
        return aggr_out


class GNN_node(torch.nn.Module):
    """
    Output:
        node representations
    """
    def __init__(self, num_layer, emb_dim):
        '''
            emb_dim (int): node embedding dimensionality
            num_layer (int): number of GNN message passing layers

        '''

        super(GNN_node, self).__init__()
        self.num_layer = num_layer

        if self.num_layer < 2:
            raise ValueError("Number of GNN layers must be greater than 1.")

        self.atom_encoder = AtomEncoder(emb_dim)

        self.convs = torch.nn.ModuleList()
        self.batch_norms = torch.nn.ModuleList()

        for layer in range(num_layer):
            self.convs.append(GCNConv(emb_dim))
            self.batch_norms.append(torch.nn.BatchNorm1d(emb_dim))

    def forward(self, batched_data):
        x, edge_index, edge_attr = batched_data.x, batched_data.edge_index, batched_data.edge_attr
        h_list = [self.atom_encoder(x)]
        for layer in range(self.num_layer):
            h = self.convs[layer](h_list[layer], edge_index, edge_attr)
            h = self.batch_norms[layer](h)

            if layer < self.num_layer - 1:
                h = F.relu(h)
            
            h = F.dropout(h, 0.5, training = self.training)

            h_list.append(h)

        node_representation = h_list[-1]

        return node_representation

    def my_forward(self, batched_data):
        x, edge_index, edge_attr = batched_data.x, batched_data.edge_index, batched_data.edge_attr
        node_attr_tot = x.numel()
        h_list = [self.atom_encoder(x)]
        emb_dim = h_list[0].shape[1] # emb_dim not changed in the forward pass
        self.flops += node_attr_tot * emb_dim # atom encoder flops
        for layer in range(self.num_layer):
            h = self.convs[layer](h_list[layer], edge_index, edge_attr)
            self.flops += edge_attr.numel() * emb_dim # bond encoder flops
            self.flops += edge_index.numel() * emb_dim * 3 # message passing flops (norm, plus edge_attr, plus x_j per edge)
            h = self.batch_norms[layer](h)
            self.flops += h.numel() * 2 # batch norm flops
            if layer < self.num_layer-1:
                h = F.relu(h)
                self.flops += h.numel() # relu flops
            h_list.append(h)

        node_representation = h_list[-1]

        return node_representation

    def my_eval(self, confidences):
        self.eval()
        self.flops = 0
        assert len(confidences) == self.num_layer - 1
        confidences += [None]
        for layer in range(self.num_layer):
            self.convs[layer].my_eval(confidences[layer])
        self.forward = self.my_forward

    def gather_flops(self):
        return self.flops + sum(conv.gather_flops() for conv in self.convs)

class GNN(torch.nn.Module):

    def __init__(self, num_tasks, num_layer = 5, emb_dim = 300):
        '''
            num_tasks (int): number of labels to be predicted
            virtual_node (bool): whether to add virtual node or not
        '''

        super(GNN, self).__init__()

        self.num_layer = num_layer
        self.emb_dim = emb_dim
        self.num_tasks = num_tasks

        if self.num_layer < 2:
            raise ValueError("Number of GNN layers must be greater than 1.")

        self.gnn_node = GNN_node(num_layer, emb_dim)
        # Pooling function to generate whole-graph embeddings
        self.pool = global_mean_pool

        self.graph_pred_linear = torch.nn.Linear(self.emb_dim, self.num_tasks, bias=False)
    
    def my_eval(self, confidences):
        self.flops = 0
        self.gnn_node.my_eval(confidences)
        self.forward = self.my_forward

    def my_forward(self, batched_data):
        self.eval()
        h_node = self.gnn_node(batched_data)
        self.flops += h_node.numel() # global mean pooling flops
        h_graph = self.pool(h_node, batched_data.batch)
        self.flops += h_graph.numel() * self.num_tasks * 2 # linear layer flops, 2 per MAC
        return self.graph_pred_linear(h_graph)
    
    def gather_flops(self):
        return self.flops + self.gnn_node.gather_flops()
    
    def forward(self, batched_data):
        h_node = self.gnn_node(batched_data)

        h_graph = self.pool(h_node, batched_data.batch)

        return self.graph_pred_linear(h_graph)