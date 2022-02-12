from copy import copy
import argparse
from tqdm import tqdm

import torch
import torch.nn.functional as F
from torch.nn import ModuleList, Linear, ParameterDict, Parameter
from torch_sparse import SparseTensor
from torch_geometric.utils import to_undirected
from torch_geometric.data import Data, GraphSAINTRandomWalkSampler
from torch_geometric.utils.hetero import group_hetero_graph
from torch_geometric.nn import MessagePassing

from ogb.nodeproppred import PygNodePropPredDataset, Evaluator

import dill
from logger import Logger
import numpy as np
from collections import defaultdict
import multiprocessing as mp
import torch
import scipy.sparse as sp
import random

parser = argparse.ArgumentParser(description='OGBN-MAG (GraphSAINT)')
parser.add_argument('--device', type=int, default=0)
parser.add_argument('--num_layers', type=int, default=2)
parser.add_argument('--hidden_channels', type=int, default=64)
parser.add_argument('--dropout', type=float, default=0.5)
parser.add_argument('--lr', type=float, default=0.005)
parser.add_argument('--epochs', type=int, default=30)
parser.add_argument('--runs', type=int, default=10)
parser.add_argument('--batch_size', type=int, default=20000)
parser.add_argument('--walk_length', type=int, default=2)
parser.add_argument('--num_steps', type=int, default=30)
parser.add_argument('--sample_depth', type=int, default=6)
parser.add_argument('--sample_width', type=int, default=520)
parser.add_argument('--data_dir', type=str, default='./dataset/OGB_MAG.pk')
args = parser.parse_args()
print(args)

dataset = PygNodePropPredDataset(name='ogbn-mag')
data = dataset[0]
edge_index_dict = data.edge_index_dict

class Graph():
    def __init__(self):
        super(Graph, self).__init__()
        '''
            node_forward and bacward are only used when building the data. 
            Afterwards will be transformed into node_feature by DataFrame
            
            node_forward: name -> node_id
            node_bacward: node_id -> feature_dict
            node_feature: a DataFrame containing all features
        '''
        self.node_forward = defaultdict(lambda: {})
        self.node_bacward = defaultdict(lambda: [])
        self.node_feature = defaultdict(lambda: [])

        '''
            edge_list: index the adjacancy matrix (time) by 
            <target_type, source_type, relation_type, target_id, source_id>
        '''
        self.edge_list = defaultdict( #target_type
                            lambda: defaultdict(  #source_type
                                lambda: defaultdict(  #relation_type
                                    lambda: defaultdict(  #target_id
                                        lambda: defaultdict( #source_id(
                                            lambda: int # time
                                        )))))
        self.times = {}
    def add_node(self, node):
        nfl = self.node_forward[node['type']]
        if node['id'] not in nfl:
            self.node_bacward[node['type']] += [node]
            ser = len(nfl)
            nfl[node['id']] = ser
            return ser
        return nfl[node['id']]
    def add_edge(self, source_node, target_node, time = None, relation_type = None, directed = True):
        edge = [self.add_node(source_node), self.add_node(target_node)]
        '''
            Add bi-directional edges with different relation type
        '''
        self.edge_list[target_node['type']][source_node['type']][relation_type][edge[1]][edge[0]] = time
        if directed:
            self.edge_list[source_node['type']][target_node['type']]['rev_' + relation_type][edge[0]][edge[1]] = time
        else:
            self.edge_list[source_node['type']][target_node['type']][relation_type][edge[0]][edge[1]] = time
        self.times[time] = True
        
    def update_node(self, node):
        nbl = self.node_bacward[node['type']]
        ser = self.add_node(node)
        for k in node:
            if k not in nbl[ser]:
                nbl[ser][k] = node[k]

    def get_meta_graph(self):
        types = self.get_types()
        metas = []
        for target_type in self.edge_list:
            for source_type in self.edge_list[target_type]:
                for r_type in self.edge_list[target_type][source_type]:
                    metas += [(target_type, source_type, r_type)]
        return metas
    
    def get_types(self):
        return list(self.node_feature.keys())
    
def randint():
    return np.random.randint(2**32 - 1)

def normalize(mx):
    """Row-normalize sparse matrix"""
    rowsum = np.array(mx.sum(1))
    r_inv = np.power(rowsum, -1).flatten()
    r_inv[np.isinf(r_inv)] = 0.
    r_mat_inv = sp.diags(r_inv)
    mx = r_mat_inv.dot(mx)
    return mx

dataset = PygNodePropPredDataset(name='ogbn-mag')
data = dataset[0]
split_idx = dataset.get_idx_split()
evaluator = Evaluator(name='ogbn-mag')
logger = Logger(args.runs, args)

def feature_MAG(layer_data, graph):
    feature = {}
    indxs   = {}
    texts   = []
    for _type in layer_data:
        if len(layer_data[_type]) == 0:
            continue
        idxs  = np.array(list(layer_data[_type].keys()), dtype = np.int)
        feature[_type] = graph.node_feature[_type][idxs]
        indxs[_type]   = idxs
        
    return feature, indxs, texts


def to_torch(feature, indxs, edge_list, graph):
    '''
        Transform a sampled sub-graph into pytorch Tensor
        node_dict: {node_type: <node_number, node_type_ID>} node_number is used to trace back the nodes in original graph.
        edge_dict: {edge_type: edge_type_ID}
    '''
    node_dict = {}
    node_feature = []
    node_type    = []
    node_idx = []
    edge_index   = []
    edge_type    = []
    
    node_num = 0
    types = graph.get_types()
    for t in types:
        node_dict[t] = [node_num, len(node_dict)]
        node_num     += len(feature[t])

    for t in types:
        node_feature += list(feature[t])
        node_type    += [node_dict[t][1] for _ in range(len(feature[t]))]
        node_idx.extend(indxs[t].tolist())
    edge_dict = {e[2]: i for i, e in enumerate(graph.get_meta_graph())}
    
    for target_type in edge_list:
        for source_type in edge_list[target_type]:
            for relation_type in edge_list[target_type][source_type]:
                for ii, (ti, si) in enumerate(edge_list[target_type][source_type][relation_type]):
                    tid, sid = ti + node_dict[target_type][0], si + node_dict[source_type][0]
                    edge_index += [[sid, tid]]
                    edge_type  += [edge_dict[relation_type]]   

    node_feature = torch.FloatTensor(node_feature)
    node_type    = torch.LongTensor(node_type)
    edge_index   = torch.LongTensor(edge_index).t()
    edge_type    = torch.LongTensor(edge_type)
    node_idx = torch.LongTensor(node_idx)
    return node_feature, node_type, node_idx, edge_index, edge_type, node_dict, edge_dict

def EGOSampling(graph, sampled_depth = 2, sampled_number = 8, inp = None, feature_extractor = feature_MAG):
    
    layer_data  = defaultdict(lambda: {})
    cache     = defaultdict(lambda: defaultdict(lambda: [0., 0.]))

    def add_to_cache(te, target_type, target_id, layer_data, cache):
        for source_type in te:
            tes = te[source_type]
            for relation_type in tes:
                if relation_type == 'self' or target_id not in tes[relation_type]:
                    continue
                adl = tes[relation_type][target_id]
                if len(adl) < sampled_number:
                    sampled_ids = list(adl.keys())
                else:
                    sampled_ids = np.random.choice(list(adl.keys()), sampled_number, replace = False)
                for source_id in sampled_ids:
                    
                    if source_id in layer_data[source_type]:
                        continue
                    cache[source_type][source_id][0] += 1. / len(sampled_ids)
                    
                    
                    te1 = graph.edge_list[source_type]
                    if 'rev' not in relation_type:
                        rev_rel = 'rev_' + relation_type
                    if 'rev' in relation_type:
                        rev_rel = relation_type.strip('rev_')
                    tes1 = te1[target_type][rev_rel]
                    adl1 = tes1[source_id]
                    div = (1. / len(sampled_ids)) * (1. / len(adl1.keys()))
                    if cache[source_type][source_id][1] == 0:
                        cache[source_type][source_id][1] = div
                    if cache[source_type][source_id][1] != 0:
                        prop = max(cache[source_type][source_id][1],div)
                        cache[source_type][source_id][1] = prop
 
    for _type in inp:
        for _id, _time in inp[_type]:
            layer_data[_type][_id] = [len(layer_data[_type])]
    for _type in inp:
        te = graph.edge_list[_type]
        for _id, _time in inp[_type]:
            add_to_cache(te, _type, _id, layer_data, cache)
    
    for layer in range(sampled_depth):
        sts = list(cache.keys())
        new_layer  = defaultdict( #target_type
                        lambda: [] # {target_ids
                    )
        for source_type in sts:
            keys  = np.array(list(cache[source_type].keys()))
            if sampled_number > len(keys):
                sampled_ids = np.arange(len(keys))
            else:
                score = np.array(list(cache[source_type].values()))[:,0] ** 2
                score = score / np.sum(score)
                sampled_ids_0 = np.random.choice(len(score), sampled_number, p = score, replace = False)
                
                score_div = score_div ** 2
                score_div = score_div / np.sum(score_div)
                sampled_ids_1 = np.random.choice(len(score_div), 100, p = score_div, replace = False)
                sampled_ids = \
                np.array(list(set(sampled_ids_0.tolist()).union(set(sampled_ids_1.tolist()))))
            sampled_keys = keys[sampled_ids]
           
            for k in sampled_keys:
                layer_data[source_type][k] = [len(layer_data[source_type])]
                new_layer[source_type].append(k)
        for node_type_ in new_layer:
            te = graph.edge_list[node_type_]
            for k in new_layer[node_type_]:
                add_to_cache(te, node_type_, k, layer_data, cache)
                cache[node_type_].pop(k)
   
    feature, indxs, texts = feature_extractor(layer_data, graph)
            
    edge_list = defaultdict( #target_type
                        lambda: defaultdict(  #source_type
                            lambda: defaultdict(  #relation_type
                                lambda: [] # [target_id, source_id] 
                                    )))

    for target_type in graph.edge_list:
        te = graph.edge_list[target_type]
        tld = layer_data[target_type]
        for source_type in te:
            tes = te[source_type]
            sld  = layer_data[source_type]
            for relation_type in tes:
                tesr = tes[relation_type]
                for target_key in tld:
                    if target_key not in tesr:
                        continue
                    target_ser = tld[target_key][0]
                    for source_key in tesr[target_key]:
                        if source_key in sld:
                            source_ser = sld[source_key][0]
                            edge_list[target_type][source_type][relation_type] += [[target_ser, source_ser]]
    return feature, edge_list, indxs, texts

def prepare_data():
    target_nodes_p = np.arange(len(graph.node_feature['paper']))
    samp_nodes_p = np.random.choice(target_nodes_p, args.batch_size, replace = False)
    feature, edge_list, indxs, _ = EGOSampling(graph, \
    inp = {'paper': np.concatenate([samp_nodes_p, graph.years[samp_nodes_p]]).reshape(2, -1).transpose(),},\
                sampled_depth = args.sample_depth, sampled_number = args.sample_width, \
                    feature_extractor = feature_MAG)
    node_feature, node_type, node_idx, edge_index, edge_type, node_dict, edge_dict = \
            to_torch(feature, indxs, edge_list, graph)
    train_mask_p = graph.train_mask[indxs['paper']]
    train_mask_left = np.zeros((len(node_type)-len(train_mask_p)),dtype=bool)
    train_mask = np.concatenate((train_mask_p,train_mask_left),axis=0)
    temp_y = graph.y[indxs['paper']]
    y_left = np.ones((len(node_type)-len(temp_y)),dtype=int)
    y_left *= -1
    y = np.concatenate((temp_y,y_left),axis=0)
    y = np.expand_dims(y,1)
    y = torch.tensor(y)
    return edge_index, edge_type, node_type, node_idx, y, train_mask
    
graph = dill.load(open(args.data_dir, 'rb'))
        
edge_index_dict = data.edge_index_dict

# We need to add reverse edges to the heterogeneous graph.
r, c = edge_index_dict[('author', 'affiliated_with', 'institution')]
edge_index_dict[('institution', 'rev_affiliated_with', 'author')] = torch.stack([c, r])

r, c = edge_index_dict[('author', 'writes', 'paper')]
edge_index_dict[('paper', 'rev_writes', 'author')] = torch.stack([c, r])

r, c = edge_index_dict[('paper', 'has_topic', 'field_of_study')]
edge_index_dict[('field_of_study', 'rev_has_topic', 'paper')] = torch.stack([c, r])

# Convert to undirected paper <-> paper relation.
r, c = edge_index_dict[('paper', 'cites', 'paper')]
edge_index_dict[('paper', 'rev_cites', 'paper')] = torch.stack([c, r])


# Map informations to their canonical type.
x_dict = {}
for key, x in data.x_dict.items():
    x_dict[0] = x

key2int = {'paper':0,'author':1,'field_of_study':2,'institution':3,\
          ('author', 'affiliated_with', 'institution'):0,('institution', 'rev_affiliated_with', 'author'):1,\
          ('paper', 'rev_writes', 'author'):2, ('author', 'writes', 'paper'):3,('paper', 'cites', 'paper'):4,\
          ('paper', 'rev_cites', 'paper'):5,('field_of_study', 'rev_has_topic', 'paper'):6,\
          ('paper', 'has_topic', 'field_of_study'):7}
    
num_nodes_dict = {}
for key, N in data.num_nodes_dict.items():
    num_nodes_dict[key2int[key]] = N

    

class RGCNConv(MessagePassing):
    def __init__(self, in_channels, out_channels, num_node_types,
                 num_edge_types):
        super(RGCNConv, self).__init__(aggr='mean')

        self.in_channels = in_channels
        self.out_channels = out_channels
        self.num_node_types = num_node_types
        self.num_edge_types = num_edge_types

        self.rel_lins = ModuleList([
            Linear(in_channels, out_channels, bias=False)
            for _ in range(num_edge_types)
        ])

        self.root_lins = ModuleList([
            Linear(in_channels, out_channels, bias=True)
            for _ in range(num_node_types)
        ])

        self.reset_parameters()

    def reset_parameters(self):
        for lin in self.rel_lins:
            lin.reset_parameters()
        for lin in self.root_lins:
            lin.reset_parameters()

    def forward(self, x, edge_index, edge_type, node_type):
        out = x.new_zeros(x.size(0), self.out_channels)
        for i in range(self.num_edge_types):
            mask = edge_type == i
            out.add_(self.propagate(edge_index[:, mask], x=x, edge_type=i))

        for i in range(self.num_node_types):
            mask = node_type == i
            print(x.shape)
            out[mask] += self.root_lins[i](x[mask])

        return out

    def message(self, x_j, edge_type: int):
        return self.rel_lins[edge_type](x_j)


class RGCN(torch.nn.Module):
    def __init__(self, in_channels, hidden_channels, out_channels, num_layers,
                 dropout, num_nodes_dict, x_types, num_edge_types):
        super(RGCN, self).__init__()

        self.in_channels = in_channels
        self.hidden_channels = hidden_channels
        self.out_channels = out_channels
        self.num_layers = num_layers
        self.dropout = dropout

        node_types = list(num_nodes_dict.keys())
        num_node_types = len(node_types)

        self.num_node_types = num_node_types
        self.num_edge_types = num_edge_types

        # Create embeddings for all node types that do not come with features.
        self.emb_dict = ParameterDict({
            f'{key}': Parameter(torch.Tensor(num_nodes_dict[key], in_channels))
            for key in set(node_types).difference(set(x_types))
        })

        I, H, O = in_channels, hidden_channels, out_channels  # noqa

        # Create `num_layers` many message passing layers.
        self.convs = ModuleList()
        self.convs.append(RGCNConv(I, H, num_node_types, num_edge_types))
        for _ in range(num_layers - 2):
            self.convs.append(RGCNConv(H, H, num_node_types, num_edge_types))
        self.convs.append(RGCNConv(H, O, self.num_node_types, num_edge_types))

        self.reset_parameters()

    def reset_parameters(self):
        for emb in self.emb_dict.values():
            torch.nn.init.xavier_uniform_(emb)
        for conv in self.convs:
            conv.reset_parameters()

    def group_input(self, x_dict, node_type, local_node_idx):
        # Create global node feature matrix.
        h = torch.zeros((node_type.size(0), self.in_channels),
                        device=node_type.device)

        for key, x in x_dict.items():
            mask = node_type == key
            h[mask] = x[local_node_idx[mask]]

        for key, emb in self.emb_dict.items():
            mask = node_type == int(key)
            h[mask] = emb[local_node_idx[mask]]

        return h

    def forward(self, x_dict, edge_index, edge_type, node_type,
                local_node_idx):

        x = self.group_input(x_dict, node_type, local_node_idx)

        for i, conv in enumerate(self.convs):
            x = conv(x, edge_index, edge_type, node_type)
            if i != self.num_layers - 1:
                x = F.relu(x)
                x = F.dropout(x, p=0.5, training=self.training)

        return x.log_softmax(dim=-1)

    def inference(self, x_dict, edge_index_dict, key2int):
        # We can perform full-batch inference on GPU.

        device = list(x_dict.values())[0].device

        x_dict = copy(x_dict)
        for key, emb in self.emb_dict.items():
            x_dict[int(key)] = emb

        adj_t_dict = {}
        for key, (row, col) in edge_index_dict.items():
            adj_t_dict[key] = SparseTensor(row=col, col=row).to(device)

        for i, conv in enumerate(self.convs):
            out_dict = {}

            for j, x in x_dict.items():
                out_dict[j] = conv.root_lins[j](x)

            for keys, adj_t in adj_t_dict.items():
                src_key, target_key = keys[0], keys[-1]
                out = out_dict[key2int[target_key]]
                tmp = adj_t.matmul(x_dict[key2int[src_key]], reduce='mean')
                out.add_(conv.rel_lins[key2int[keys]](tmp))

            if i != self.num_layers - 1:
                for j in range(self.num_node_types):
                    F.relu_(out_dict[j])

            x_dict = out_dict

        return x_dict


device = f'cuda:{args.device}' if torch.cuda.is_available() else 'cpu'

model = RGCN(128, args.hidden_channels, dataset.num_classes, args.num_layers,
             args.dropout, num_nodes_dict, list(x_dict.keys()),
             len(edge_index_dict.keys())).to(device)

x_dict = {k: v.to(device) for k, v in x_dict.items()}


def train(epoch):
    model.train()

    pbar = tqdm(total=args.num_steps * args.batch_size)
    pbar.set_description(f'Epoch {epoch:02d}')

    total_loss = total_examples = 0
    for i in range(args.num_steps):
        edge_index, edge_type, node_type, node_idx, y, train_mask = prepare_data()
        optimizer.zero_grad()
        y = y.to(device)
        out = model(x_dict, edge_index.to(device), edge_type.to(device), node_type.to(device),
                    node_idx.to(device))
        out = out[train_mask]
        y = y[train_mask].squeeze()
        loss = F.nll_loss(out, y)
        loss.backward()
        optimizer.step()

        num_examples = train_mask.sum().item()
        total_loss += loss.item() * num_examples
        total_examples += num_examples
        pbar.update(args.batch_size)

    pbar.close()

    return total_loss / total_examples


@torch.no_grad()
def test():
    model.eval()

    out = model.inference(x_dict, edge_index_dict, key2int)
    out = out[key2int['paper']]

    y_pred = out.argmax(dim=-1, keepdim=True).cpu()
    y_true = data.y_dict['paper']

    train_acc = evaluator.eval({
        'y_true': y_true[split_idx['train']['paper']],
        'y_pred': y_pred[split_idx['train']['paper']],
    })['acc']
    valid_acc = evaluator.eval({
        'y_true': y_true[split_idx['valid']['paper']],
        'y_pred': y_pred[split_idx['valid']['paper']],
    })['acc']
    test_acc = evaluator.eval({
        'y_true': y_true[split_idx['test']['paper']],
        'y_pred': y_pred[split_idx['test']['paper']],
    })['acc']

    return train_acc, valid_acc, test_acc


test()  # Test if inference on GPU succeeds.
for run in range(args.runs):
    model.reset_parameters()
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    for epoch in range(1, 1 + args.epochs):
        loss = train(epoch)
        torch.cuda.empty_cache()
        result = test()
        logger.add_result(run, result)
        train_acc, valid_acc, test_acc = result
        print(f'Run: {run + 1:02d}, '
              f'Epoch: {epoch:02d}, '
              f'Loss: {loss:.4f}, '
              f'Train: {100 * train_acc:.2f}%, '
              f'Valid: {100 * valid_acc:.2f}%, '
              f'Test: {100 * test_acc:.2f}%')
    logger.print_statistics(run)
logger.print_statistics()
