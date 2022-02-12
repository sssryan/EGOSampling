import json, os
import math, copy, time
import numpy as np
from collections import defaultdict
import pandas as pd
from .utils import *
import random
import math
from tqdm import tqdm

import seaborn as sb
import matplotlib.pyplot as plt
import matplotlib.cm as cm

import dill
from functools import partial
import multiprocessing as mp

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




def EGOSampling(graph, sampled_depth = 2, sampled_number = 8, inp = None, feature_extractor = feature_MAG):
    layer_data  = defaultdict(lambda: {})
    cache     = defaultdict(lambda: defaultdict(lambda: [0., 0, 0.]))
    
    def add_to_cache(te, target_type, target_id, target_time, layer_data, cache):
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
                    source_time = adl[source_id]
                    if source_time == None:
                        source_time = target_time
                    if source_id in layer_data[source_type]:
                        continue
                    cache[source_type][source_id][0] += 1. / len(sampled_ids)
                    cache[source_type][source_id][1] = source_time
                    
                    te1 = graph.edge_list[source_type]
                    if 'rev' not in relation_type:
                        rev_rel = 'rev_' + relation_type
                    if 'rev' in relation_type:
                        rev_rel = relation_type.strip('rev_')
                    tes1 = te1[target_type][rev_rel]
                    adl1 = tes1[source_id]
                    div = (1. / len(sampled_ids)) * (1. / len(adl1.keys()))
                    if cache[source_type][source_id][2] == 0:
                        cache[source_type][source_id][2] = div
                    if cache[source_type][source_id][2] != 0:
                        prop = max(cache[source_type][source_id][2],div)
                        cache[source_type][source_id][2] = prop
 

    for _type in inp:
        for _id, _time in inp[_type]:
            layer_data[_type][_id] = [len(layer_data[_type]), _time]
    for _type in inp:
        te = graph.edge_list[_type]
        for _id, _time in inp[_type]:
            add_to_cache(te, _type, _id, _time, layer_data, cache)
    
    
    for layer in range(sampled_depth):
        new_layer  = defaultdict( #target_type
                        lambda: [] # {target_ids
                    )
        sts = list(cache.keys())
        
        for source_type in sts:
            keys  = np.array(list(cache[source_type].keys()))
            if sampled_number > len(keys):
                sampled_ids = np.arange(len(keys))
            else:
                
                score = np.array(list(cache[source_type].values()))[:,0] ** 2
                score = score / np.sum(score)
                sampled_ids_0 = np.random.choice(len(score), sampled_number, p = score, replace = False)

                score_con = np.array(list(cache[source_type].values()))[:,2] ** 2
                score_con = score_con / np.sum(score_con)
                sampled_ids_1 = np.random.choice(len(score_con), sampled_number, p = score_con, replace = False)
                sampled_ids = \
            np.array(list(set(sampled_ids_0.tolist()).union(set(sampled_ids_1.tolist()))))
            sampled_keys = keys[sampled_ids]
            
            for k in sampled_keys:
                layer_data[source_type][k] = [len(layer_data[source_type]), cache[source_type][k][1]]
                new_layer[source_type].append(k)
        for node_type in new_layer:
            te = graph.edge_list[node_type]
            for k in new_layer[node_type]:
                add_to_cache(te, node_type, k, cache[node_type][k][1], layer_data, cache)
                cache[node_type].pop(k)
        
    feature, times, indxs, texts = feature_extractor(layer_data, graph)
            
    edge_list = defaultdict( #target_type
                        lambda: defaultdict(  #source_type
                            lambda: defaultdict(  #relation_type
                                lambda: [] # [target_id, source_id] 
                                    )))
    for _type in layer_data:
        for _key in layer_data[_type]:
            _ser = layer_data[_type][_key][0]
            edge_list[_type][_type]['self'] += [[_ser, _ser]]
    
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
    return feature, times, edge_list, indxs, texts

def to_torch(feature, time, edge_list, graph):
    '''
        Transform a sampled sub-graph into pytorch Tensor
        node_dict: {node_type: <node_number, node_type_ID>} node_number is used to trace back the nodes in original graph.
        edge_dict: {edge_type: edge_type_ID}
    '''
    node_dict = {}
    node_feature = []
    node_type    = []
    node_time    = []
    edge_index   = []
    edge_type    = []
    edge_time    = []
    
    node_num = 0
    types = graph.get_types()
    for t in types:
        node_dict[t] = [node_num, len(node_dict)]
        node_num     += len(feature[t])

    for t in types:
        node_feature += list(feature[t])
        node_time    += list(time[t])
        node_type    += [node_dict[t][1] for _ in range(len(feature[t]))]
        
    edge_dict = {e[2]: i for i, e in enumerate(graph.get_meta_graph())}
    edge_dict['self'] = len(edge_dict)

    for target_type in edge_list:
        for source_type in edge_list[target_type]:
            for relation_type in edge_list[target_type][source_type]:
                for ii, (ti, si) in enumerate(edge_list[target_type][source_type][relation_type]):
                    tid, sid = ti + node_dict[target_type][0], si + node_dict[source_type][0]
                    edge_index += [[sid, tid]]
                    edge_type  += [edge_dict[relation_type]]   
                    '''
                        Our time ranges from 1900 - 2020, largest span is 120.
                    '''
                    edge_time  += [node_time[tid] - node_time[sid] + 120]
    node_feature = torch.FloatTensor(node_feature)
    node_type    = torch.LongTensor(node_type)
    edge_time    = torch.LongTensor(edge_time)
    edge_index   = torch.LongTensor(edge_index).t()
    edge_type    = torch.LongTensor(edge_type)
    return node_feature, node_type, edge_time, edge_index, edge_type, node_dict, edge_dict
    
