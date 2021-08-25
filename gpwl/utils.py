# Xingchen Wan | 5 March 2020

import networkx as nx
from grakel.utils import graph_from_networkx
from typing import List
import dgl


def dgl2networkx(g_list: List[dgl.DGLGraph], attr_name='node_attr'):
    """Convert a list of dgl graphs to a networkx graph"""
    def convert_single_graph(g):
        g_nx = g.to_networkx().to_undirected()
        nx.set_node_attributes(g_nx, dict(g_nx.degree()), attr_name)
        return g_nx
    graphs = [convert_single_graph(g) for g in g_list]
    return graphs


def dgl2grakel(g_list: List[dgl.DGLGraph], attr_name='node_attr'):
    """Convert a list of DGL graphs to a list of graphs understood by the grakel interface"""
    nx_graph = dgl2networkx(g_list, attr_name)
    graphs = graph_from_networkx(nx_graph, attr_name)
    return graphs


def networkx2grakel(g_list: List[nx.Graph], attr_name='node_attr'):
    """Alias for graph_from_networkx for consistency"""
    return graph_from_networkx(g_list, attr_name)


def to_unit_cube(x, lb, ub):
    """Project to [0, 1]^d from hypercube with bounds lb and ub"""
    # assert lb.ndimension() == 1 and ub.ndimension() == 1 and x.ndimension() == 2
    xx = (x - lb) / (ub - lb)
    return xx


def from_unit_cube(x, lb, ub):
    """Project from [0, 1]^d to hypercube with bounds lb and ub"""
    # assert lb.ndimension() == 1 and ub.ndimension() == 1 and x.ndimension() == 2
    xx = x * (ub - lb) + lb
    return xx


def to_unit_normal(y, mean, std):
    """Normalise targets into the range ~N(0, 1)"""
    return (y - mean) / std


def from_unit_normal(y, mean, std, scale_variance=False):
    """Project the ~N(0, 1) to the original range.
    :param scale_variance: whether we are scaling variance instead of mean.
    """
    if not scale_variance:
        return y * std + mean
    else:
        return y * (std ** 2)
