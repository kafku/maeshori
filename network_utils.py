# coding: utf-8

import math
import random
import functools
import numpy as np
from keras.utils import Sequence
from more_itertools import random_product
from .gen_utils import stack_batch_list


def node2index(nodes):
    return dict((node, i) for i, node in enumerate(nodes[0]))


def _random_node_pairs(nodes1, nodes2, graph, n_sample):
    """
    return `n_sample` random node pairs (nodes1 vs node2)
    """
    sampled_pairs = []
    while True:
        node_pair = random_product(nodes1, nodes2)
        if not graph.has_edge(*node_pair):
            sampled_pairs.append(node_pair)

        if len(sampled_pairs) == n_sample:
            return sampled_pairs


def _prepare_batch(node_pairs, nodes, node1_idx, node2_idx,
                   node_features, get_link_weight,
                   domain1_key, domain2_key):
    """
    prepare batch from node pairs
    """
    node1 = [x[0] if x[0] in nodes[0] else x[1] for x in node_pairs]
    node2 = [x[1] if x[1] in nodes[1] else x[0] for x in node_pairs]

    # view-1 feature
    if node_features is not None and node_features[0] is not None:
        domain1_feature = node_features[0].loc[node1].values
    else:
        domain1_feature = np.array([node1_idx[n] for n in node1]).reshape((len(node1), 1))

    # view-2 feature
    if node_features is not None and node_features[1] is not None:
        domain2_feature = node_features[1].loc[node2].values
    else:
        domain2_feature = np.array([node2_idx[n] for n in node2]).reshape((len(node2), 1))

    link_weight = np.array([get_link_weight(n1, n2) for n1, n2 in zip(node1, node2)])
    link_weight = link_weight.reshape((len(node1), 1))

    return {domain1_key: domain1_feature,
            domain2_key: domain2_feature}, link_weight


def network_rand_gen(graph, nodes=None, node_features=None, dtype='float32',
                     batch_size=64, n_negative=32,
                     domain1_key="domain1_input",
                     domain2_key="domain2_input"):
    if nodes is None:
        # take it as 1-view graph if nodes is not specified
        nodes = [graph.nodes, graph.nodes]

    assert len(nodes) == 2

    # prepare node index
    # FIXME: what if more than 3 views?
    node1_idx = dict()
    node2_idx = dict()
    for i, node in enumerate(nodes[0]):
        node1_idx[node] = i

    for i, node in enumerate(nodes[1]):
        node2_idx[node] = i

    # TODO: enable getting edge-weight
    get_link_weight = lambda n1, n2: 1 if graph.has_edge(n1, n2) else 0

    prepare_batch = functools.partial(
        _prepare_batch,
        nodes=nodes, node1_idx=node1_idx, node2_idx=node2_idx,
        node_features=node_features,
        get_link_weight=get_link_weight,
        domain1_key=domain1_key,
        domain2_key=domain2_key)

    # randomly outputs positive and negative samples
    while True:
        if n_negative > 0:
            yield stack_batch_list([
                prepare_batch(random.sample(graph.edges, batch_size - n_negative)),
                prepare_batch(_random_node_pairs(nodes[0], nodes[1], graph, n_negative))
            ])
        else:
            yield prepare_batch(random.sample(graph.edges, batch_size)),


class NodePairSeq(Sequence):
    def __init__(self, graph, nodes=None, node_features=None, dtype='float32',
                 batch_size=64,
                 domain1_key="domain1_input",
                 domain2_key="domain2_input"):
        if nodes is None:
            # take it as 1-view graph if nodes is not specified
            nodes = [graph.nodes, graph.nodes]

        assert len(nodes) == 2

        # prepare node index
        # FIXME: what if more than 3 views?
        node1_idx = dict()
        node2_idx = dict()
        for i, node in enumerate(nodes[0]):
            node1_idx[node] = i

        for i, node in enumerate(nodes[1]):
            node2_idx[node] = i

        # TODO: enable getting edge-weight
        get_link_weight = lambda n1, n2: 1 if graph.has_edge(n1, n2) else 0

        self.batch_size = batch_size
        self.nodes = nodes
        self.prepare_batch = functools.partial(
            _prepare_batch,
            nodes=nodes, node1_idx=node1_idx, node2_idx=node2_idx,
            node_features=node_features,
            get_link_weight=get_link_weight,
            domain1_key=domain1_key,
            domain2_key=domain2_key)

    def __len__(self):
        return math.ceil(len(self.nodes[0]) * len(self.nodes[1]) / self.batch_size)

    def __getitem__(self, idx):
        data_idx = idx * self.batch_size
        node1_batch_idx = data_idx // len(self.nodes[1])
        node2_batch_idx = data_idx % len(self.nodes[1])

        node_pairs = []
        for _ in range(self.batch_size):
            node_pairs.append((self.nodes[0][node1_batch_idx],
                               self.nodes[1][node2_batch_idx]))
            node2_batch_idx += 1
            if node2_batch_idx == len(self.nodes[1]):
                node2_batch_idx = 0
                node1_batch_idx += 1

            if node1_batch_idx == len(self.nodes[0]):
                break

        return self.prepare_batch(node_pairs)
