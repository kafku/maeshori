# coding: utf-8

import random
import numpy as np
import itertools
from more_itertools import chunked


def network_generator(graph, nodes=None, node_features=None, dtype='float32',
                      batch_size=128, shuffle=True,
                      domain1_key="domain1_input", domain2_key="domain2_input"):
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

    while True:
        if shuffle:
            nodes[0] = np.random.permutation(nodes[0])
            nodes[1] = np.random.permutation(nodes[1])

        for node_pairs in chunked(itertools.product(*nodes), batch_size): #FIXME: make it shuffled
            node1 = [x[0] for x in node_pairs]
            node2 = [x[1] for x in node_pairs]

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
            link_weight.reshape((len(node1), 1))

            return {domain1_key: domain1_feature, domain2_key: domain2_feature}, link_weight
