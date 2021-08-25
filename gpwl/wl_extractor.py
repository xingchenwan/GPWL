from models.gpwl.continuous_wl import ContinuousWeisfeilerLehman
import numpy as np
from typing import List
import dgl


class WeisfeilerLehmanExtractor:
    def __init__(self, h: int = 1, debug=False, node_attr='node_attr'):
        """
        This class extracts the Weisfeiler-Lehman features from graphs and return as np.arrays.
        :param h: the maximum number of Weisfiler-Lehman iterations
        :param debug: bool. Whether to display diagnostic information
        """
        self.h = h
        self.wl = ContinuousWeisfeilerLehman(h=h, node_feat_name=node_attr)
        self.node_attr = node_attr
        self.debug = debug
        # the feature vector of the training set
        self.base_kernels = None

    def fit(self, g_list: List[dgl.DGLGraph]):
        """Fit the WL feature vector on the input list of graphs. These graphs will be considered the "training set".
        g_list: A list of DGL graphs
        """
        if self.wl is not None:
            del self.wl
        self.wl = ContinuousWeisfeilerLehman(h=self.h, node_feat_name=self.node_attr)
        train_graphs = g_list
        self.wl.fit(train_graphs)
        self.base_kernels = self.wl.X

    def update(self, g_list: List[dgl.DGLGraph]):
        """This function concatenates the stored training feature vector with the new feature vectors in g_list provided.
        Since the new graphs supplied might introduce new WL features, this def also updates the fitted inv_label
        dict of the WL kernels, and that of base VertexHistogram kernel at each WL iteration level."""
        if self.wl is None or self.wl.X is None:
            print('The WL kernel is uninitialised. Call the fit method instead')
            self.fit(g_list)
        else:
            # eval_graphs = dgl2networkx(g_list, self.node_attr)
            feat_vector = self.wl.parse_input(g_list, train_mode=True)
            self.wl.X = np.vstack((self.wl.X, feat_vector))

    def transform(self, h_list: List[dgl.DGLGraph]):
        """This is used for prediction mode. Similar to update but the new features seen in the h_list graphs will
        not be recorded by the WL or the base (vertex histogram) kernels."""
        if self.wl is None or self.wl.X is None:
            raise ValueError("Base kernel is None. Did you call fit or fit_transform first?")
        # eval_graphs = dgl2networkx(h_list, self.node_attr)
        feature_vector = self.wl.parse_input(h_list, train_mode=False)
        return feature_vector

    def get_train_features(self):
        """Get the WL feature vector of the graphs of which the WLFeatureExtractor is currently fitted"""
        return np.copy(self.wl.X)
