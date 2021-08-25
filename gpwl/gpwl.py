import gpytorch
import torch
import dgl
from models.gpwl.wl_extractor import WeisfeilerLehmanExtractor
from gpytorch.likelihoods import GaussianLikelihood
from gpytorch.constraints.constraints import Interval
import numpy as np
from copy import deepcopy
from models.gpwl.utils import to_unit_cube, from_unit_normal, to_unit_normal
from models.predictor import Predictor
from typing import List
from utils.encodings import encode
import networkx as nx


class GP(gpytorch.models.ExactGP):
    """Implementation of the exact GP in the gpytorch framework"""
    def __init__(self, train_x, train_y, kernel: gpytorch.kernels.Kernel, likelihood):
        super(GP, self).__init__(train_x, train_y, likelihood)
        self.mean_module = gpytorch.means.ConstantMean()
        self.covar_module = gpytorch.kernels.ScaleKernel(kernel)

    def forward(self, x):
        mean_x = self.mean_module(x)
        covar_x = self.covar_module(x)
        return gpytorch.distributions.MultivariateNormal(mean_x, covar_x)


class OptimalAssignment(gpytorch.kernels.Kernel):
    """Implementation of the optimal assignment kernel as histogram intersection between two vectors"""
    def forward(self, x1, x2, diag=False, last_dim_is_batch=False, **params):
        """x1 shape = [N x d], x2_shape = [M x d]. This computes the pairwise histogram intersection between the two"""
        ker = torch.zeros(x1.shape[0], x2.shape[0])
        for n in range(x1.shape[0]):
            for m in range(x2.shape[0]):
                ker[n, m] = torch.sum(torch.minimum(x1[n, :].reshape(-1, 1), x2[m, :].reshape(-1, 1)))
        if diag:
            return torch.diag(ker)
        return ker


def train_gp(train_x, train_y, training_iter=5, kernel='linear', verbose=False, init_noise_var=None, hypers={}, ls=[0.01, 0.1]):
    """Train a GP model. Since in our case we do not have lengthscale, the optimisation is about finding the optimal
    noise only.
    train_x, train_y: the training input/targets (in torch.Tensors) for the GP
    training_iter: the number of optimiser iterations. Set to 0 if you do not wish to optimise
    kernel: 'linear' for the original WL kernel. 'oa' for the optimal assignment variant
    verbose: if True, display diagnostic information during optimisation
    init_noise_var: Initial noise variance. Supply a value here and set training_iter=0 when you have a good knowledge
        of the noise variance a-priori to skip inferring noise from the data.
    hypers: Optional dict of hyperparameters for the GP.
    Return: a trained GP object.
    """

    assert train_x.ndim == 2
    assert train_y.ndim == 1
    assert train_x.shape[0] == train_y.shape[0]

    noise_constraint = Interval(1e-6, 0.1)
    likelihood = GaussianLikelihood(noise_constraint=noise_constraint).to(device=train_x.device, dtype=train_y.dtype)
    lengthscale_constraint = Interval(ls[0], ls[1])
    if kernel == 'linear':
        k = gpytorch.kernels.LinearKernel()
    elif kernel == 'oa':
        k = OptimalAssignment()
    elif kernel == 'rbf':
        k = gpytorch.kernels.RBFKernel(lengthscale_constraint=lengthscale_constraint)
    elif kernel == 'matern':
        k = gpytorch.kernels.MaternKernel(lengthscale_constraint=lengthscale_constraint)
    elif kernel == 'cosine':
        k = gpytorch.kernels.CosineKernel(lengthscale_constraint=lengthscale_constraint)

    else:
        raise NotImplementedError
    # model
    model = GP(train_x, train_y, k, likelihood).to(device=train_x.device, dtype=train_x.dtype)

    model.train()
    likelihood.train()

    # Use the adam optimizer
    optimizer = torch.optim.Adam([
        {'params': model.parameters()},  # Includes GaussianLikelihood parameters
    ], lr=0.1)

    # "Loss" for GPs - the marginal log likelihood
    mll = gpytorch.mlls.ExactMarginalLogLikelihood(likelihood, model)

    if hypers:
        model.load_state_dict(hypers)
    else:
        hypers = {}
        hypers["covar_module.outputscale"] = 1.0
        if model.covar_module.base_kernel.has_lengthscale:
            hypers["covar_module.base_kernel.lengthscale"] = np.sqrt(0.01 * 0.5)
        hypers["likelihood.noise"] = 0.005 if init_noise_var is None else init_noise_var
        model.initialize(**hypers)

    for i in range(training_iter):
        # Zero gradients from previous iteration
        optimizer.zero_grad()
        # Output from model
        output = model(train_x)
        # Calc loss and backprop gradients
        loss = -mll(output, train_y)
        loss.backward()
        if verbose:
            print('Iter %d/%d - Loss: %.3f. Noise %.3f' % (
                i + 1, training_iter, loss.item(),  model.likelihood.noise.item()
            ))
            if model.covar_module.base_kernel.has_lengthscale:
                print(model.covar_module.base_kernel.lengthscale)
        optimizer.step()

    model.eval()
    likelihood.eval()
    return model


def parse_to_dgl(adjacency_matrix: List[np.array],
                 node_feat_matrix: List[np.array],
                 node_feat_name: str = 'node_attr'):

    def numpy_to_graph(A, graph_type='dgl', node_features=None):
        """Convert numpy arrays to graph

        Parameters
        ----------
        A : mxm array
            Adjacency matrix
        graph_type : str
            'dgl' or 'nx'
        node_features : dict
            Optional, dictionary with key=feature name, value=list of size m
            Allows user to specify node features

        Returns
        -------
        Graph of 'type_graph' specification
        """

        G = nx.from_numpy_array(A)

        if node_features is not None:
            for n in G.nodes():
                for k, v in node_features.items():
                    G.nodes[n][k] = v[n]
        if graph_type == 'nx':
            return G

        G = G.to_directed()

        if node_features is not None:
            node_attrs = list(node_features.keys())
        else:
            node_attrs = []

        g = dgl.from_networkx(G, node_attrs=node_attrs, edge_attrs=['weight'])
        return g

    assert len(adjacency_matrix) == len(node_feat_matrix), 'unequal length between adjacency_matrix list and ' \
                                                           'node_feat_matrix list'
    n_graphs = len(adjacency_matrix)
    dgl_graphs = []
    for i in range(n_graphs):
        new_graph = numpy_to_graph(adjacency_matrix[i], node_features={node_feat_name: node_feat_matrix[i]})
        dgl_graphs.append(new_graph)
    return dgl_graphs


class GPWL(Predictor):
    def __init__(self,
                 ss_type=None,
                 encoding_type=None,
                 max_adj_size=None, max_feat_size=None, default_hyperparams=None,):
        """
        A simple GPWL interface which uses GP with WL kernel (note that when the original WL kernel is used, due to
        the linear kernel formulation it is simply WL kernel + Bayesian linear regression

        default_hyperparams: options
            h: int. Number of Weisfeiler-Lehman iterations. Default is 1.
            kernel: str. The type of kernel to use. 'oa' (optimal assignment) or 'linear'. 'oa' is much slower but
                often has an empirically better performance. Default is 'oa'. If you find the training to be too
                slow, consider switching to 'linear' for this flag.
            noise_var: float. The noise variance in the data if known a-priori. If None, the noise variance will be
                inferred from data via optimising the log-marginal likelihood
            node_attr: str. The node feature name in the default graph we are interested in. Default is 'node_attr'.
            n_training_step: int. Number of training steps for the GP log-marginal likelihood optimisation.
                NOTE: if oa is chosen, it is advisable to use a small value as otherwise the training will be quite
                    slow.
        """
        super().__init__(ss_type, encoding_type)
        if default_hyperparams is None: default_hyperparams = {}

        self.hyperparams = default_hyperparams
        if 'h' not in self.hyperparams.keys(): self.hyperparams.update({'h': 1})
        if 'kernel' not in self.hyperparams.keys(): self.hyperparams.update({'kernel': 'oa'})
        if 'noise_var' not in self.hyperparams.keys(): self.hyperparams.update({'noise_var': None})
        if 'node_attr' not in self.hyperparams.keys(): self.hyperparams.update({'node_attr': 'node_attr'})
        if 'log_transform' not in self.hyperparams.keys(): self.hyperparams.update({'log_transform': True})
        if 'n_training_step' not in self.hyperparams.keys(): self.hyperparams.update({'n_training_step': 5})

        self.extractor = WeisfeilerLehmanExtractor(h=self.hyperparams['h'], node_attr=self.hyperparams['node_attr'])
        self.kernel = self.hyperparams['kernel']
        self.noise_var = self.hyperparams['noise_var']
        self.max_adj_size, self.max_feat_size = max_adj_size, max_feat_size
        self.gp = None

    def fit(self, xtrain, ytrain, train_info=None, **kwargs):
        """See BasePredictor"""

        train_adj, train_feat = [], []
        for i, arch in enumerate(xtrain):
            encoded = encode(arch, ss_type=self.ss_type, max_adj_size=self.max_adj_size, max_feat_size=self.max_feat_size,
                             method='gpwl')
            train_adj.append(encoded['adjacency'])
            train_feat.append(encoded['node_features'])
        # convert into appropriate format
        xtrain = parse_to_dgl(train_adj, train_feat)

        if not isinstance(ytrain, torch.Tensor): ytrain = torch.tensor(ytrain, dtype=torch.float32)
        if self.hyperparams['log_transform']: ytrain = torch.log(ytrain)
        if len(ytrain.shape) == 0:  # y_train is a scalar
            ytrain = ytrain.reshape(1)
        assert len(xtrain) == ytrain.shape[0]
        assert ytrain.ndim == 1
        # Fit the feature extractor with the graph input
        self.extractor.fit(xtrain)
        self.X = deepcopy(xtrain)
        self.y = deepcopy(ytrain)
        # Get the vector representation out
        x_feat_vector = torch.tensor(self.extractor.get_train_features(), dtype=torch.float32)
        # the noise variance is provided, no need for training

        # standardise x_feat_vector into unit hypercube [0, 1]^d
        self.lb, self.ub = torch.min(x_feat_vector, dim=0)[0]-1e-3, torch.max(x_feat_vector, dim=0)[0]+1e-3
        x_feat_vector_gp = to_unit_cube(x_feat_vector, self.lb, self.ub)
        # normalise y vector into unit normal distribution
        self.ymean, self.ystd = torch.mean(ytrain), torch.std(ytrain)
        y_train_normal = to_unit_normal(ytrain, self.ymean, self.ystd)
        if self.noise_var is not None:
            self.gp = train_gp(x_feat_vector_gp, y_train_normal, training_iter=0, kernel=self.kernel, init_noise_var=self.noise_var)
        else:
            self.gp = train_gp(x_feat_vector_gp, y_train_normal, kernel=self.kernel, training_iter=self.hyperparams['n_training_step'])

    def update(self, x_update, y_update):
        """See BasePredictor"""
        if len(y_update.shape) == 0:  # y_train is a scalar
            y_update = y_update.reshape(1)
        assert len(x_update) == y_update.shape[0]
        assert y_update.ndim == 1

        update_adj, update_feat = [], []
        for i, arch in enumerate(x_update):
            encoded = encode(arch, ss_type=self.ss_type, max_adj_size=self.max_adj_size,
                             max_feat_size=self.max_feat_size, )
            update_adj.append(encoded['adjacency'])
            update_feat.append(encoded['node_features'])
        # convert into appropriate format
        x_update = parse_to_dgl(update_adj, update_feat)
        if not isinstance(y_update, torch.Tensor): ytrain = torch.tensor(y_update, dtype=torch.float32)
        if self.hyperparams['log_transform']: y_update = torch.log(y_update)

        self.extractor.update(x_update)
        x_feat_vector = torch.tensor(self.extractor.get_train_features(), dtype=torch.float32)

        # update the lb and ub, in case new information changes those
        self.lb, self.ub = torch.min(x_feat_vector, dim=0)[0]-1e-2, torch.max(x_feat_vector, dim=0)[0]+1e-2
        # x_feat_vector_gp = x_feat_vector
        x_feat_vector_gp = to_unit_cube(x_feat_vector, self.lb, self.ub)
        self.X += deepcopy(x_update)
        self.y = torch.cat((self.y, y_update))
        self.ymean, self.ystd = torch.mean(self.y), torch.std(self.y)
        y = to_unit_normal(deepcopy(self.y), self.ymean, self.ystd)
        if self.noise_var is not None:
            self.gp = train_gp(x_feat_vector_gp, y, training_iter=0, kernel=self.kernel, init_noise_var=self.noise_var)
        else:
            self.gp = train_gp(x_feat_vector_gp, y, kernel=self.kernel)

    def predict(self, xtest, include_noise_variance=False, full_cov=False, **kwargs):
        """
        See BasePredict
        :param include_noise_variance: bool. Whether include noise variance in the prediction. This does not impact
            the posterior mean inference, but the posterior variance inference will be enlarged accordingly if this flag
            is True
        :return: (mean, std (if full_covariance=True) or the full covariance matrix (if full_covariance=True))
        """
        if self.gp is None:
            raise ValueError("The GPWL object is not fitted to any data yet! Call fit or update to do so first.")

        test_adj, test_feat = [], []
        for i, arch in enumerate(xtest):
            encoded = encode(arch, ss_type=self.ss_type, max_adj_size=self.max_adj_size,
                             max_feat_size=self.max_feat_size, method='gpwl')
            test_adj.append(encoded['adjacency'])
            test_feat.append(encoded['node_features'])
        # convert into appropriate format
        x_eval = parse_to_dgl(test_adj, test_feat)

        x_feat_vector = torch.tensor(self.extractor.transform(x_eval), dtype=torch.float32)
        x_feat_vector = to_unit_cube(x_feat_vector, self.lb, self.ub)
        self.gp.eval()
        pred = self.gp(x_feat_vector)
        # print(pred.mean)
        if include_noise_variance:
            self.gp.likelihood.eval()
            pred = self.gp.likelihood(pred)
        mean, variance = pred.mean.detach(), pred.variance.detach()
        if full_cov:
            covariance = pred.covariance_matrix.detach()
        mean = from_unit_normal(mean, self.ymean, self.ystd)
        variance = from_unit_normal(variance, self.ymean, self.ystd, scale_variance=True)
        stddev = torch.sqrt(variance)
        if self.hyperparams['log_transform']:
            mean = torch.exp(mean)
            stddev = torch.exp(stddev)
        if full_cov:
            return mean.numpy(), stddev.numpy(), covariance.numpy()
        else:
            return mean.numpy(), stddev.numpy()

