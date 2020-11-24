import tensorflow as tf
import numpy as np
from gpflow.mean_functions import Identity, Linear
from gpflow import settings
from gpflow import transforms
from gpflow.misc import _broadcasting_elementwise_op
float_type = settings.float_type
from gpflow import params_as_tensors, params_as_tensors_for, ParamList
from gpflow.mean_functions import Zero, MeanFunction
from gpflow.kernels import Stationary, RBF, Kernel
from compatible.ver_adaption import *
from doubly_stochastic_dgp.dgp import DGP_Base
from doubly_stochastic_dgp.layers import Layer
from jack_utils.common import time_it
from gpflow.decors import autoflow
from gpflow.transforms import LowerTriangular, Transform
from dgp_graph.my_op import *

IP3D = InducingPoints  # IP is compatiable with 3d Z

# todo auto mem inc(maybe done by gpflow)
# config = tf.ConfigProto()
# config.gpu_options.allow_growth=True
# sess = tf.Session(config=config)


class LowerTriangular3D(LowerTriangular):
    """
    LowerTriangular Transform for 3D (4d actually) inputs
    It's a reshape wrapper to the original LowerTriangular but keeps computation efficiency.
    The node-fim must be at the first dim.
    """
    def __init__(self, num_nodes, N, dim_out=1, **kwargs):
        super(LowerTriangular3D, self).__init__(N, num_nodes * dim_out, **kwargs)
        self.num_nodes = num_nodes
        self.dim_out = dim_out

    def forward(self, x):
        """
        triangle -> vec each
        :param x: packed x(num_nodes, num_matrices, num_non_zero)
        :return: triangle matrices y(num_nodes, num_matrices, N, N)
        """
        x_ = x.reshape(self.num_nodes*self.dim_out, -1)
        y = super(LowerTriangular3D, self).forward(x_).reshape(self.num_nodes, self.dim_out, self.N, self.N)
        return y

    def backward(self, y):
        """
        triangle -> vec each
        :param y: (num_nodes, num_matrices, N, N) input triangle matrices
        :return: packed x(num_nodes, num_matrices, num_non_zero)
        """
        y_ = y.reshape(self.num_nodes*self.dim_out, self.N, self.N)
        x = super(LowerTriangular3D, self).backward(y_).reshape(self.num_nodes, self.dim_out, -1)
        return x

    def forward_tensor(self, x):
        x_ = tf.reshape(x, (self.num_nodes * self.dim_out, -1))
        y_ = super(LowerTriangular3D, self).forward_tensor(x_)
        y = tf.reshape(y_, (self.num_nodes, self.dim_out, self.N, self.N))
        return y

    def backward_tensor(self, y):
        y_ = tf.reshape(y, (self.num_nodes * self.dim_out, self.N, self.N))
        x_ = super(LowerTriangular3D, self).backward_tensor(y_)
        x = tf.reshape(x_, (self.num_nodes, self.dim_out, -1))
        return x

    def __str__(self):
        return "LoTri_3d->matrix"

    @staticmethod
    def test():
        nodes, dim_in, dim_out = 3, 2, 4
        y = np.random.randint(1, 10, size=(dim_out, dim_in, dim_in)) * np.tri(dim_in)
        y_all = np.tile(y[None, ...], [nodes, 1, 1, 1])

        trans = LowerTriangular(dim_in, num_matrices=dim_out)
        trans_3d = LowerTriangular3D(nodes, dim_in, dim_out)

        # np version
        assert np.array_equal(y, trans.forward(trans.backward(y)))
        assert np.array_equal(y_all, trans_3d.forward(trans_3d.backward(y_all)))
        x_all = trans_3d.backward(y_all)
        x_all_for = np.stack([trans.backward(y_all[i]) for i in range(nodes)], axis=0)
        assert np.array_equal(x_all, x_all_for)
        assert np.array_equal(trans_3d.forward(x_all),
                              np.stack([trans.forward(x_all[i]) for i in range(nodes)], axis=0))

        # tf version
        sess = tf.Session()
        sess.run(tf.assert_equal(y, trans.forward_tensor(trans.backward_tensor(y))))
        sess.run(tf.assert_equal(y_all, trans_3d.forward_tensor(trans_3d.backward_tensor(y_all))))

        x_all = trans_3d.backward_tensor(y_all)
        x_all_for = tf.stack([trans.backward_tensor(y_all[i]) for i in range(nodes)], axis=0)
        sess.run(tf.assert_equal(x_all, x_all_for))
        sess.run(tf.assert_equal(trans_3d.forward_tensor(x_all),
                                 tf.stack([trans.forward_tensor(x_all[i]) for i in range(nodes)], axis=0)))


class DGPG(DGP_Base):
    @time_it
    def __init__(self, X, Y, Z, input_dims, likelihood, adj,
                 agg_op_name='concat3d', ARD=False,
                 is_Z_forward=True, mean_trainable=False, out_mf0=True,
                 kern_type='RBF',
                 **kwargs):
        """
        init layers for graph dgp model.
        :param X: (s1, n, d_in)
        :param Y: (s1, n, d_out)
        :param Z: (s2, n, d_in)
        :param kernels: [(n, d_in)...] length=L
        :param likelihood: todo
        :param adj: (n, n)
        :param is_Z_forward: whether Z should be aggregated and propagated among layers
        """
        assert np.ndim(X) == 3 and np.ndim(Z) == 3 and np.ndim(Y) == 3
        nb_agg = get_nbf_op(agg_op_name)
        num_nodes = adj.shape[0]
        raw_mask = adj.copy()

        # 1. constructing layers
        layers, X_running, Z_running = [], X.copy(), Z.copy()
        layer_n = 0
        for dim_in, dim_out in zip(input_dims[:-1], input_dims[1:]):
            # get in->out dimension for current layer

            # constructing mean function
            W, fixed_nmf = FixedNMF.init(X_running, adj, dim_out, agg_op_name, mean_trainable)

            # constructing kernel
            if 'concat' in agg_op_name:
                mask_concat = neighbour_feats(raw_mask, np.ones((num_nodes, dim_in)))  # (n, n*feat)
                kern = RBFNodes(num_nodes, num_nodes*dim_in, mask=mask_concat, ARD=ARD, layer_n=layer_n, kern_type=kern_type)
            else:
                kern = RBFNodes(num_nodes, dim_in, ARD=ARD, layer_n=layer_n, kern_type=kern_type)

            # init layer
            layers.append(SVGPG_Layer(fixed_nmf, kern, Z_running, adj, dim_out, agg_op_name, is_Z_forward))
            print('input-output dim ({}(agg:{})->{})'.format(dim_in, kern.input_dim, dim_out))

            # propagating X & Z
            if is_Z_forward:
                Z_running = nb_agg(adj, Z_running)
            # warn: if aggregation mode of X is 'concat' and Z is not aggregated，the mean_function of X and Z should be different.
            Z_running = FixedNMF.np_call(Z_running, W)  # (s2, n, d_in) -> (s2, n, d_out)
            X_running = FixedNMF.np_call(nb_agg(adj, X_running), W)  # (s1, n, d_in) -> (s1, n, d_out)
            
            layer_n += 1

        # 2. constructing the last/output layer recording to the shape of Y
        # constructing mean function
        dim_in = input_dims[-1]
        if 'concat' in agg_op_name:
            mask_concat = neighbour_feats(raw_mask, np.ones((num_nodes, dim_in)))
            kern = RBFNodes(num_nodes, num_nodes * dim_in, mask=mask_concat, ARD=ARD, layer_n=layer_n, kern_type=kern_type)
        else:
            kern = RBFNodes(num_nodes, dim_in, ARD=ARD, layer_n=layer_n, kern_type=kern_type)
        mf = Zero() if out_mf0 else FixedNMF.init(X_running, adj, Y.shape[-1], agg_op_name, mean_trainable)[1]
        # init layer
        layers.append(SVGPG_Layer(mf, kern, Z_running, adj, Y.shape[-1], agg_op_name, is_Z_forward))
        print('input-output dim ({}(agg:{})->{})'.format(dim_in, kern.input_dim, Y.shape[-1]))

        # 3. using layers to init the Base model with 2d inputs
        DGP_Base.__init__(self, X.reshape(X.shape[0], -1), Y.reshape(Y.shape[0], -1), likelihood, layers, 
                          name='DGPG', **kwargs)


class SVGPG_Layer(Layer):
    """
        Single layer implementation of SVGP-Graph:
            1) conditional_ND() - compute the mean and covariance of q_f - related to the first term of ELBO
            2) KL() - compute the divergence of q_u - related to the second term of ELBO
    """
    def __init__(self, nodes_mf, kern, Z, adj, dim_per_out, agg_op_name='concat3d', is_Z_forward=True, **kwargs):
        assert np.ndim(Z) == 3
        Layer.__init__(self, input_prop_dim=False, **kwargs)

        # agg_operation
        self.agg = get_nbf_op(agg_op_name)
        self.tf_agg = get_nbf_op(agg_op_name, is_tf=True)

        # shape variables
        self.adj = adj
        self.num_nodes = adj.shape[0]
        self.num_inducing = Z.shape[0]
        self.dim_per_out = dim_per_out
        self.num_outputs = self.num_nodes * dim_per_out

        # neighbour aggregation(e.g., sum, concat, mean) of Z
        Z_agg = self.agg(adj, Z) if is_Z_forward else Z

        # construct and init key Params:
        #   func m: mean_function
        #   func k: kern_function
        # q(u)~N(u|bf{m},bf{S})
        #   bf{m} : q_mu
        #   bf{S} : q_sqrt @ q_sqrt.T
        self.mean_function = nodes_mf
        self.kern = kern
        self.q_mu = Parameter(np.zeros((self.num_nodes, self.num_inducing, dim_per_out)))  # (n, s2, d_out)
        self.feature = IP3D(Z_agg)

        # constructing Param *q_sqrt* with cholesky(Kuu)
        Ku = kern.compute_K_symm(Z_agg)
        Ku += np.eye(self.num_inducing) * settings.jitter  # k(Z, Z)+ I*jitter : (n, s2, s2)
        Lu = np.linalg.cholesky(Ku)  # L = sqrt(k(Z,Z)) : (n, s2, s2)
        q_sqrt = np.tile(np.expand_dims(Lu, axis=1), [1, dim_per_out, 1, 1])  # (n, dim_out, s2, s2)
        self.q_sqrt = Parameter(q_sqrt, transform=LowerTriangular3D(self.num_nodes, self.num_inducing, dim_per_out))

        self.needs_build_cholesky = True

    @params_as_tensors
    def build_cholesky_if_needed(self):
        # make sure we only compute this once
        if self.needs_build_cholesky:
            self.Ku = self.feature.Kzz(self.kern, jitter=settings.jitter)  # (n, s2, s2)
            self.Lu = tf.linalg.cholesky(self.Ku)  # (n, s2, s2) Low Triangle
            self.Ku_tiled = tf.tile(tf.expand_dims(self.Ku, 1), [1, self.dim_per_out, 1, 1])  # (n,d_out,s2,s2)
            self.Lu_tiled = tf.tile(tf.expand_dims(self.Lu, 1), [1, self.dim_per_out, 1, 1])
            self.needs_build_cholesky = False

    def conditional_ND(self, X, full_cov=False):
        """
            wrapper 2d and 3d input/output
        :param X: **2-dim** inputs with shape(samples1, nodes * dim_per_in)
        :param full_cov: not used currently
        :return: return 2-dim mean and variance
        """
        #
        self.build_cholesky_if_needed()
        if tf.shape(X).shape[0] == 2:
            X = tf.reshape(X, [tf.shape(X)[0], self.num_nodes, -1])  # (s1, n*d_in) -> (s1, n, d_in)
            mean, var = self.conditional_ND_3D(X, full_cov=False)
            mean = tf.reshape(tf.transpose(mean, [1, 0, 2]), [tf.shape(X)[0], -1])  # (n,s1,d_out) ->(s1, n*d_out)
            var = tf.reshape(tf.transpose(var, [1, 0, 2]), [tf.shape(X)[0], -1])
            return mean, var
        else:
            return self.conditional_ND_3D(X, full_cov=False)

    def conditional_ND_3D(self, X, full_cov=False):
        """
        implementation of equation (7)(8) of paper Doubly-Stochastic-DGP
        :param X: input X (num_observation=s1, num_nodes=n, dim_per_in=d_in)
        :param full_cov: whether calculating the full covariance
        :return: mean, var
        """
        if full_cov:
            raise NotImplementedError

        # 0. neighbour feat aggregation
        X_agg = self.tf_agg(self.adj, X)  # (s1, n, d_in/n*d_in)

        # 1. calc alpha=k(Z,Z)^{-1}@k(Z, X)
        Kuf = self.feature.Kuf(self.kern, X_agg)  # Kuf(n, s2, s1)
        alpha = tf.matrix_triangular_solve(self.Lu, Kuf, lower=True)  # Lu(n, s2, s2), A(n, s2, s1)
        alpha = tf.matrix_triangular_solve(tf.transpose(self.Lu, [0, 2, 1]), alpha, lower=False)  # the same as above
        alpha_tiled = tf.tile(tf.expand_dims(alpha, axis=1), [1, self.dim_per_out, 1, 1])  # (n, d_out, s2, s1)

        # 2. calc mean of q(f)
        # m(x) + alpha.T @ (mm - m(Z)), mm=0 here. m(Z):self.q_mu
        mean = tf.transpose(alpha, [0, 2, 1]) @ self.q_mu  # (n, s1, s2)@(n, s2, d_out)=(n, s1, d_out)
        mean += tf.transpose(self.mean_function(X_agg), [1, 0, 2])  # (n, s1, d_out) + (s1, n, d_out).T(0,1)

        # 3.1 calc the the 2nd term of covariance
        # SK = -k(Z, Z) + S   | Ku_tiled(n, d_out, s2, s2), q_sqrt(n, d_out, s2, s2)
        SK = -self.Ku_tiled + self.q_sqrt @ tf.transpose(self.q_sqrt, [0, 1, 3, 2])
        # alpha(x).T @ SK @ alpha(x), covariance computation for diag elements only
        delta_cov = tf.reduce_sum(alpha_tiled * (SK @ alpha_tiled), axis=2)  # (n, d_out, s2, s1)-> (n, d_out, s1)

        # 3.2 calc cov = k(X, X) + delta_cov
        Kff = self.kern.Kdiag(X_agg)  # (n, s1)
        var = tf.expand_dims(Kff, 1) + delta_cov  # (n, 1, s1)+(n, d_out, s1) = (n, d_out, s1)
        var = tf.transpose(var, [0, 2, 1])  # (n, d_out, s1) -> (n, s1, d_out)

        return mean, var  # (n, s1, d_out) both

    def KL(self):
        """
        compute KL divergence for each node and sum them up.
        :return: KL divergence from N(q_mu, q_sqrt) to N(0, I), independently for each GP
        """
        self.build_cholesky_if_needed()

        KL = -0.5 * self.num_inducing * self.num_nodes * self.dim_per_out
        # q_sqrt(n,d_out,s2,s2) -> diag:(n,d_out,s2)-> reduce_sum: (n,)
        KL -= 0.5 * tf.reduce_sum(tf.log(tf.square(tf.linalg.diag_part(self.q_sqrt))))
        # Lu(n, s2, s2) -> diag(n, s2) -> reduce_sum(n,)
        KL += tf.reduce_sum(tf.log(tf.linalg.diag_part(self.Lu))) * self.dim_per_out
        # Lu_tiled(n, d_out, s2, s2)
        KL += 0.5 * tf.reduce_sum(tf.square(tf.matrix_triangular_solve(self.Lu_tiled, self.q_sqrt, lower=True)))

        Kinv_m = tf.cholesky_solve(self.Lu, self.q_mu)  # (n, s2, s2),(n, s2, d_out) -> (n, s2, d_out)
        KL += 0.5 * tf.reduce_sum(self.q_mu * Kinv_m)
        return KL

    def conditional_SND(self, X, full_cov=False):
        """
        A multisample conditional, where X is shape (S,N,D_out), independent over samples S

        if full_cov is True
            mean is (S,N,D_out), var is (S,N,N,D_out)

        if full_cov is False
            mean and var are both (S,N,D_out)

        :param X:  The input locations (S,N,D_in)
        :param full_cov: Whether to calculate full covariance or just diagonal
        :return: mean (S,N,D_out), var (S,N,D_out or S,N,N,D_out)
        """
        if full_cov is True:
            raise NotImplementedError
        else:
            S, N, D = tf.shape(X)[0], tf.shape(X)[1], tf.shape(X)[2]
            X_flat = tf.reshape(X, [S * N, D])
            mean, var = self.conditional_ND(X_flat)
            return [tf.reshape(m, [S, N, self.num_outputs]) for m in [mean, var]]


class RBFNodes(Kernel):

    def __init__(self, nodes, input_dim, mask=None, ARD=False, layer_n=0, name='RBFNode', kern_type='RBF'):
        """
        rbf kernel for each node, computed parallel
        :param nodes: number of nodes
        :param input_dim: number of node features
        :param mask: 0-1 adj used to mask represent active dims of feats_dim. (nodes, feats)
        """
        super().__init__(input_dim, active_dims=None, name=name+str(layer_n))
        self.nodes = nodes
        self.mask = mask

        # init params
        self.variance = Parameter(np.ones(nodes), transform=transforms.positive,
                                  dtype=settings.float_type)
        if ARD:
            lengthscales = mask if mask is not None else np.ones((nodes, input_dim))
        else:
            lengthscales = np.ones(nodes, dtype=settings.float_type)
        self.lengthscales = Parameter(lengthscales, transform=transforms.positive,
                                      dtype=settings.float_type)
        
        self.kern_type = kern_type
        self.build()  # very important, it's a confusing point that this class can't be auto built.

    # autoflow annotation used to build tf graph automatically，acquire executed results.
    # a sort of eager excution...
    @autoflow((settings.float_type, [None, None, None]),
              (settings.float_type, [None, None, None]))
    def compute_K(self, X, Z):
        return self.K(X, Z)

    @autoflow((settings.float_type, [None, None, None]))
    def compute_K_symm(self, X):
        return self.K(X)

    @autoflow((settings.float_type, [None, None, None]))
    def compute_Kdiag(self, X):
        return self.Kdiag(X)

    @staticmethod
    def rbf(args):
        """
        :param args: tuples of [X:(s1/s2, d), X2:(s1/s2, d), LS:(d/1,), VAR(1,)]
        :return: rbf variance of X and X2.
        """
        X, X2, lengthscales, variance = args
        #print(X.shape, lengthscales.shape)
        # calculate r2
        X = X / lengthscales
        
        if X2 is None:
            Xs = tf.reduce_sum(tf.square(X), axis=-1, keepdims=True)
            dist = -2 * tf.matmul(X, X, transpose_b=True)
            dist += Xs + tf.transpose(Xs, [1, 0])
            r2 = dist
        else:
            Xs = tf.reduce_sum(tf.square(X), axis=-1)
            X2 = X2 / lengthscales
            X2s = tf.reduce_sum(tf.square(X2), axis=-1)
            r2 = -2 * tf.tensordot(X, X2, [[-1], [-1]])
            r2 += _broadcasting_elementwise_op(tf.add, Xs, X2s)

        # calculate rbf
        rbf = variance * tf.exp(-r2 / 2.)
        return rbf
  
    @staticmethod
    def r(X, X2, lengthscales):
        X = X / lengthscales
        
        if X2 is None:
            Xs = tf.reduce_sum(tf.square(X), axis=-1, keepdims=True)
            dist = -2 * tf.matmul(X, X, transpose_b=True)
            dist += Xs + tf.transpose(Xs, [1, 0])
            r2 = dist
        else:
            Xs = tf.reduce_sum(tf.square(X), axis=-1)
            X2 = X2 / lengthscales
            X2s = tf.reduce_sum(tf.square(X2), axis=-1)
            r2 = -2 * tf.tensordot(X, X2, [[-1], [-1]])
            r2 += _broadcasting_elementwise_op(tf.add, Xs, X2s)

        r = tf.sqrt(tf.maximum(r2, 1e-36))
        return r
    
    @staticmethod
    def m12(args):
        X, X2, lengthscales, variance = args
        r = RBFNodes.r(X, X2, lengthscales)
        m12 = variance * tf.exp(-r)
        return m12
    
    @staticmethod
    def m32(args):
        X, X2, lengthscales, variance = args
        r = RBFNodes.r(X, X2, lengthscales)
        sqrt3 = np.sqrt(3.)
        m32 = variance * (1. + sqrt3 * r) * tf.exp(-sqrt3 * r)
        return m32
    
    @staticmethod
    def m52(args):
        X, X2, lengthscales, variance = args
        r = RBFNodes.r(X, X2, lengthscales)
        sqrt5 = np.sqrt(5.)
        m52 = variance * (1.0 + sqrt5 * r + 5. / 3. * tf.square(r)) * tf.exp(-sqrt5 * r)
        return m52
        
    @staticmethod
    def poly(args):
        X, X2, lengthscales, variance = args
        X = X * (lengthscales)
        if X2 is None:
            ss = tf.matmul(X, X, transpose_b=True)
        else:
            X2 = X2 * (lengthscales)
            ss = tf.tensordot(X, X2, [[-1], [-1]])
        
        dg = 2
        #return ((ss+1)**dg)
        return (ss)
        
    @staticmethod
    def poly2(args):
        X, X2, lengthscales, variance = args
        X = X * (lengthscales)
        if X2 is None:
            ss = tf.matmul(X, X, transpose_b=True)
        else:
            X2 = X2 * (lengthscales)
            ss = tf.tensordot(X, X2, [[-1], [-1]])
        
        dg = 2
        return ((ss+1)**dg)
        #return (ss)
    
    @staticmethod
    def rbf_self(args):
        X, lengthscales, variance = args
        return RBFNodes.rbf([X, None, lengthscales, variance])
    
    @staticmethod
    def m12_self(args):
        X, lengthscales, variance = args
        return RBFNodes.m12([X, None, lengthscales, variance])
    
    @staticmethod
    def m32_self(args):
        X, lengthscales, variance = args
        return RBFNodes.m32([X, None, lengthscales, variance])
    
    @staticmethod
    def m52_self(args):
        X, lengthscales, variance = args
        return RBFNodes.m52([X, None, lengthscales, variance])
    
    @staticmethod
    def poly_self(args):
        X, lengthscales, variance = args
        return RBFNodes.poly([X, None, lengthscales, variance])
    
    @staticmethod
    def poly_self2(args):
        X, lengthscales, variance = args
        return RBFNodes.poly2([X, None, lengthscales, variance])

    @params_as_tensors
    def K(self, X, X2=None):
        """
        calc rbf similarity for each node.
        
        There are two ways to parallel in tf:
        tf.map_fn(lambda x: scaled_square_dist(x[0], x[1]), (A, B), dtype=tf.float32)
        tf.vectorized_map(RBFNodes.rbf, (X_, X2_))
        the second one is recommended.
        :param X: (s1, n, d)
        :param X2: (s2, n, d)
        :return: K(X, X2) = (n, s1, s2)
        """
        assert tf.shape(X).shape[0] == 3

        #print(X.shape, self.lengthscales.shape)
        X_ = tf.transpose(X, [1, 0, 2])  # (n, s1, d)
        if X2 is None:
            if self.kern_type == 'RBF':
                return tf.vectorized_map(RBFNodes.rbf_self, (X_, self.lengthscales, self.variance))  # (n, s1, s1)
            elif self.kern_type == 'Matern12':
                return tf.vectorized_map(RBFNodes.m12_self, (X_, self.lengthscales, self.variance))  # (n, s1, s1)
            elif self.kern_type == 'Matern32':
                return tf.vectorized_map(RBFNodes.m32_self, (X_, self.lengthscales, self.variance))  # (n, s1, s1)
            elif self.kern_type == 'Matern52':
                return tf.vectorized_map(RBFNodes.m52_self, (X_, self.lengthscales, self.variance))  # (n, s1, s1)
            elif self.kern_type == 'Poly1':
                return tf.vectorized_map(RBFNodes.poly_self, (X_, self.lengthscales, self.variance))  # (n, s1, s1)
            elif self.kern_type == 'Poly2':
                return tf.vectorized_map(RBFNodes.poly_self2, (X_, self.lengthscales, self.variance))  # (n, s1, s1)
        else:
            X2_ = tf.transpose(X2, [1, 0, 2])  # (n, s1, d)
            if self.kern_type == 'RBF':
                return tf.vectorized_map(RBFNodes.rbf, (X_, X2_, self.lengthscales, self.variance))  # (n, s1, s2)
            elif self.kern_type == 'Matern12':
                return tf.vectorized_map(RBFNodes.m12, (X_, X2_, self.lengthscales, self.variance))  # (n, s1, s2)
            elif self.kern_type == 'Matern32':
                return tf.vectorized_map(RBFNodes.m32, (X_, X2_, self.lengthscales, self.variance))  # (n, s1, s2)
            elif self.kern_type == 'Matern52':
                return tf.vectorized_map(RBFNodes.m52, (X_, X2_, self.lengthscales, self.variance))  # (n, s1, s2)
            elif self.kern_type == 'Poly1':
                return tf.vectorized_map(RBFNodes.poly, (X_, X2_, self.lengthscales, self.variance))  # (n, s1, s2)
            elif self.kern_type == 'Poly2':
                return tf.vectorized_map(RBFNodes.poly2, (X_, X2_, self.lengthscales, self.variance))  # (n, s1, s2)

    @params_as_tensors
    def Kdiag(self, X):
        """
        calc diag covariance only 
        :param X: (s1, n, d_in)
        :return: (n, s1)
        """
        return tf.tile(tf.expand_dims(self.variance, axis=-1), [1, tf.shape(X)[0]])
        # return tf.fill(tf.shape(X)[:-1], tf.squeeze(self.variance))


def singe_sparse_svd(sparse_matrix, mask, topk, k_apx=1000):
    """
    sparse_matrix: (m, n*feats), mask(n*feats,)
    there are many zero columns in sparse_matrix, masked by mask array.
    return topk component of right singular matrix
    """
    dense_matrix = sparse_matrix.T[np.where(mask > 0)].T
    # approximation
    if dense_matrix.shape[0] > k_apx:
        dense_matrix = dense_matrix[:k_apx]
    _, _, V = np.linalg.svd(dense_matrix)  # V(nb*feats, nb*feats)
    result = np.zeros(shape=(sparse_matrix.shape[-1], topk))
    result[np.where(mask > 0)] = V[:,:topk]
    return result

vec_sparse_svd = np.vectorize(singe_sparse_svd, signature='(n,nf),(nf)->(nf,topk)', excluded=['topk'])


def test_sparse_svd():
    """
    testing for sparse svd
    """
    mask = np.random.randint(0, 2, size=(5, 5 * 4))
    sparse_matrix = np.random.randint(1, 10, size=(5, 3, 5 * 4))
    sparse_matrix = sparse_matrix * mask[:, None, :]
    topk = 2

    # raw method
    results = []
    for i in range(sparse_matrix.shape[0]):
        dense_matrix = sparse_matrix[i].T[np.where(mask[i] == 1)].T
        results.append(np.linalg.svd(dense_matrix)[-1][:, :topk])
    y = results

    # masked & padded method
    x = vec_sparse_svd(sparse_matrix, mask, topk=topk)

    # assert almost equal (ignore extremely small numerical errors)
    for i in range(sparse_matrix.shape[0]):
        np.testing.assert_almost_equal(np.dot(sparse_matrix[i], x[i]),
                                       np.dot(sparse_matrix[i].T[np.where(mask[i] == 1)].T, y[i]))


class FixedNMF(MeanFunction):
    """
    Fixed Nodes Mean Function, The projection parameter W is fixed (i.e., non-trainable)
    """
    def __init__(self, W, b, trainable=False):
        MeanFunction.__init__(self)
        self.W = Parameter(W, trainable=trainable)
        self.b = Parameter(b, trainable=trainable)
        self.trainable = trainable

    @staticmethod
    def init(graph_signal, adj, feat_out, agg_op_name='concat', trainable=False):
        assert np.ndim(graph_signal) == 3  # observations, nodes, feats
        agg_op = get_nbf_op(agg_op_name)
        s, n, feat_in = graph_signal.shape

        #aggregation = agg_op(adj, graph_signal)  # (b, n, f_in/n*f_in)
        aggregation = agg_op(adj, graph_signal[:5000])  # approximate (b, n, f_in/n*f_in)
        feat_in_expand = aggregation.shape[-1]
        if feat_in_expand == feat_out:
            W = np.identity(feat_in_expand)
        elif feat_in_expand > feat_out:
            # calc svd for every node and extract the primary component.
            if 'concat' in agg_op_name:
                mask_concat = neighbour_feats(adj, np.ones((n, feat_in)))  # (nodes, nodes*nbs)
                W = vec_sparse_svd(np.transpose(aggregation, [1, 0, 2]), mask_concat, topk=feat_out)
                # W: (nodes, nbs*feat_in, feat_out)
            else:
                _, _, V = np.linalg.svd(np.transpose(aggregation, [1, 0, 2]))  # (nodes, feat_in, feat_in)
                W = np.transpose(V[:, :feat_out], [0, 2, 1])  # (nodes, feat_in, feat_out)
        else:
            # (f_in, f_out)
            W = np.concatenate([np.eye(feat_in_expand), np.zeros((feat_in_expand, feat_out - feat_in_expand))], 1)
        b = np.zeros(feat_out)
        mean_function = FixedNMF(W, b, trainable=trainable)
        return W, mean_function

    def __call__(self, X):
        """
        calculate mean for every node recording to different dimension cases.
        :param input: X(s,n,f_in) or X(s, n*f_in)
        :return: mean(X) - (s, n, f_out)
        """
        assert tf.shape(X).shape[0] == 3
        if tf.shape(self.W).shape[0] == 2:  # identity/padding
            mX = tf.tensordot(X, self.W, [[-1], [0]])  # X(s,n,f_in), W(f_in,f_out) padding case
        else:  # transforming for each node separately with different
            mX = tf.expand_dims(X, axis=-2) @ self.W  # X(s,n,1,f_in), W(n,f_in,f_out) pca case
            mX = tf.reduce_sum(mX, axis=-2)  # mX(s,n,f_out)
        return mX + self.b

    @staticmethod
    def np_call(X, W):
        assert np.ndim(X) == 3
        if np.ndim(W) == 2:  # identity/padding
            mX = np.matmul(X, W)  # X(s,n,f_in), W(f_in,f_out) padding case
        else:  # transforming for each node separately with different
            mX = np.matmul(np.expand_dims(X, axis=-2), W)  # X(s,n,1,f_in), W(n,f_in,f_out) pca case
            mX = np.sum(mX, axis=-2)  # mX(s,n,f_out)
        return mX

    @staticmethod
    def test_agg():
        gs = np.random.rand(4, 3, 2)
        adj = np.round(np.random.rand(3, 3), 0)
        agg = np.matmul(adj, gs)
        agg_for = np.stack([np.matmul(adj, gs[i]) for i in range(gs.shape[0])], axis=0)
        assert np.array_equal(agg, agg_for)

    @staticmethod
    def test_call():
        x = tf.random.uniform((9, 3, 4))

        # case 1: for f_in < f_out (padding)
        w = tf.random.uniform((4, 6))
        xw = tf.tensordot(x, w, [[-1], [0]])
        xw_for = tf.stack([x[i]@w for i in range(x.shape[0])], axis=0)
        tf.assert_equal(xw, xw_for)

        # case 2 for f_in > f_out (pca case, do pca for each node)
        x = tf.expand_dims(x, axis=-2)  # (9, 3, 1, 4)
        w = tf.random.uniform((3, 4, 2))
        xw = x@w
        xw_for = tf.stack([x[i]@w for i in range(x.shape[0])], axis=0)
        tf.assert_equal(xw, xw_for)

if __name__ == '__main__':
    test_sparse_svd()
