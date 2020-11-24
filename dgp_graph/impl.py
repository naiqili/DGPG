import tensorflow as tf
import numpy as np

from gpflow.params import DataHolder, Minibatch
from gpflow import autoflow, params_as_tensors, ParamList
from gpflow.models.model import Model
from gpflow.mean_functions import Identity, Linear
from gpflow.mean_functions import Zero
from gpflow.quadrature import mvhermgauss
from gpflow.likelihoods import Gaussian
from gpflow import settings
from gpflow import transforms
from gpflow.kernels import RBF, White
float_type = settings.float_type
from gpflow import params_as_tensors, params_as_tensors_for, ParamList
from gpflow.mean_functions import Zero
from compatible.ver_adaption import *
from doubly_stochastic_dgp.utils import reparameterize
from doubly_stochastic_dgp.utils import BroadcastingLikelihood
from doubly_stochastic_dgp.layers import SVGP_Layer
from doubly_stochastic_dgp.dgp import DGP_Base
from doubly_stochastic_dgp.layers import Layer
from jack_utils.common import time_it

"""
Deprecated implementation. Please use impl_parallel.
"""

@DeprecationWarning
class DGPG(DGP_Base):
    @time_it
    def __init__(self, X, Y, Z, kernels, likelihood, gmat,
                 num_layers=2,
                 num_nodes=None, 
                 dim_per_node=5,
                 dim_per_X=5, dim_per_Y=5,
                 mean_function=Zero(),  # the final layer mean function,
                 num_samples=1, num_data=None,
                 minibatch_size=None,
                 full_cov=False,
                 share_Z=False,
                 nb_init=True,
                 **kwargs):                
        layers = init_layers_graph(X, Y, Z, kernels, gmat,
                             num_layers,
                             num_nodes,
                             dim_per_node,
                             dim_per_X, dim_per_Y,
                             share_Z=share_Z, nb_init=nb_init)

        DGP_Base.__init__(self, X, Y, likelihood, layers, **kwargs)

@DeprecationWarning
class SVGPG_Layer(Layer):
    def __init__(self, kern, Z, mean_function, num_nodes, dim_per_in, dim_per_out,
                 gmat, share_Z=False, nb_init=True, **kwargs):

        Layer.__init__(self, input_prop_dim=False, **kwargs)

        self.kern = kern
        self.num_nodes = num_nodes
        self.dim_per_in, self.dim_per_out = dim_per_in, dim_per_out
        self.gmat = gmat
        self.share_Z = share_Z
        self.nb_init = nb_init
        self.num_outputs = num_nodes * dim_per_out
        self.num_inducing = Z.shape[0]

        self.q_mu = Parameter(np.zeros((self.num_inducing, num_nodes * dim_per_out)))
        self.mean_function = ParamList([], trainable=False)
        self.q_sqrt_lst = ParamList([])
        transform = transforms.LowerTriangular(self.num_inducing, num_matrices=self.dim_per_out)

        if share_Z:
            self.feature = InducingPoints(Z)
        else:
            self.feature = ParamList([])  # InducingPoints(Z)

        for nd in range(num_nodes):
            if mean_function:
                self.mean_function.append(mean_function[nd])
            else:
                self.mean_function.append(Zero())
            if share_Z:
                pa_nd = self.pa_idx(nd)
                Ku_nd = self.kern[nd].compute_K_symm(Z)
                Lu_nd = np.linalg.cholesky(Ku_nd + np.eye(Z.shape[0]) * settings.jitter)
                q_sqrt = np.tile(Lu_nd[None, :, :], [dim_per_out, 1, 1])
                self.q_sqrt_lst.append(Parameter(q_sqrt, transform=transform))
            else:
                pa_nd = self.pa_idx(nd)
                Z_tmp = Z[:, pa_nd].copy()
                self.feature.append(InducingPoints(Z_tmp))
                Ku_nd = self.kern[nd].compute_K_symm(Z_tmp)
                Lu_nd = np.linalg.cholesky(Ku_nd + np.eye(Z_tmp.shape[0]) * settings.jitter)
                q_sqrt = np.tile(Lu_nd[None, :, :], [dim_per_out, 1, 1])
                self.q_sqrt_lst.append(Parameter(q_sqrt, transform=transform))

        self.needs_build_cholesky = True

    def pa_idx(self, nd):
        res = []
        for n in range(self.num_nodes):
            w = self.gmat[nd, n]
            if w > 0:
                res = res + list(range(n * self.dim_per_in, (n + 1) * self.dim_per_in))
        res = np.asarray(res)
        return res

    @params_as_tensors
    def build_cholesky_if_needed(self):
        # make sure we only compute this once
        if self.needs_build_cholesky:
            self.Ku, self.Lu = [None] * self.num_nodes, [None] * self.num_nodes
            self.Ku_tiled_lst, self.Lu_tiled_lst = [], []
            for nd in range(self.num_nodes):
                if self.share_Z:
                    Ku_nd = self.feature.Kuu(self.kern[nd], jitter=settings.jitter)
                else:
                    Ku_nd = self.feature[nd].Kuu(self.kern[nd], jitter=settings.jitter)
                Lu_nd = tf.cholesky(Ku_nd)
                self.Ku[nd] = Ku_nd
                self.Lu[nd] = Lu_nd
                self.Ku_tiled_lst.append(tf.tile(Ku_nd[None, :, :], [self.dim_per_out, 1, 1]))
                self.Lu_tiled_lst.append(tf.tile(Lu_nd[None, :, :], [self.dim_per_out, 1, 1]))
            self.needs_build_cholesky = False

    @time_it
    def conditional_ND(self, X, full_cov=False):
        self.build_cholesky_if_needed()

        if self.share_Z:
            return self.conditional_ND_share_Z(X, full_cov=False)
        else:
            return self.conditional_ND_not_share_Z(X, full_cov=False)

    def conditional_ND_share_Z(self, X, full_cov=False):
        mean_lst, var_lst, A_tiled_lst = [], [], []
        for nd in range(self.num_nodes):
            pa_nd = self.pa_idx(nd)
            Kuf_nd = self.feature.Kuf(self.kern[nd], X)

            A_nd = tf.matrix_triangular_solve(self.Lu[nd], Kuf_nd, lower=True)
            A_nd = tf.matrix_triangular_solve(tf.transpose(self.Lu[nd]), A_nd, lower=False)
            mean_tmp = tf.matmul(A_nd, self.q_mu[:, nd * self.dim_per_out:(nd + 1) * self.dim_per_out],
                                 transpose_a=True)
            X_tmp = tf.gather(X, pa_nd, axis=1)
            if self.nb_init:
                mean_tmp += self.mean_function[nd](X_tmp)
            else:
                mean_tmp += self.mean_function[nd](X[:, nd * self.dim_per_in:(nd + 1) * self.dim_per_in])
            mean_lst.append(mean_tmp)

            A_tiled_lst.append(tf.tile(A_nd[None, :, :], [self.dim_per_out, 1, 1]))
            SK_nd = -self.Ku_tiled_lst[nd]
            q_sqrt_nd = self.q_sqrt_lst[nd]
            with params_as_tensors_for(q_sqrt_nd, convert=True):
                SK_nd += tf.matmul(q_sqrt_nd, q_sqrt_nd, transpose_b=True)

            B_nd = tf.matmul(SK_nd, A_tiled_lst[nd])
            # (num_latent, num_X)
            delta_cov_nd = tf.reduce_sum(A_tiled_lst[nd] * B_nd, 1)
            Kff_nd = self.kern[nd].Kdiag(X)

            # either (1, num_X) + (num_latent, num_X)
            var_nd = tf.expand_dims(Kff_nd, 0) + delta_cov_nd
            var_nd = tf.transpose(var_nd)

            var_lst.append(var_nd)

        mean = tf.concat(mean_lst, axis=1)
        var = tf.concat(var_lst, axis=1)
        return mean, var

    def conditional_ND_not_share_Z(self, X, full_cov=False):
        mean_lst, var_lst, A_tiled_lst = [], [], []
        for nd in range(self.num_nodes):
            pa_nd = self.pa_idx(nd)
            X_tmp = tf.gather(X, pa_nd, axis=1)
            Kuf_nd = self.feature[nd].Kuf(self.kern[nd], X_tmp)

            A_nd = tf.matrix_triangular_solve(self.Lu[nd], Kuf_nd, lower=True)
            A_nd = tf.matrix_triangular_solve(tf.transpose(self.Lu[nd]), A_nd, lower=False)

            mean_tmp = tf.matmul(A_nd, self.q_mu[:, nd * self.dim_per_out:(nd + 1) * self.dim_per_out],
                                 transpose_a=True)
            if self.nb_init:
                mean_tmp += self.mean_function[nd](X_tmp)
            else:
                mean_tmp += self.mean_function[nd](X[:, nd * self.dim_per_in:(nd + 1) * self.dim_per_in])
            mean_lst.append(mean_tmp)
            A_tiled_lst.append(tf.tile(A_nd[None, :, :], [self.dim_per_out, 1, 1]))

            SK_nd = -self.Ku_tiled_lst[nd]
            q_sqrt_nd = self.q_sqrt_lst[nd]
            with params_as_tensors_for(q_sqrt_nd, convert=True):
                SK_nd += tf.matmul(q_sqrt_nd, q_sqrt_nd, transpose_b=True)

            B_nd = tf.matmul(SK_nd, A_tiled_lst[nd])

            # (num_latent, num_X)
            delta_cov_nd = tf.reduce_sum(A_tiled_lst[nd] * B_nd, 1)
            Kff_nd = self.kern[nd].Kdiag(X_tmp)

            # (1, num_X) + (num_latent, num_X)
            var_nd = tf.expand_dims(Kff_nd, 0) + delta_cov_nd
            var_nd = tf.transpose(var_nd)

            var_lst.append(var_nd)

        mean = tf.concat(mean_lst, axis=1)
        var = tf.concat(var_lst, axis=1)
        return mean, var

    @time_it
    def KL(self):
        """
        The KL divergence from the variational distribution to the prior

        :return: KL divergence from N(q_mu, q_sqrt) to N(0, I), independently for each GP
        """

        self.build_cholesky_if_needed()

        KL = -0.5 * self.num_inducing * self.num_nodes * self.dim_per_out

        for nd in range(self.num_nodes):
            q_sqrt_nd = self.q_sqrt_lst[nd]
            with params_as_tensors_for(q_sqrt_nd, convert=True):
                KL -= 0.5 * tf.reduce_sum(tf.log(tf.matrix_diag_part(q_sqrt_nd) ** 2))

                KL += tf.reduce_sum(tf.log(tf.matrix_diag_part(self.Lu[nd]))) * self.dim_per_out
                KL += 0.5 * tf.reduce_sum(
                    tf.square(tf.matrix_triangular_solve(self.Lu_tiled_lst[nd], q_sqrt_nd, lower=True)))
                q_mu_nd = self.q_mu[:, nd * self.dim_per_out:(nd + 1) * self.dim_per_out]
                Kinv_m_nd = tf.cholesky_solve(self.Lu[nd], q_mu_nd)
                KL += 0.5 * tf.reduce_sum(q_mu_nd * Kinv_m_nd)

        return KL

@DeprecationWarning
@time_it
def init_layers_graph(X, Y, Z, kernels, gmat,
                      num_layers=2,
                      num_nodes=None,
                      dim_per_node=5,
                      dim_per_X=5, dim_per_Y=5,
                      share_Z=False,
                      nb_init=True):
    layers = []

    def pa_idx(nd, dim_per_in):
        res = []
        for n in range(num_nodes):
            w = gmat[nd, n]
            if w > 0:
                # print(res, range(n*self.dim_per_in, (n+1)*self.dim_per_in))
                res = res + list(range(n * dim_per_in, (n + 1) * dim_per_in))
        res = np.asarray(res)
        return res

    X_running, Z_running = X.copy(), Z.copy()
    for l in range(num_layers - 1):
        if l == 0:
            dim_in = dim_per_X
            dim_out = dim_per_node
        else:
            dim_in = dim_per_node
            dim_out = dim_per_node
        # print(dim_in, dim_out)
        X_running_tmp = np.zeros((X.shape[0], dim_out * num_nodes))
        Z_running_tmp = np.zeros((Z.shape[0], dim_out * num_nodes))
        mf_lst = ParamList([], trainable=False)
        for nd in range(num_nodes):
            if nb_init:
                pa = pa_idx(nd, dim_in)
            else:
                pa = np.asarray(range(nd * dim_in, (nd + 1) * dim_in))
            agg_dim_in = len(pa)

            if agg_dim_in == dim_out:
                mf = Identity()

            else:
                if agg_dim_in > dim_out:  # stepping down, use the pca projection
                    # _, _, V = np.linalg.svd(X_running[:, nd*dim_in : (nd+1)*dim_in], full_matrices=False)
                    _, _, V = np.linalg.svd(X_running[:, pa], full_matrices=False)
                    W = V[:dim_out, :].T

                else:  # stepping up, use identity + padding
                    W = np.concatenate([np.eye(agg_dim_in), np.zeros((agg_dim_in, dim_out - agg_dim_in))], 1)

                mf = Linear(W)
                mf.set_trainable(False)
            mf_lst.append(mf)
            if agg_dim_in != dim_out:
                # print(Z_running_tmp[:, nd*dim_out:(nd+1)*dim_out].shape, Z_running[:, nd*dim_in:(nd+1)*dim_in].shape,
                #       W.shape, Z_running[:, nd*dim_in:(nd+1)*dim_in].dot(W).shape)
                Z_running_tmp[:, nd * dim_out:(nd + 1) * dim_out] = Z_running[:, pa].dot(W)
                X_running_tmp[:, nd * dim_out:(nd + 1) * dim_out] = X_running[:, pa].dot(W)
            else:
                Z_running_tmp[:, nd * dim_out:(nd + 1) * dim_out] = Z_running[:, pa]
                X_running_tmp[:, nd * dim_out:(nd + 1) * dim_out] = X_running[:, pa]

        layers.append(
            SVGPG_Layer(kernels[l], Z_running, mf_lst, num_nodes, dim_in, dim_out, gmat, share_Z=share_Z, nb_init=nb_init))
        Z_running = Z_running_tmp
        X_running = X_running_tmp

    # final layer
    if num_layers == 1:
        fin_dim_in = dim_per_X
    else:
        fin_dim_in = dim_per_node
    layers.append(
        SVGPG_Layer(kernels[-1], Z_running, None, num_nodes, fin_dim_in, dim_per_Y, gmat, share_Z=share_Z, nb_init=nb_init))
    return layers