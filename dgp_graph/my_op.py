import numpy as np
import tensorflow as tf


def neighbour_feats(adj, x):
    # in: adj(n, n),  x(n, feat)
    # out: selected(n, n*feat)
    a_ = np.expand_dims(adj, axis=-1)  # note the location of expanding
    selected = a_*x
    return selected.reshape(x.shape[0], -1)


def tf_neighbour_feats(adj, x):
    # in: adj(n, n),  x(n, feat)
    # out: selected(n, n*feat)
    a_ = tf.expand_dims(adj, axis=-1)  # note the location of expanding
    selected = a_*x
    return tf.reshape(selected, (x.shape[0], -1))


def neighbour_feats3d(adj, batch_x):
    # adj(n, n),  batch_x（batch, n, feat）
    # out: selected(batch, n, n*feat)
    a_ = np.expand_dims(adj, axis=-1)
    x_ = np.expand_dims(batch_x, axis=1)  # note the location of expanding
    selected = a_*x_
    return selected.reshape(batch_x.shape[0], batch_x.shape[1], -1)


def tf_neighbour_feats3d(adj, batch_x):
    # adj(n, n),  batch_x（batch, n, feat）
    # out: selected(batch, n, n*feat)
    a_ = tf.expand_dims(adj, axis=-1)
    x_ = tf.expand_dims(batch_x, axis=1)  # note the location of expanding
    selected = a_ * x_
    x_shape = tf.shape(batch_x)
    return tf.reshape(selected, (x_shape[0], x_shape[1], -1))


def test_neighbour_feats():
    a = np.random.randint(0, 2, size=(3, 3))
    a += a.T
    a[a>1] = 1
    bx = np.random.randint(1, 10, size=(2, 3, 2))
    x = bx[0]
    assert np.array_equal(neighbour_feats(a,x), np.stack([a*x[:,i] for i in range(2)], axis=-1).reshape(3, -1))
    assert np.array_equal(neighbour_feats3d(a,bx), np.stack([neighbour_feats(a, bx[i]) for i in range(2)], axis=0).reshape(2, 3, -1))


def neighbour_feats_sum(adj, X):
    # compatiable with batched X
    return np.matmul(adj, X)


def tf_neighbour_feats_sum(adj, X):
    # compatiable with batched X
    return adj@X

func_dict = dict(
    nbf_sum = neighbour_feats_sum,
    nbf_concat = neighbour_feats,
    nbf_concat3d = neighbour_feats3d,

    tf_nbf_sum = tf_neighbour_feats_sum,
    tf_nbf_concat = tf_neighbour_feats,
    tf_nbf_concat3d = tf_neighbour_feats3d
)


def get_nbf_op(name, is_tf=False):
    func_name = 'tf_nbf_' + name if is_tf else 'nbf_'+name
    func = func_dict.get(func_name, None)
    if func is None:
        raise NotImplementedError('required func {}(is_tf={}) is not implemented'.format(name, is_tf))
    return func
