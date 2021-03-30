import os
import joblib
import pickle
import tensorflow as tf
from baselines.common.tf_util import get_session
from baselines.common.models import register
import argparse
import numpy as np
from baselines.a2c.utils import fc

@register("custom_mlp")
def custom_mlp():
    def network_fn(X):
        h = tf.layers.flatten(X)
        for i in range(2):
            if i == 0:
                num_hidden = 64
            else:
                num_hidden = 32
            h = fc(h, 'mlp_fc{}'.format(i), nh=num_hidden, init_scale=np.sqrt(2))
            # tanh activation function
            h = tf.nn.relu(h)
        return h
    return network_fn

def var_shape(x):
    out = x.get_shape().as_list()
    assert all(isinstance(a, int) for a in out), \
        "shape function assumes that shape is fully known"
    return out

def intprod(x):
    return int(np.prod(x))

class SetFromFlat(object):
    def __init__(self, var_list, sess, dtype=tf.float32):
        self.sess = sess
        shapes = list(map(var_shape, var_list))
        total_size = np.sum([intprod(shape) for shape in shapes])

        self.theta = theta = tf.placeholder(dtype, [total_size])
        start = 0
        assigns = []
        for (shape, v) in zip(shapes, var_list):
            size = intprod(shape)
            assigns.append(tf.assign(v, tf.reshape(theta[start:start + size], shape)))
            start += size
        self.op = tf.group(*assigns)

    def __call__(self, theta):
        self.sess.run(self.op, feed_dict={self.theta: theta})

def setFromFlat(var_list, flat_params, sess=None):
    shapes = list(map(lambda x: x.get_shape().as_list(), var_list))
    total_size = np.sum([int(np.prod(shape)) for shape in shapes])
    if total_size != flat_params.shape[0]:
        redundant = flat_params.shape[0] - total_size
        flat_params = flat_params[redundant:]
        assert flat_params.shape[0] == total_size, \
            print('Number of variables does not match when loading pretrained victim agents.')
    theta = tf.placeholder(tf.float32, [total_size])
    start = 0
    assigns = []
    for (shape, v) in zip(shapes, var_list):
        size = int(np.prod(shape))
        assigns.append(tf.assign(v, tf.reshape(theta[start:start + size], shape)))
        start += size
    op = tf.group(*assigns)
    if sess == None:
        tf.get_default_session().run(op, {theta: flat_params})
    else:
        sess.run(op, {theta: flat_params})

def load_from_file(param_pkl_path):
    with open(param_pkl_path, 'rb') as f:
        params = pickle.load(f)
    return params

def save_trainable_variables(save_path, variables=None, scope=None, sess=None):
    sess = sess or get_session()
    variables = variables or tf.trainable_variables(scope)
    ps = sess.run(variables)
    save_dict = {v.name.replace(v.name.split('/')[0], ''): value for v, value in zip(variables, ps)}
    dirname = os.path.dirname(save_path)
    if any(dirname):
        os.makedirs(dirname, exist_ok=True)
    joblib.dump(save_dict, save_path)


def load_trainable_variables(load_path, variables=None, scope=None, sess=None):
    sess = sess or get_session()
    variables = variables or tf.trainable_variables(scope)

    loaded_params = joblib.load(os.path.expanduser(load_path))
    restores = []
    if isinstance(loaded_params, list):
        assert len(loaded_params) == len(variables), 'number of variables loaded mismatches len(variables)'
        for d, v in zip(loaded_params, variables):
            restores.append(v.assign(d))
    else:
        for v in variables:
            name = v.name.replace(v.name.split('/')[0], '')
            if name in loaded_params:
                restores.append(v.assign(loaded_params[name]))

    sess.run(restores)

# load variables
def load_variables(load_path, op=None, variables=None, scope=None):
    variables = variables or tf.trainable_variables(scope)
    loaded_params = joblib.load(os.path.expanduser(load_path))
    flat_param = []
    for v in variables:
        flat_param.append(loaded_params[v.name.replace(v.name.split('/')[0], '')].reshape(-1))
    flat_param = np.concatenate(flat_param, axis=0)
    op(flat_param)


def parse_args():
    parser = argparse.ArgumentParser()
    
    parser.add_argument("--total_timesteps", type=int, default=2)

    parser.add_argument("--out_dir", type=str, default=None)
    parser.add_argument("--n_steps", type=int, default=2048)
    parser.add_argument("--ent_coef", type=float, default=0.0)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--gamma", type=float, default=0.95)
    parser.add_argument("--nminibatches", type=int, default=4)
    parser.add_argument("--noptepochs", type=int, default=4)
    parser.add_argument("--nagents", type=int, default=2)
    parser.add_argument("--inner_loop_party_0", type=int, default=1)
    parser.add_argument("--inner_loop_party_1", type=int, default=1)
    parser.add_argument("--player_n", type=int, default=0)
    parser.add_argument("--win_info_file", type=str, default=None)
    parser.add_argument("--load_pretrain", type=int, default=0)
    parser.add_argument("--server_id", type=int, default=0)
    parser.add_argument("--pretrain_model_path", type=str, default=None)

    ## args for adv-train
    parser.add_argument("--adv_party", type=int, default=1)
    parser.add_argument("--opp_model_path", type=str, default=None)
    parser.add_argument("--mimic_model_path", type=str, default=None)
    return parser.parse_args()
