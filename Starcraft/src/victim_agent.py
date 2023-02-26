import gym
import copy
import pickle
import numpy as np
import tensorflow as tf
from abc import ABC, abstractmethod
from zoo_utils import remove_prefix, pkl_to_joblib
from gym.spaces import Discrete

def load_victim_agent(ob_space, action_space, model_path):

    sess = tf.get_default_session()
    if sess is None:
        tf_config = tf.ConfigProto(inter_op_parallelism_threads=1, intra_op_parallelism_threads=1)
        sess = tf.Session(config=tf_config)
        sess.__enter__()

    victim_agent = MlpPolicy(scope_name="mlp_policy", reuse=tf.AUTO_REUSE,
                             ob_space=ob_space, nbatch=1, nsteps=1,
                             ac_space=action_space, sess=sess)
    sess.run(tf.variables_initializer(victim_agent.get_variables()))
    # Load params
    params = pkl_to_joblib(model_path)
    flat_params = np.concatenate(params, axis=0)
    setFromFlat(victim_agent.get_variables(), flat_params, sess)
    return victim_agent


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

class Pd(object):

    def neglogp(self, x):
        raise NotImplementedError

    def entropy(self):
        raise NotImplementedError

    def sample(self):
        raise NotImplementedError

    @classmethod
    def fromlogits(cls, logits):
        return cls(logits)

class CategoricalPd(Pd):

    def __init__(self, logits):
        self.logits = logits

    def neglogp(self, x):
        one_hot_actions = tf.one_hot(x, self.logits.get_shape().as_list()[-1])
        return tf.nn.softmax_cross_entropy_with_logits(logits=self.logits,
                                                       labels=one_hot_actions)
    def entropy(self):
        a = self.logits - tf.reduce_max(self.logits, axis=-1, keep_dims=True)
        ea = tf.exp(a)
        z = tf.reduce_sum(ea, axis=-1, keep_dims=True)
        p = ea / z
        return tf.reduce_sum(p * (tf.log(z) - a), axis=-1)

    def sample(self):
        """
        sampling from concrete distribution.
        :return: samples
        """
        u = tf.random_uniform(tf.shape(self.logits))
        return tf.argmax(self.logits - tf.log(-tf.log(u)), axis=-1)


def fc(x, scope, nh, init_scale=1.0, init_bias=0.0):
    with tf.variable_scope(scope):
        nin = x.get_shape()[1].value
        w = tf.get_variable("w", [nin, nh], initializer=ortho_init(init_scale))
        b = tf.get_variable("b", [nh],
                            initializer=tf.constant_initializer(init_bias))
        return tf.matmul(x, w) + b


def ortho_init(scale=1.0):

    def _ortho_init(shape, dtype, partition_info=None):
        shape = tuple(shape)
        if len(shape) == 2: flat_shape = shape
        elif len(shape) == 4: flat_shape = (np.prod(shape[:-1]), shape[-1]) #NHWC
        else: raise NotImplementedError
        a = np.random.normal(0.0, 1.0, flat_shape)
        u, _, v = np.linalg.svd(a, full_matrices=False)
        q = u if u.shape == flat_shape else v
        q = q.reshape(shape)
        return (scale * q[:shape[0], :shape[1]]).astype(np.float32)

    return _ortho_init


class MlpPolicy(object):
    # what is a mask discrete???
    def __init__(self, sess, scope_name, ob_space, ac_space, nbatch, nsteps,
                 reuse=False):

        mask_space = ob_space['action_mask']
        _ob_space = ob_space['obs']

        X = tf.placeholder(
            shape=(nbatch,) + _ob_space.shape, dtype=tf.float32, name="x_screen")
        if isinstance(ac_space, Discrete):
            MASK = tf.placeholder(
                shape=(nbatch,) + mask_space.shape, dtype=tf.float32, name="mask")

        ff_out = []

        with tf.variable_scope(scope_name, reuse=reuse):
            x = tf.layers.flatten(X)
            pi_h1 = tf.tanh(fc(x, 'pi_fc1', nh=128, init_scale=np.sqrt(2)))
            ff_out.append(pi_h1)
            pi_h2 = tf.tanh(fc(pi_h1, 'pi_fc2', nh=128, init_scale=np.sqrt(2)))
            ff_out.append(pi_h2)
            pi_h3 = tf.tanh(fc(pi_h2, 'pi_fc3', nh=128, init_scale=np.sqrt(2)))
            ff_out.append(pi_h3)

            vf_h1 = tf.tanh(fc(x, 'vf_fc1', nh=128, init_scale=np.sqrt(2)))
            vf_h2 = tf.tanh(fc(vf_h1, 'vf_fc2', nh=128, init_scale=np.sqrt(2)))
            vf_h3 = tf.tanh(fc(vf_h2, 'vf_fc3', nh=128, init_scale=np.sqrt(2)))
            vf = fc(vf_h3, 'vf', 1)[:,0]
            pi_logit = fc(pi_h3, 'pi', ac_space.n, init_scale=0.01, init_bias=0.0)
            if isinstance(ac_space, Discrete):
                # Discrete: only used unmasked actions.
                # prevent from sampling masked actions.
                pi_logit -= (1 - MASK) * 1e30
            self.pd = CategoricalPd(pi_logit)

        self.policy_ff_acts = tf.concat(ff_out, axis=-1)
        action = self.pd.sample()
        neglogp = self.pd.neglogp(action)
        self.initial_state = None

        def step(ob, *_args, **_kwargs):
            if len(_args) != 0:
                if isinstance(ac_space, Discrete):
                    a, v, nl = sess.run([action, vf, neglogp], {X:ob[0], MASK:ob[-1]})
                else:
                    a, v, nl = sess.run([action, vf, neglogp], {X:ob})
                return a, v, self.initial_state, nl
            else:
                if isinstance(ac_space, Discrete):
                    a, ff_acts = sess.run([action, self.policy_ff_acts], {X:ob[0], MASK:ob[-1]})
                else:
                    a, ff_acts = sess.run([action, self.policy_ff_acts], {X:ob})
                return a, ff_acts

        def value(ob, *_args, **_kwargs):
            if isinstance(ac_space, Discrete):
                return sess.run(vf, {X:ob[0], MASK:ob[-1]})
            else:
                return sess.run(vf, {X:ob})

        self.X = X
        if isinstance(ac_space, Discrete):
            self.MASK = MASK
        self.vf = vf
        self.step = step
        self.value = value
        self.scope = scope_name
    
    def get_variables(self):
        return tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, self.scope)