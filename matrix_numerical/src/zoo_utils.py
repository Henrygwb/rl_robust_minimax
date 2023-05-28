import os
import joblib
import pickle
import tensorflow as tf
import numpy as np
from baselines.common import tf_util
from baselines.common.distributions import make_pdtype
from baselines.common.input import observation_placeholder
from baselines.common.tf_util import adjust_shape, get_session


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
            restores.append(v.assign(loaded_params[v.name.replace(v.name.split('/')[0], '')]))

    sess.run(restores)

def init_trainable_variables(variables=None, scope=None, sess=None):
    sess = sess or get_session()
    variables = variables or tf.trainable_variables(scope)
    restores = []
    loaded_params = [np.array([3.5]).reshape(-1, 1)]

    restores = []
    if isinstance(loaded_params, list):
        for d, v in zip(loaded_params, variables):
            restores.append(v.assign(d))
    else:
        for v in variables:
            restores.append(v.assign(loaded_params[v.name.replace(v.name.split('/')[0], '')]))
    sess.run(restores)



class PolicyWithValue(object):

    def __init__(self, env, observations, latent, vf_latent, sess=None):

        self.X = observations
        self.state = tf.constant([])
        self.initial_state = None
        latent = tf.layers.flatten(latent)
        vf_latent = tf.layers.flatten(vf_latent)

        # Select the policy distribution based on the action space
        # Discrete: Categorical distribution (Bernoulli).
        # Continuous: DiagGaussian.
        self.pdtype = make_pdtype(env.action_space)

        # Policy net is used to estimate the mean of the policy distribution.
        self.pd, self.pi = self.pdtype.pdfromlatent(latent, init_scale=0.01)

        # Sample an action
        self.action = self.pd.sample()

        # Calculate the neg loglikelihood of the current action.
        self.neglogp = self.pd.neglogp(self.action)
        self.sess = sess or tf.get_default_session()

        self.vf = vf_latent[:, 0]

    def _evaluate(self, variables, observation, **extra_feed):
        """
        Given an observation, get the interested variable value.
        """
        sess = self.sess
        feed_dict = {self.X: adjust_shape(self.X, observation)}
        for inpt_name, data in extra_feed.items():
            if inpt_name in self.__dict__.keys():
                inpt = self.__dict__[inpt_name]
                if isinstance(inpt, tf.Tensor) and inpt._op.type == 'Placeholder':
                    feed_dict[inpt] = adjust_shape(inpt, data)

        return sess.run(variables, feed_dict)

    def step(self, observation, **extra_feed):
        """
        Compute the next action given the current observation
        :param observation: observation.
        :param **extra_feed: additional data such as state or mask.
        returns:
        action, estimated value, next state, negative log likelihood of the current action.
        """

        a, v, state, neglogp = self._evaluate([self.action, self.vf, self.state, self.neglogp], observation, **extra_feed)
        if state.size == 0:
            state = None
        return a, v, state, neglogp

    def value(self, ob, *args, **kwargs):
        """
        Compute the current value function of the given observation.
        :param observation: observation.
        returns: estimated value .
        """

        return self._evaluate(self.vf, ob, *args, **kwargs)

    def save(self, save_path):
        tf_util.save_state(save_path, sess=self.sess)

    def load(self, load_path):
        tf_util.load_state(load_path, sess=self.sess)


def build_policy(env, env_name, **policy_kwargs):

    def policy_fn(nbatch=None, sess=None, observ_placeholder=None):

        ob_space = env.observation_space
        X = observ_placeholder if observ_placeholder is not None else observation_placeholder(ob_space, batch_size=nbatch)

        with tf.variable_scope('pi', reuse=tf.AUTO_REUSE):
            if env_name != 'Match_Pennies' and env_name != 'As_Match_Pennies':
               latent = tf.get_variable(name="police", shape=[1, 1]) # mean of x.
            else:
               latent = tf.get_variable(name="police", shape=[1, 2]) # probability of two players choosing one action,
            latent = tf.tile(latent, [nbatch, 1])

        with tf.variable_scope('vf', reuse=tf.AUTO_REUSE):
            vf_latent = tf.get_variable(name="value", shape=[1, 1])
            vf_latent = tf.tile(vf_latent, [nbatch, 1])

        policy = PolicyWithValue(env=env, observations=X, latent=latent, vf_latent=vf_latent, sess=sess)

        return policy

    return policy_fn