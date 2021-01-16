import gym
import copy
import pickle
import numpy as np
import tensorflow as tf
from abc import ABC, abstractmethod
from zoo_utils import remove_prefix


def load_victim_agent(env_name, ob_space, action_space, model_path):
    # load victim agent

    sess = tf.get_default_session()
    if sess is None:
        tf_config = tf.ConfigProto(inter_op_parallelism_threads=1, intra_op_parallelism_threads=1)
        sess = tf.Session(config=tf_config)
        sess.__enter__()
    #     print('=============================')
    #     print(sess)
    # print('=============================')
    # print(sess)
    # print('=============================')
    use_mlp = False
    if env_name == 'multicomp/YouShallNotPassHumans-v0':
        victim_agent = MlpPolicyValue(scope="mlp_policy", reuse=tf.AUTO_REUSE,
                                      ob_space=ob_space,
                                      ac_space=action_space,
                                      sess=sess,
                                      hiddens=[64, 64],  normalize=False)
        use_mlp = True
    else:
        victim_agent = LSTMPolicy(scope="lstm_policy", reuse=tf.AUTO_REUSE,
                                  ob_space=ob_space,
                                  ac_space=action_space,
                                  sess=sess,
                                  hiddens=[128, 128], normalize=False)

    sess.run(tf.variables_initializer(victim_agent.get_variables()))

    # load weights into victim_agent
    model = pickle.load(open(model_path, 'rb'))

    if env_name in ['multicomp/YouShallNotPassHumans-v0', 'multicomp/KickAndDefend-v0']:
       model = remove_prefix(model)

    flat_params = []

    for i in victim_agent.get_variables():
        name = i.name
        name = name[len(name.split('/')[0]) + 1:]
        key = '/' + name.split(':')[0]
        if not use_mlp:
            if 'weights' in key:
                key = key.replace('weights', 'kernel')
            elif 'bias' in key:
                key = key.replace('biases', 'bias')
        else:
            if 'w' in key:
                key = key.replace('w', 'kernel')
            elif 'b' in key:
                key = key.replace('b', 'bias')

        if 'basic_lstm_cell/bias:0' in name:
            bias = model[key]
            v = np.hstack([bias[0:128], bias[256:384], bias[128:256], bias[384:512]])

        elif 'basic_lstm_cell/kernel:0' in name:
            kernel = model[key]
            key = key.replace('kernel', 'recurrent_kernel')
            lstm_kernel = model[key]
            v1 = np.hstack([kernel[:, 0:128], kernel[:, 256:384], kernel[:, 128:256], kernel[:, 384:512]])
            v2 = np.hstack([lstm_kernel[:, 0:128], lstm_kernel[:, 256:384], lstm_kernel[:, 128:256], lstm_kernel[:, 384:512]])
            v = np.vstack([v1, v2])
        else:
            v = model[key]
        flat_params.append(v.reshape(-1))
    flat_params = np.concatenate(flat_params, axis=0)
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


class Policy(object):
    def reset(self, **kwargs):
        pass

    def act(self, observation):
        # should return act, info
        raise NotImplementedError()

    @property
    def value_flat(self):
        return self.vpred

    @property
    def obs_ph(self):
        return self.observation_ph

    @abstractmethod
    def step(self, obs, state=None, mask=None):
        """
        Returns the policy for a single step
        :param obs: ([float] or [int]) The current observation of the environment
        :param state: ([float]) The last states (used in recurrent policies)
        :param mask: ([float]) The last masks (used in recurrent policies)
        :return: ([float], [float], [float], [float]) actions, values, states, neglogp
        """
        raise NotImplementedError

    @abstractmethod
    def proba_step(self, obs, state=None, mask=None):
        """
        Returns the action probability for a single step
        :param obs: ([float] or [int]) The current observation of the environment
        :param state: ([float]) The last states (used in recurrent policies)
        :param mask: ([float]) The last masks (used in recurrent policies)
        :return: ([float]) the action probability
        """
        raise NotImplementedError


class RunningMeanStd(object):
    def __init__(self, scope="running", reuse=False, epsilon=1e-2, shape=()):
        with tf.variable_scope(scope, reuse=reuse):
            self._sum = tf.get_variable(
                dtype=tf.float32,
                shape=shape,
                initializer=tf.constant_initializer(0.0),
                name="sum", trainable=False)
            self._sumsq = tf.get_variable(
                dtype=tf.float32,
                shape=shape,
                initializer=tf.constant_initializer(epsilon),
                name="sumsq", trainable=False)
            self._count = tf.get_variable(
                dtype=tf.float32,
                shape=(),
                initializer=tf.constant_initializer(epsilon),
                name="count", trainable=False)
            self.shape = shape

            self.mean = tf.to_float(self._sum / self._count)
            var_est = tf.to_float(self._sumsq / self._count) - tf.square(self.mean)
            self.std = tf.sqrt(tf.maximum(var_est, 1e-2))


def dense(x, size, name, weight_init=None, bias=True):
    w = tf.get_variable(name + "/w", [x.get_shape()[1], size], initializer=weight_init)
    ret = tf.matmul(x, w)
    if bias:
        b = tf.get_variable(name + "/b", [size], initializer=tf.zeros_initializer())
        return ret + b
    else:
        return ret


def switch(condition, if_exp, else_exp):
    x_shape = copy.copy(if_exp.get_shape())
    x = tf.cond(tf.cast(condition, 'bool'),
                lambda: if_exp,
                lambda: else_exp)
    x.set_shape(x_shape)
    return x


class DiagonalGaussian(object):
    def __init__(self, mean, logstd):
        self.mean = mean
        self.logstd = logstd
        self.std = tf.exp(logstd)

    def sample(self):
        return self.mean + self.std * tf.random_normal(tf.shape(self.mean))

    def mode(self):
        return self.mean

    def neglogp(self, x):
        return 0.5 * tf.reduce_sum(tf.square((x - self.mean) / self.std), axis=-1) \
               + 0.5 * np.log(2.0 * np.pi) * tf.cast(tf.shape(x)[-1], tf.float32) \
               + tf.reduce_sum(self.logstd, axis=-1)

    def kl(self, other):
        return tf.reduce_sum(other.logstd - self.logstd + (tf.square(self.std) + tf.square(self.mean - other.mean)) /
                             (2.0 * tf.square(other.std)) - 0.5, axis=-1)

    def entropy(self):
        return tf.reduce_sum(self.logstd + .5 * np.log(2.0 * np.pi * np.e), axis=-1)


# observation is normalized in the MlpPolicy, the value function's output is also normalized
# when param 'normalize' is set to 'ob', the value function's output will be normalized
class MlpPolicyValue(Policy):
    def __init__(self, scope, *, ob_space, ac_space, hiddens, rate=0.0, convs=[], n_batch_train=1,
                 sess=None, reuse=False, normalize=False):
        self.sess = sess
        self.recurrent = False
        self.normalized = normalize
        self.zero_state = np.zeros(1)
        with tf.variable_scope(scope, reuse=reuse):
            self.scope = tf.get_variable_scope().name

            assert isinstance(ob_space, gym.spaces.Box)

            self.observation_ph = tf.placeholder(tf.float32, [None] + list(ob_space.shape), name="observation")
            self.stochastic_ph = tf.placeholder(tf.bool, (), name="stochastic")
            self.taken_action_ph = tf.placeholder(dtype=tf.float32, shape=[None, ac_space.shape[0]], name="taken_action")

            if self.normalized:
                if self.normalized != 'ob':
                    self.ret_rms = RunningMeanStd(scope="retfilter")
                self.ob_rms = RunningMeanStd(shape=ob_space.shape, scope="obsfilter")

            obz = self.observation_ph
            if self.normalized:
                obz = tf.clip_by_value((self.observation_ph - self.ob_rms.mean) / self.ob_rms.std, -5.0, 5.0)

            last_out = obz
            for i, hid_size in enumerate(hiddens):
                last_out = tf.nn.tanh(dense(last_out, hid_size, "vffc%i" % (i + 1)))

            self.vpredz = dense(last_out, 1, "vffinal")[:, 0]

            self.vpred = self.vpredz
            # reverse normalization. because the reward is normalized, reversing it to see the real value.

            if self.normalized and self.normalized != 'ob':
                self.vpred = self.vpredz * self.ret_rms.std + self.ret_rms.mean
            last_out = obz
            for i, hid_size in enumerate(hiddens):
                last_out = tf.nn.tanh(dense(last_out, hid_size, "polfc%i" % (i + 1)))
                last_out = tf.nn.dropout(last_out, rate=rate)
            mean = dense(last_out, ac_space.shape[0], "polfinal")
            logstd = tf.get_variable(name="logstd", shape=[n_batch_train, ac_space.shape[0]],
                                     initializer=tf.zeros_initializer())

            self.pd = DiagonalGaussian(mean, logstd)
            self.proba_distribution = self.pd
            self.sampled_action = switch(self.stochastic_ph, self.pd.sample(), self.pd.mode())
            self.neglogp = self.proba_distribution.neglogp(self.sampled_action)
            self.policy_proba = [self.proba_distribution.mean, self.proba_distribution.std]

    def make_feed_dict(self, observation, taken_action):
        return {
            self.observation_ph: observation,
            self.taken_action_ph: taken_action
        }

    def act(self, observation, stochastic=True):
        outputs = [self.sampled_action, self.vpred]
        if self.sess==None:
            a, v = tf.get_default_session().run(outputs, {self.observation_ph: observation[None],
                                                          self.stochastic_ph: stochastic})
        else:
            a, v = self.sess.run(outputs, {self.observation_ph: observation[None],
                                           self.stochastic_ph: stochastic})
        return a[0], {'vpred': v[0]}

    def get_variables(self):
        return tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, self.scope)

    def get_trainable_variables(self):
        return tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, self.scope)

    @property
    def initial_state(self):
        return None

    def step(self, obs, state=None, mask=None, deterministic=False):
        stochastic = not deterministic
        if self.sess==None:
            action, value, neglogp = tf.get_default_session().run([self.sampled_action, self.value_flat, self.neglogp],
                                                   {self.obs_ph: obs, self.stochastic_ph: stochastic})
        else:
            action, value, neglogp = self.sess.run([self.sampled_action, self.value_flat, self.neglogp],
                                              {self.obs_ph: obs, self.stochastic_ph: stochastic})
        return action, value, self.initial_state, neglogp

    def proba_step(self, obs, state=None, mask=None):
        if self.sess==None:
            return tf.get_default_session().run(self.policy_proba, {self.obs_ph: obs})
        else:
            return self.sess.run(self.policy_proba, {self.obs_ph: obs})

    def value(self, obs, state=None, mask=None):
        if self.sess==None:
            return tf.get_default_session().run(self.value_flat, {self.obs_ph: obs})
        else:
            return self.sess.run(self.value_flat, {self.obs_ph: obs})


# observation is normalized in the LSTM policy, the value function's output is also normalized
# when param 'normalize' is set to 'ob', the value function's output will be normalized
class LSTMPolicy(Policy):
    def __init__(self, scope, *, ob_space, ac_space, hiddens, n_batch_train=1,
                 n_envs=1, sess=None, reuse=False, normalize=False):
        self.sess = sess
        self.recurrent = True
        self.normalized = normalize
        self.n_envs = n_envs
        with tf.variable_scope(scope, reuse=reuse):
            self.scope = tf.get_variable_scope().name

            assert isinstance(ob_space, gym.spaces.Box)

            self.observation_ph = tf.placeholder(tf.float32, [None, None] + list(ob_space.shape), name="observation")
            self.stochastic_ph = tf.placeholder(tf.bool, (), name="stochastic")
            self.taken_action_ph = tf.placeholder(dtype=tf.float32, shape=[None, None, ac_space.shape[0]], name="taken_action")
            self.dones_ph = tf.placeholder(dtype=tf.float32, shape=[None, 1], name="done_ph")

            if self.normalized:
                if self.normalized != 'ob':
                    self.ret_rms = RunningMeanStd(scope="retfilter")
                self.ob_rms = RunningMeanStd(shape=ob_space.shape, scope="obsfilter")

            obz = self.observation_ph
            if self.normalized:
                obz = tf.clip_by_value((self.observation_ph - self.ob_rms.mean) / self.ob_rms.std, -5.0, 5.0)

            last_out = obz
            for hidden in hiddens[:-1]:
                last_out = tf.contrib.layers.fully_connected(last_out, hidden)
            self.zero_state = []
            self.state_in_ph = []
            self.state_out = []
            cell = tf.contrib.rnn.BasicLSTMCell(hiddens[-1], forget_bias=0.0, reuse=reuse)
            size = cell.state_size
            self.zero_state.append(np.zeros(size.c, dtype=np.float32))
            self.zero_state.append(np.zeros(size.h, dtype=np.float32))
            self.state_in_ph.append(tf.placeholder(tf.float32, [None, size.c], name="lstmv_c"))
            self.state_in_ph.append(tf.placeholder(tf.float32, [None, size.h], name="lstmv_h"))
            self.initial_state_1 = tf.contrib.rnn.LSTMStateTuple(self.state_in_ph[-2] * (1-self.dones_ph),
                                                                 self.state_in_ph[-1]* (1-self.dones_ph))
            last_out, state_out = tf.nn.dynamic_rnn(cell, last_out, initial_state=self.initial_state_1, scope="lstmv")
            self.state_out.append(state_out)

            self.vpredz = tf.contrib.layers.fully_connected(last_out, 1, activation_fn=None)[:, :, 0]
            self.vpred = self.vpredz
            if self.normalized and self.normalized != 'ob':
                self.vpred = self.vpredz * self.ret_rms.std + self.ret_rms.mean  # raw = not standardized

            last_out = obz
            for hidden in hiddens[:-1]:
                last_out = tf.contrib.layers.fully_connected(last_out, hidden)
            cell = tf.contrib.rnn.BasicLSTMCell(hiddens[-1], forget_bias=0.0, reuse=reuse)
            size = cell.state_size
            self.zero_state.append(np.zeros(size.c, dtype=np.float32))
            self.zero_state.append(np.zeros(size.h, dtype=np.float32))
            self.state_in_ph.append(tf.placeholder(tf.float32, [None, size.c], name="lstmp_c"))
            self.state_in_ph.append(tf.placeholder(tf.float32, [None, size.h], name="lstmp_h"))
            self.initial_state_1 = tf.contrib.rnn.LSTMStateTuple(self.state_in_ph[-2] * (1-self.dones_ph),
                                                                 self.state_in_ph[-1]* (1-self.dones_ph))
            last_out, state_out = tf.nn.dynamic_rnn(cell, last_out, initial_state=self.initial_state_1, scope="lstmp")
            self.state_out.append(state_out)
            self.mean = tf.contrib.layers.fully_connected(last_out, ac_space.shape[0], activation_fn=None)
            logstd = tf.get_variable(name="logstd", shape=[1, ac_space.shape[0]], initializer=tf.zeros_initializer())

            if reuse:
                self.pd_mean = tf.reshape(self.mean, (n_batch_train, ac_space.shape[0]))
            else:
                self.pd_mean = tf.reshape(self.mean, (n_envs, ac_space.shape[0]))
            self.pd = DiagonalGaussian(self.pd_mean, logstd)
            self.proba_distribution = self.pd
            self.sampled_action = switch(self.stochastic_ph, self.pd.sample(), self.pd.mode())
            self.neglogp = self.proba_distribution.neglogp(self.sampled_action)
            self.policy_proba = [self.proba_distribution.mean, self.proba_distribution.std]

            self.zero_state = np.array(self.zero_state)
            self.state_in_ph = tuple(self.state_in_ph)
            self.state = self.zero_state

            for p in self.get_trainable_variables():
                tf.add_to_collection(tf.GraphKeys.REGULARIZATION_LOSSES, tf.reduce_sum(tf.square(p)))

    def make_feed_dict(self, observation, state_in, taken_action):
        return {
            self.observation_ph: observation,
            self.state_in_ph: list(np.transpose(state_in, (1, 0, 2))),
            self.taken_action_ph: taken_action
        }

    def act(self, observation, stochastic=True):
        outputs = [self.sampled_action, self.vpred, self.state_out]
        # design for the pre_state
        # notice the zero state
        if self.sess == None:
            a, v, s = tf.get_default_session().run(outputs, {
                self.observation_ph: observation[None, None],
                self.state_in_ph: list(self.state[:, None, :]),
                self.stochastic_ph: stochastic,
                self.dones_ph:np.zeros(self.state[0, None, 0].shape)[:,None]})
        else:
            a, v, s = self.sess.run(outputs, {
                self.observation_ph: observation[None, None],
                self.state_in_ph: list(self.state[:, None, :]),
                self.stochastic_ph: stochastic,
                self.dones_ph: np.zeros(self.state[0, None, 0].shape)[:, None]})
        self.state = []
        for x in s:
            self.state.append(x.c[0])
            self.state.append(x.h[0])
        self.state = np.array(self.state)

        # finish checking.
        return a[0, ], {'vpred': v[0, 0], 'state': self.state}

    def get_variables(self):
        return tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, self.scope)

    def get_trainable_variables(self):
        return tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, self.scope)

    def reset(self):
        self.state = self.zero_state

    @property
    def initial_state(self):
        initial_state_shape = []
        for i in range(4):
            initial_state_shape.append(np.repeat(self.zero_state[i][None,], self.n_envs, axis=0))
        self._initial_state = np.array(initial_state_shape)
        return self._initial_state

    def step(self, obs, state=None, mask=None, deterministic=False):
        stochastic = not deterministic
        if mask is not None:
            mask = np.array(mask)[:, None]
        if self.sess==None:
            action, value, state, neglogp = tf.get_default_session().run([self.sampled_action, self.value_flat, self.state_out, self.neglogp],
                                                                         {self.obs_ph: obs[:, None, :], self.state_in_ph: list(state), self.dones_ph: mask,
                                                                          self.stochastic_ph: stochastic})
        else:
            action, value, state, neglogp = self.sess.run([self.sampled_action, self.value_flat, self.state_out, self.neglogp],
                                                          {self.obs_ph: obs[:, None, :], self.state_in_ph: list(state), self.dones_ph: mask,
                                                           self.stochastic_ph: stochastic})
        value = value[:, 0]
        state_np = []
        for state_tmp in state:
            for state_tmp_1 in state_tmp:
                state_np.append(state_tmp_1)

        return action, value, np.array(state_np), neglogp

    def proba_step(self, obs, state=None, mask=None):
        if mask is not None:
            mask = np.array(mask)[:, None]
        if self.sess==None:
            return tf.get_default_session().run(self.policy_proba, {self.obs_ph: obs, self.state_in_ph: state,
                                                                    self.dones_ph: mask})
        else:
            return self.sess.run(self.policy_proba, {self.obs_ph: obs[:, None, :], self.state_in_ph: list(state),
                                                     self.dones_ph: mask})

    def value(self, obs, state=None, mask=None):
        if mask is not None:
            mask = np.array(mask)[:, None]
        if self.sess==None:
            return tf.get_default_session().run(self.value_flat, {self.obs_ph: obs[:, None, :],
                                                                  self.state_in_ph: list(state),
                                                                  self.dones_ph: mask})[:,0]
        else:
            return self.sess.run(self.value_flat, {self.obs_ph: obs[:, None, :],
                                                   self.state_in_ph: list(state),
                                                   self.dones_ph: mask})[:,0]
