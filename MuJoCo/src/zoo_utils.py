import os
import datetime
import pickle
import numpy as np
from ray.rllib.utils import try_import_tf
from ray.rllib.models.modelv2 import ModelV2
from ray.rllib.models.tf.recurrent_net import RecurrentNetwork
from ray.rllib.models.tf.fcnet import FullyConnectedNetwork
from ray.rllib.models.tf.misc import normc_initializer
from ray.rllib.utils.filter import MeanStdFilter, RunningStat
from ray.rllib.utils.annotations import override
tf1, tf, tfv = try_import_tf()


def make_timestamp():
    ISO_TIMESTAMP = "%Y%m%d_%H%M%S"
    return datetime.datetime.now().strftime(ISO_TIMESTAMP)


def setup_logger(out_dir='results', exp_name='test'):
    timestamp = make_timestamp()
    exp_name = exp_name.replace('/', '_')  # environment names can contain /'s
    out_dir = os.path.join(out_dir, '{}-{}'.format(timestamp, exp_name))
    os.makedirs(out_dir, exist_ok=True)
    return out_dir


def create_mean_std(obs_norm_path):

    _, (obs_mean, obs_std, obs_count) = load_rms(obs_norm_path)

    # rs, buffer
    shape = obs_mean.shape
    rs = RunningStat(shape=shape)
    rs._n = obs_count
    rs._M[:] = obs_mean[:]
    rs._S[:] = np.power(obs_std[:], 2) * (obs_count - 1)

    mean_std_filter = MeanStdFilter(shape, demean=True, destd=True, clip=5.0) # normalize with clip.
    mean_std_filter.rs.update(rs) # Copy the parameters of rs to mean_std_filter.rs.

    return mean_std_filter


def load_model(file_name):
    pretrain_model = pickle.load(open(file_name, 'rb'))
    model = {}
    for k, v in pretrain_model.items():
        model['default_policy/'+k] = v
    return model


def load_pretrain_model(file_name):
    pretrain_model = pickle.load(open(file_name, 'rb'))
    model = {}
    opp_model = {}

    # Change the keys to the model parameter names in the LSTM class.
    dic_new_name = {}
    for key in list(pretrain_model.keys()):
        key_1 = key.split(':')[0]
        dic_new_name[key] = key_1
    pretrain_model_newname = dict((dic_new_name[key], value) for (key, value) in pretrain_model.items())

    for k, v in pretrain_model_newname.items():
        model['model/'+k] = v
        opp_model['opp_model/'+k] = v
    return model, opp_model


def load_rms(norm_path):
    rms_params = pickle.load(open(norm_path, 'rb'))
    reward_mean = rms_params['retfilter/mean']
    reward_std = rms_params['retfilter/std']
    reward_count = rms_params['retfilter/count:0']

    obs_mean = rms_params['obsfilter/mean']
    obs_std = rms_params['obsfilter/std']
    obs_count = rms_params['obsfilter/count:0']
    return (reward_mean, reward_std, reward_count), (obs_mean, obs_std, obs_count),


def add_prefix(key_val, prefix):
    ret = {}
    for k, v in key_val.items():
        name = prefix + k 
        ret[name] = v
    return ret 


def remove_prefix(key_val):
    ret = {}
    for k, v in key_val.items():
        name = k.replace(k.split('/')[0], '')
        ret[name] = v
    return ret


# No normalization in the model. Both the reward and observation have been normalized in the env.
class LSTM(RecurrentNetwork):

    def __init__(self, obs_space, action_space, num_outputs, model_config, name):
        super(LSTM, self).__init__(obs_space, action_space, num_outputs,
                                       model_config, name)

        # note that in the original father class "RecurrentNetwork",
        # log_std is dependent on observation, not like the fashion in our rnn_newloss implementation in which log_std is a independent variable.
        # In such case, num_outputs = 2 * action_size.
        # To implement the free_log_std (free means independent), we need firstly reduce the num_outputs in half.

        # free_log_std 
        # independent of state
        num_outputs = num_outputs // 2

        hiddens = model_config.get('fcnet_hiddens')
        self.cell_size = model_config['lstm_cell_size']


        # obs_ph: (batch_size, length, obs_dim)
        obs_ph = tf.keras.layers.Input((None,) + obs_space.shape, name='observation') # todo: check obs_space.shape
    
        # Value network
        # FC -> LSTM -> FC 
        # FC
        i = 0
        last_out = obs_ph
        for size in hiddens:
            if i == 0:
                last_out = tf.keras.layers.Dense(
                    size,
                    name='fully_connected',
                    activation='relu',
                    kernel_initializer=normc_initializer(1.0))(last_out)
            else:
                last_out = tf.keras.layers.Dense(
                    size,
                    name='fully_connected_{}'.format(i),
                    activation='relu',
                    kernel_initializer=normc_initializer(1.0))(last_out)
            i += 1

        # last_out: (batch_size, length,  fcnet_hiddens)

        # LSTM
        # vstate_in_h: (batch_size, lstm_cell_size)
        # vstate_in_c: (batch_size, lstm_cell_size)
        vstate_in_h = tf.keras.layers.Input(shape=(self.cell_size,), name='v_h')
        vstate_in_c = tf.keras.layers.Input(shape=(self.cell_size,), name='v_c')

        # seq_in: (batch_size, )
        seq_in = tf.keras.layers.Input(shape=(), name='seq_in', dtype=tf.int32)

        # last_out: (batch_size, length, lstm_cell_size)
        # vstate_h: (batch_size, lstm_cell_size)
        # vstate_c: (batch_size, lstm_cell_size)
        last_out, vstate_h, vstate_c = tf.keras.layers.LSTM(
            self.cell_size, return_sequences=True, return_state=True, unit_forget_bias=False,
            name='lstmv/basic_lstm_cell')(
            inputs=last_out,
            mask=tf.sequence_mask(seq_in),
            initial_state=[vstate_in_h, vstate_in_c])
        # FC
        values = tf.keras.layers.Dense(1, activation=None, name='fully_connected_{}.format(i)')(last_out)
        i = i+1

        # Policy network
        # FC -> LSTM -> FC
        # FC
        last_out = obs_ph
        for size in hiddens:
            if i == 0:
                last_out = tf.keras.layers.Dense(
                    size,
                    name='fully_connected',
                    activation='relu',
                    kernel_initializer=normc_initializer(1.0))(last_out)
            else:
                last_out = tf.keras.layers.Dense(
                    size,
                    name='fully_connected_{}'.format(i),
                    activation='relu',
                    kernel_initializer=normc_initializer(1.0))(last_out)
            i += 1

        # last_out: (batch_size, length,  fcnet_hiddens)

        # LSTM
        # pstate_in_h: (batch_size, lstm_cell_size)
        # pstate_in_c: (batch_size, lstm_cell_size)
        pstate_in_h = tf.keras.layers.Input(shape=(self.cell_size,), name='p_h')
        pstate_in_c = tf.keras.layers.Input(shape=(self.cell_size,), name='p_c')


        # last_out: (batch_size, length, lstm_cell_size)
        # pstate_h: (batch_size, lstm_cell_size)
        # pstate_c: (batch_size, lstm_cell_size)
        last_out, pstate_h, pstate_c = tf.keras.layers.LSTM(
            self.cell_size, return_sequences=True, return_state=True, unit_forget_bias=False,
            name='lstmp/basic_lstm_cell')(
            inputs=last_out,
            mask=tf.sequence_mask(seq_in),
            initial_state=[pstate_in_h, pstate_in_c]) # todo check mask and return sequence, input and output shape
        # FC
        action = tf.keras.layers.Dense(
            num_outputs,
            activation=None,
            name='fully_connected_{}'.format(i))(last_out)

        self.log_std_var = tf.get_variable(
                shape=[1, num_outputs], dtype=tf.float32, name='log_std') # todo logstd shape.
        self.register_variables([self.log_std_var])

        def tiled_log_std(x):
            return tf.tile(
                tf.expand_dims(self.log_std_var, 0), [tf.shape(x)[0], tf.shape(x)[1], 1])

        log_std_out = tf.keras.layers.Lambda(tiled_log_std)(obs_ph)

        # action: (batch_size, length, action_size)
        # log_std_out: (batch_size, length, action_size)
        # Here, we need to concate the action and log_std_out
        action = tf.keras.layers.Concatenate(axis=2)(
            [action, log_std_out])

        inputs = [obs_ph, seq_in, vstate_in_h, vstate_in_c, pstate_in_h, pstate_in_c]
        outputs = [action, values, vstate_h, vstate_c, pstate_h, pstate_c]

        self.rnn_model = tf.keras.Model(
            inputs=inputs,
            outputs=outputs)

        self.register_variables(self.rnn_model.variables)

    @override(RecurrentNetwork)
    def forward_rnn(self, input_dict, state, seq_lens, prev_action=None):

        model_out, self._value_out, v_h, v_c, p_h, p_c = self.rnn_model([input_dict, seq_lens] + state)

        return model_out, [v_h, v_c, p_h, p_c]

    @override(ModelV2)
    def get_initial_state(self):
        return [
            np.zeros(self.cell_size, np.float32),
            np.zeros(self.cell_size, np.float32),
            np.zeros(self.cell_size, np.float32),
            np.zeros(self.cell_size, np.float32),
        ]

    @override(ModelV2)
    def value_function(self):
        return tf.reshape(self._value_out, [-1])


# No normalization in the model. Both the reward and observation have been normalized in the env.
class MLP(FullyConnectedNetwork):

    def __init__(self, obs_space, action_space, num_outputs, model_config, name):
        super(MLP, self).__init__(obs_space, action_space, num_outputs,
                                   model_config, name)

        # note that in the original father class "RecurrentNetwork",
        # log_std is dependent on observation, not like the fashion in our rnn_newloss implementation in which log_std is a independent variable.
        # In such case, num_outputs = 2 * action_size.
        # To implement the free_log_std (free means independent), we need firstly reduce the num_outputs in half.

        # free_log_std
        # independent of state
        num_outputs = num_outputs // 2
        hiddens = model_config.get('fcnet_hiddens')

        obs_ph = tf.keras.layers.Input(
            shape=(int(np.product(obs_space.shape)),), name="observations")

        # Value network
        # FC
        last_out = obs_ph
        for i, size in enumerate(hiddens):
            last_out = tf.keras.layers.Dense(
                size,
                name="vffc%i" % (i + 1),
                activation='tanh',
                kernel_initializer=normc_initializer(1.0))(last_out)

        # FC
        values = tf.keras.layers.Dense(1, activation=None, name='vffinal')(last_out)

        # Policy network
        # FC
        last_out = obs_ph
        for i, size in enumerate(hiddens):
            last_out = tf.keras.layers.Dense(
                size,
                name="polfc%i" % (i + 1),
                activation='tanh',
                kernel_initializer=normc_initializer(1.0))(last_out)

        action = tf.keras.layers.Dense(
            num_outputs,
            activation=None,
            name='polfinal')(last_out)

        self.log_std_var = tf.get_variable(
            shape=[1, num_outputs], dtype=tf.float32, name='log_std')  # todo logstd shape.
        self.register_variables([self.log_std_var])

        def tiled_log_std(x):
            return tf.tile(
                tf.expand_dims(self.log_std_var, 0), [tf.shape(x)[0], tf.shape(x)[1], 1])

        log_std_out = tf.keras.layers.Lambda(tiled_log_std)(obs_ph)

        action = tf.keras.layers.Concatenate(axis=2)(
            [action, log_std_out])

        inputs = [obs_ph]
        outputs = [action, values]

        self.rnn_model = tf.keras.Model(
            inputs=inputs,
            outputs=outputs)

        self.register_variables(self.rnn_model.variables)

    def forward(self, input_dict, state, seq_lens):

        model_out, self._value_out = self.base_model(input_dict["obs_flat"])

        return model_out, state

    def value_function(self):
        return tf.reshape(self._value_out, [-1])
