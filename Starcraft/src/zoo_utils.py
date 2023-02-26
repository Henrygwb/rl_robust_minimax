import os
import datetime
import pickle
import joblib
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

def load_adv_model(file_name):
    pretrain_model = pickle.load(open(file_name, 'rb'))
    model = {}
    for k, v in pretrain_model.items():
        model['default_policy'+k] = v
    return model


def format_change_model(file_name):
    # specific keys
    keys = ['polfc1/w:0', 'polfc1/b:0', 'polfc2/w:0', 'polfc2/b:0', 'polfc3/w:0', 'polfc3/b:0', \
            'vffc1/w:0', 'vffc1/b:0', 'vffc2/w:0', 'vffc2/b:0', 'vffc3/w:0', 'vffc3/b:0', \
            'vffinal/w:0', 'vffinal/b:0', 'polfinal/w:0', 'polfinal/b:0']
    # key-value
    save_model = {}
    loaded_params = joblib.load(file_name)
    assert len(keys) == len(loaded_params)
    for i in range(len(keys)):
        save_model[keys[i]] = loaded_params[i]

    # pickle save model
    out_name = 'starcraft'
    pickle.dump(save_model, open(out_name, 'wb'))


def pkl_to_joblib(file_name, out_name=None):
    model = pickle.load(open(file_name, 'rb'))
    keys = ['/polfc1/kernel', '/polfc1/bias', '/polfc2/kernel', '/polfc2/bias', '/polfc3/kernel', '/polfc3/bias', \
            '/vffc1/kernel', '/vffc1/bias', '/vffc2/kernel', '/vffc2/bias', '/vffc3/kernel', '/vffc3/bias', \
            '/vffinal/kernel', '/vffinal/bias', '/polfinal/kernel', '/polfinal/bias']
    save, flatten = [], []
    for k in keys:
        save.append(model[k])
        flatten.append(model[k].flatten())
    # save to joblib
    if out_name == None:
        return flatten
    else:
        joblib.dump(save, out_name)

def load_pretrain_model(file_name_0, file_name_1):
    pretrain_model_0 = pickle.load(open(file_name_0, 'rb'))
    pretrain_model_1 = pickle.load(open(file_name_1, 'rb'))
    model = {}
    opp_model = {}

    # Change the keys to the model parameter names in the LSTM/MLP class.
    dic_new_name = {}
    for key in list(pretrain_model_0.keys()):
        key_1 = key.split(':')[0]
        if 'weights' in key_1:
            key_1 = key_1.replace('/weights', '/kernel')
        elif '/w' in key_1:
            key_1 = key_1.replace('/w', '/kernel')
        elif 'biases' in key_1:
            key_1 = key_1.replace('/biases', '/bias')
        elif '/b' in key_1 and '/basic' not in key_1:
            key_1 = key_1.replace('/b', '/bias')
        dic_new_name[key] = key_1
    pretrain_model_newname_0 = dict((dic_new_name[key], value) for (key, value) in pretrain_model_0.items())

    for k, v in pretrain_model_newname_0.items():
        model['model/'+k] = v

    # Change the keys to the model parameter names in the LSTM/MLP class.
    dic_new_name = {}
    for key in list(pretrain_model_1.keys()):
        key_1 = key.split(':')[0]
        if '/weights' in key_1:
            key_1 = key_1.replace('/weights', '/kernel')
        elif '/w' in key_1:
            key_1 = key_1.replace('/w', '/kernel')
        elif '/biases' in key_1:
            key_1 = key_1.replace('/biases', '/bias')
        elif '/b' in key_1 and '/basic' not in key_1:
            key_1 = key_1.replace('/b', '/bias')
        dic_new_name[key] = key_1
    pretrain_model_newname_1 = dict((dic_new_name[key], value) for (key, value) in pretrain_model_1.items())

    for k, v in pretrain_model_newname_1.items():
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
class MLP(FullyConnectedNetwork):

    def __init__(self, obs_space, action_space, num_outputs, model_config, name):
        super(FullyConnectedNetwork, self).__init__(obs_space, action_space, num_outputs, 
                                                    model_config, name)

        # note that in the original father class "RecurrentNetwork",
        # log_std is dependent on observation, not like the fashion in our rnn_newloss implementation in which log_std is a independent variable.
        # In such case, num_outputs = 2 * action_size.
        # To implement the free_log_std (free means independent), we need firstly reduce the num_outputs in half.

        # free_log_std
        # independent of state
        # num_outputs = num_outputs // 2
        hiddens = model_config.get('fcnet_hiddens')

        obs_sz = int(np.product(obs_space.shape)) - action_space.n
        obs_ph = tf.keras.layers.Input(
            shape=(obs_sz,), name="observations")
        
        mask_ph = tf.keras.layers.Input(
            shape=(action_space.n,), name='mask')

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

        action -= (1 - mask_ph) * 1e30


        inputs = [obs_ph, mask_ph]
        outputs = [action, values]

        self.base_model = tf.keras.Model(
            inputs=inputs,
            outputs=outputs)

        self.register_variables(self.base_model.variables)

    def forward(self, input_dict, state, seq_lens):

        model_out, self._value_out = self.base_model([input_dict["obs"]["obs"], input_dict["obs"]["action_mask"]])

        return model_out, state

    def value_function(self):
        return tf.reshape(self._value_out, [-1])