import pickle
import numpy as np
import tensorflow as tf

from utils import setFromFlat, MlpPolicyValue, LSTMPolicy
from zoo_utils import remove_prefix


def load_victim_agent(env_name, ob_space, action_space, model_path, init):
    # load victim agent
    sess=tf.get_default_session()
    if sess is None:
        tf_config = tf.ConfigProto(inter_op_parallelism_threads=1, intra_op_parallelism_threads=1)
        sess = tf.Session(config=tf_config)
        sess.__enter__()

    victim_agent = None
    use_mlp = False
    if env_name == 'multicomp/YouShallNotPassHumans-v0':
        victim_agent = MlpPolicyValue(scope="mlp_policy", reuse=tf.AUTO_REUSE,
                            ob_space=ob_space,
                            ac_space=action_space,
                            hiddens=[64, 64],  normalize=False)
        use_mlp = True
    else:
        victim_agent = LSTMPolicy(scope="lstm_policy", reuse=tf.AUTO_REUSE,
                            ob_space=ob_space,
                            ac_space=action_space,
                            hiddens=[128, 128], normalize=False)

    if init:
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
    setFromFlat(victim_agent.get_variables(), flat_params)
    return victim_agent