import os
import logger
import functools
import numpy as np
import tensorflow as tf
from abc import ABC, abstractmethod
from baselines.common.tf_util import get_session
from zoo_utils import save_trainable_variables, load_trainable_variables, load_variables, parse_args
from zoo_utils import SetFromFlat, setFromFlat, load_from_file
from policies import build_policy
from stable_baselines.common.vec_env import SubprocVecEnv, DummyVecEnv
import gym
from os import path as osp
import roboschool
from adv_utils import RL_func, MimicModel, GradientExp

class Model(object):
    """
    Model class for the policy (player) that is trained.
    Create train and act models, PPO objective function.
    """

    # change_mse_coef: -0.1
    # state_modeling_coef: 1.0
    def __init__(self, *, policy, nbatch_act, nbatch_train, ent_coef, vf_coef, max_grad_norm, sess, 
                 change_mse_coef=-0.1, state_modeling_coef=1.0, microbatch_size=None, model_index='0'):

        self.sess = sess
        self.model_index = model_index

        with tf.variable_scope('ppo2_model%s'%model_index, reuse=tf.AUTO_REUSE):
            # CREATE OUR TWO MODELS
            # act_model that is used for sampling
            act_model = policy(nbatch_act, sess)

            # Train model for training
            if microbatch_size is None:
                train_model = policy(nbatch_train, sess)
            else:
                train_model = policy(microbatch_size, sess)

        self.scope = 'ppo2_model%s'%model_index
        self.A = A = train_model.pdtype.sample_placeholder([None])
        self.ADV = ADV = tf.placeholder(tf.float32, [None])
        self.R = R = tf.placeholder(tf.float32, [None])


        self.action_opp_next_ph = tf.placeholder(dtype=tf.float32, shape=self.A.shape,
                                                             name="action_opp_next_ph")
        self.obs_opp_next_ph = tf.placeholder(dtype=tf.float32, shape=train_model.X.shape,
                                                          name="obs_opp_next_ph")
        self.attention = tf.placeholder(dtype=tf.float32, shape=[None], name="attention_ph")
        action_ph_noise = train_model.action

        with tf.variable_scope("statem", reuse=True):
             obs_oppo_predict, obs_oppo_noise_predict = modeling_state(self.A, action_ph_noise,
                                                                       train_model.X)

        with tf.variable_scope("victim_param", reuse=tf.AUTO_REUSE):
             victim_model = RL_func(13, 2)
             action_opp_mal_noise = victim_model(obs_oppo_noise_predict)

        self.change_opp_action_mse = tf.reduce_mean(
            tf.abs(tf.multiply(action_opp_mal_noise - self.action_opp_next_ph,
                               tf.expand_dims(self.attention, axis=-1))
                   )
        )

        self.change_state_mse = tf.reduce_mean(tf.abs(obs_oppo_noise_predict - self.obs_opp_next_ph))
        self.change_mse = self.change_opp_action_mse - self.change_state_mse
        # Prediction error on oppo's next observation
        # change into the L infinity norm
        # L(infinity) = max(0, ||l1 -l2|| - c)^2
        self.state_modeling_mse = tf.reduce_mean(
             tf.square(tf.math.maximum(tf.abs(obs_oppo_predict - self.obs_opp_next_ph) - 1, 0)))


        # Keep track of old actor
        self.OLDNEGLOGPAC = OLDNEGLOGPAC = tf.placeholder(tf.float32, [None])
        # Keep track of old critic
        self.OLDVPRED = OLDVPRED = tf.placeholder(tf.float32, [None])

        self.LR = LR = tf.placeholder(tf.float32, [])
        # Clip range
        self.CLIPRANGE = CLIPRANGE = tf.placeholder(tf.float32, [])

        neglogpac = train_model.pd.neglogp(A)

        # Calculate the entropy
        entropy = tf.reduce_mean(train_model.pd.entropy())

        # Calculate the loss: Policy gradient loss - ent_coef * entropy loss +  vf_coef * value function loss

        # value function loss
        vpred = train_model.vf
        # clip value
        vpredclipped = OLDVPRED + tf.clip_by_value(train_model.vf - OLDVPRED, - CLIPRANGE, CLIPRANGE)
        vf_losses1 = tf.square(vpred - R)
        vf_losses2 = tf.square(vpredclipped - R)

        vf_loss = .5 * tf.reduce_mean(tf.maximum(vf_losses1, vf_losses2))

        # Calculate ratio: current policy / old policy.
        ratio = tf.exp(OLDNEGLOGPAC - neglogpac)

        # minimize -loss -> maximize loss
        pg_losses = -ADV * ratio

        pg_losses2 = -ADV * tf.clip_by_value(ratio, 1.0 - CLIPRANGE, 1.0 + CLIPRANGE)

        # policy gradient loss
        pg_loss = tf.reduce_mean(tf.maximum(pg_losses, pg_losses2))
        approxkl = .5 * tf.reduce_mean(tf.square(neglogpac - OLDNEGLOGPAC))
        clipfrac = tf.reduce_mean(tf.to_float(tf.greater(tf.abs(ratio - 1.0), CLIPRANGE)))

        # total loss
        loss = pg_loss - entropy * ent_coef + vf_loss * vf_coef + change_mse_coef * self.change_mse
        loss_mimic = state_modeling_coef * self.state_modeling_mse

        # Update the parameters.

        # 1. Get the model parameters
        params = tf.trainable_variables('ppo2_model%s'%model_index)
        self.params = [params, tf.trainable_variables("statem")]

        # 2. Build our trainer
        self.trainer = tf.train.AdamOptimizer(learning_rate=LR, epsilon=1e-5)
        self.trainer_mimic = tf.train.AdamOptimizer(learning_rate=LR, epsilon=1e-5)

        # 3.1 Calculate the gradients --loss
        grads_and_var = self.trainer.compute_gradients(loss, self.params[0])
        grads, var = zip(*grads_and_var)
        if max_grad_norm is not None:
            # Clip the gradients.
            grads, _grad_norm = tf.clip_by_global_norm(grads, max_grad_norm)
        grads_and_var = list(zip(grads, var))

        # 3.2 Calculate the gradients --loss_mimic
        mimic_grads_and_var = self.trainer_mimic.compute_gradients(loss_mimic, self.params[1])
        mimic_grads, mimic_var = zip(*mimic_grads_and_var)
        if max_grad_norm is not None:
            # Clip the gradients.
            mimic_grads, _mimic_grad_norm = tf.clip_by_global_norm(mimic_grads, max_grad_norm)
        mimic_grads_and_var = list(zip(mimic_grads, mimic_var))

        self.grads = grads
        self.var = var
        self._train_op = self.trainer.apply_gradients(grads_and_var)
        self._train_mimic_op = self.trainer_mimic.apply_gradients(mimic_grads_and_var)

        self.loss_names = ['policy_loss', 'value_loss', 'policy_entropy', 'approxkl', 'clipfrac']
        self.loss_list = [pg_loss, vf_loss, entropy, approxkl, clipfrac]

        self.train_model = train_model
        self.act_model = act_model
        self.step = act_model.step
        self.value = act_model.value
        self.initial_state = act_model.initial_state

        # save and load
        self.save = functools.partial(save_trainable_variables, scope="ppo2_model%s"%model_index,sess=sess)
        self.load_pretrain = functools.partial(load_trainable_variables, scope="ppo2_model%s"%model_index, sess=sess)

        self.assign_op = SetFromFlat(params, sess=self.sess)
        self.load = functools.partial(load_variables, op=self.assign_op, variables=params)

    def train_step(self, lr, cliprange, obs, returns, actions, values, neglogpacs, obs_opp_next,
                   act_opp_next, attention):
        """
        One training step.
        """
        # Here we calculate advantage A(s,a) = R + yV(s') - V(s)
        # Returns = R + yV(s')
        advs = returns - values

        # Normalize the advantages
        advs = (advs - advs.mean()) / (advs.std() + 1e-8)

        td_map = {
            self.train_model.X: obs,
            self.action_opp_next_ph: act_opp_next,
            self.obs_opp_next_ph: obs_opp_next,
            self.attention: attention,
            self.A: actions,
            self.ADV: advs,
            self.R: returns,
            self.LR: lr,
            self.CLIPRANGE: cliprange,
            self.OLDNEGLOGPAC: neglogpacs,
            self.OLDVPRED: values
        }

        return self.sess.run(self.loss_list + [self._train_op, self._train_mimic_op],td_map)[:-2]

    def load_mimic_model(self, mimic_model_path):
        victim_variable = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, "victim_param")
        param = load_from_file(param_pkl_path=mimic_model_path)
        setFromFlat(victim_variable, param, sess=self.sess)


class Act_Model(object):
    """
    Model class for the policy (player) that is only used for acting.
    """
    def __init__(self, *, policy, nbatch_act, sess, model_index='0'):
        self.model_index = model_index
        self.sess = sess
        with tf.variable_scope('ppo2_act_model%s'%model_index):
            # act_model that is used for sampling
            act_model = policy(nbatch_act, sess)

        self.act_model = act_model
        self.step = act_model.step
        self.initial_state = act_model.initial_state
        scope = "ppo2_act_model%s"%model_index

        self.save = functools.partial(save_trainable_variables,variables=tf.trainable_variables(scope=scope), sess=sess)

        params = tf.trainable_variables(scope=scope)
        self.assign_op = SetFromFlat(params, sess=self.sess)
        self.load = functools.partial(load_variables, op=self.assign_op, variables=params)

class AbstractEnvRunner(ABC):
    def __init__(self, *, env, model, n_steps, nplayer=2):
        self.env = env
        self.model = model
        # nenv num
        self.nenv = nenv = self.env.num_envs
        
        self.nplayer = nplayer

        # These games do not take observation, make all the observations as zero.
        self.obs = np.zeros((nenv,) + env.observation_space.shape,
                            dtype=model.train_model.X.dtype.name)
        self.obs[:] = env.reset()
        self.n_steps = n_steps

        # These games do not have states, mark all the states as None.
        self.states = model.initial_state

        # These games are one step games, mark all the done as False here.
        self.dones = np.array([False for _ in range(self.nenv)])

    @abstractmethod
    def run(self):
        raise NotImplementedError

class Runner(AbstractEnvRunner):
    """
    Conduct minimax play using the current agents in the environment and collect trajectories.
    """
    def __init__(self, *, env, model, n_steps, gamma, lam, nplayer=0, stochastic=False):
        super().__init__(env=env, model=model, n_steps=n_steps, nplayer=nplayer)
        # Lambda used in GAE (General Advantage Estimation)
        self.lam = lam
        # Discount rate
        self.gamma = gamma
        self.n_steps = n_steps

        self.stochastic = stochastic

    def run(self):

        def sf01(arr):
            """
            swap and then flatten axes 0 and 1
            """
            s = arr.shape
            return arr.swapaxes(0, 1).reshape(s[0] * s[1], *s[2:])

        # print('player %d model %d' %(self.nplayer, self.idx))

        mb_obs, mb_rewards, mb_actions, mb_values, mb_dones, mb_neglogpacs = [], [], [], [], [], []
        mb_states = self.states

        for _ in range(self.n_steps):

            # give zero observations, none states, false done -> action [nenv, ], value [nenv, ], self.states None, neglogpacs [nenv. ]
            actions, values, self.states, neglogpacs = self.model.step(self.obs, self.stochastic,
                                                                      S=self.states,
                                                                      M=self.dones)
            mb_obs.append(self.obs.copy()) # self.observation [nenv, nagent], self.observation[:, id] [nenv, ]
            mb_actions.append(actions)
            mb_values.append(values)
            mb_neglogpacs.append(neglogpacs)
            mb_dones.append(self.dones) # self.dones [nenv, nagent], self.dones[:, id] [nenv, ]

            self.obs, rewards, self.dones, infos = self.env.step(actions) # self.obs [nenv, nagent] (all zeros), rewards [nenv, nagent], done [nenv, nagent] (all true).
            mb_rewards.append(rewards)
        # batch of steps to batch of rollouts
        mb_obs = np.asarray(mb_obs, dtype=self.obs.dtype)  # [n_steps, nenv]

        mb_rewards = np.asarray(mb_rewards, dtype=np.float32)  # [n_steps, nenv]

        mb_actions = np.asarray(mb_actions)  # [n_steps, nenv]
        mb_values = np.asarray(mb_values, dtype=np.float32)  # [n_steps, nenv]
        mb_neglogpacs = np.asarray(mb_neglogpacs, dtype=np.float32)  # [n_steps, nenv]
        mb_dones = np.asarray(mb_dones, dtype=np.bool)  # [n_steps, nenv]
        last_values = self.model.value(self.obs, S=self.states, M=self.dones)  # [nenv, ]

        # Discount/bootstrap off value fn
        mb_advs = np.zeros_like(mb_rewards)
        lastgaelam = 0
        for t in reversed(range(self.n_steps)):
            if t == self.n_steps - 1:
                nextnonterminal = 1.0 - self.dones
                nextvalues = last_values
            else:
                nextnonterminal = 1.0 - mb_dones[t + 1]
                nextvalues = mb_values[t + 1]
            delta = mb_rewards[t] + self.gamma * nextvalues * nextnonterminal - mb_values[t]
            mb_advs[t] = lastgaelam = delta + self.gamma * self.lam * nextnonterminal * lastgaelam
        mb_returns = mb_advs + mb_values

        return (*map(sf01, (mb_obs, mb_returns, mb_rewards, mb_dones, mb_actions, mb_values, mb_neglogpacs)),
                mb_states)        

# minimax learn
def learn(*, total_timesteps, out_dir, n_steps, ent_coef=0.0, lr=1e-3, vf_coef=0.5, max_grad_norm=0.5, gamma=0.995, 
          lam=0.95, log_interval=8, nminibatches=64, noptepochs=6, cliprange=0.2, save_interval=1,  player_n=0,
          win_info_file=None, load_pretrain=0, pretrain_model_path=None, opp_model_path=None, mimic_opp_model_path=None, 
          exp_method='grad', server_id=0, **network_kwargs):
    # set logger here
    logger.configure(folder=osp.join(out_dir, str(player_n), 'rl'),
                     format_strs=['tensorboard', 'stdout'])
    # create environments
    env = gym.make('RoboschoolPong-v1')
    env.unwrapped.multiplayer(env, game_server_guid="pong_adv_"+str(server_id), player_n=player_n)
    env = DummyVecEnv([lambda: env])

    sess = get_session()
    nenvs = env.num_envs
    nbatch = nenvs*n_steps
    nbatch_train = nbatch // nminibatches

    policy = build_policy(env, 'custom_mlp', value_network='copy', **network_kwargs)

    model = Model(policy=policy, nbatch_act=nenvs, nbatch_train=nbatch_train, ent_coef=ent_coef, vf_coef=vf_coef,
                  max_grad_norm=max_grad_norm, model_index=str(player_n), sess=sess)
    # used to get the actions of the opponent
    opp_model = Act_Model(policy=policy, nbatch_act=n_steps, sess=sess, model_index=str(1-player_n))
    mimic_opp_model = MimicModel(sess)
    sess.run(tf.global_variables_initializer())

    uninitialized = sess.run(tf.report_uninitialized_variables())
    assert len(uninitialized) == 0, 'There are uninitialized variables.'

    if load_pretrain != 0:
        model.load(pretrain_model_path)

    if mimic_opp_model_path != 'None':
        model.load_mimic_model(mimic_opp_model_path + '/model.pkl')

    opp_model.load(opp_model_path)

    # training process
    # number of iterations
    nupdates = total_timesteps//nbatch

    # Define the runner for the agent under training.
    runner = Runner(env=env, model=model, n_steps=n_steps, gamma=gamma, lam=lam, nplayer=player_n)
    idx_lines = 1

    # mimic opponent model
    # used for gradient explanation
    if mimic_opp_model_path != 'None':
       mimic_opp_model.load(mimic_opp_model_path + '/model.h5')
    exp_test = GradientExp(mimic_opp_model)
   
    for update in range(0, nupdates):
        # if update > 1:
        #     print('update %d player %d ' %(update, player_n), model.log_p())

        runner.stochastic = True
        obs, returns, rewards, masks, actions, values, neglogpacs, states = runner.run() # shape [nstep*nenv, ]
        idx_lines += 1

        ob_next_ph = infer_obs_next_ph(obs)
        obs_opp_next_ph = infer_obs_opp_ph(ob_next_ph)

        obs_opp_ph = infer_obs_opp_ph(obs)
        action_opp_next_ph = infer_opp_action(opp_model, obs_opp_next_ph)

         # calculate the attention paid on opponent
        attention_ph = calculate_attention(obs_oppo=obs_opp_ph, exp_test=exp_test, exp_method=exp_method)
        # game_info = np.loadtxt(win_info_file)
        # print('player_%d, score is %f' %(player_n, np.sum(rewards)))
        # print('left is %f: right is %f' %(game_info[-1, 4], game_info[-1, 5]))
        if update % log_interval == 0:
            logger.info('Done.')

        mblossvals = []
        # Train the policy using the corrected trajectories with noptepochs epoches.
        inds = np.arange(nbatch)
        for epoch in range(noptepochs):
            np.random.shuffle(inds) # Randomize the indexes
            # train the policy with the trajectories from each batch.
            for ii, start in enumerate(range(0, nbatch, nbatch_train)):
                end = start + nbatch_train
                mbinds = inds[start:end]
                slices = (arr[mbinds] for arr in (obs, returns, actions, values, neglogpacs, 
                          obs_opp_next_ph, action_opp_next_ph, attention_ph))

                mblossvals.append(model.train_step(lr, cliprange, *slices))

        lossvals = np.mean(mblossvals, axis=0)

        if save_interval and (update % save_interval == 0 or update == 1):
            checkdir = os.path.join(out_dir, 'checkpoints', 'model', '%.5i'%update)
            os.makedirs(checkdir, exist_ok=True)
            savepath = os.path.join(checkdir, 'model')
            model.save(savepath)

        if update % log_interval == 0:
            game_total = 0
            iter = -1
            while game_total <= 40:
                runner.stochastic = False
                runner.run()
                idx_lines += 1
                iter -= 1
                game_info = np.loadtxt(win_info_file)
                assert idx_lines == game_info.shape[0]
                info = game_info[-1, :3] - game_info[iter, :3]
                game_total = np.sum(info)


        if update % log_interval == 0:
            game_info = np.loadtxt(win_info_file)
            assert idx_lines == game_info.shape[0]
            info = game_info[-1, :3] - game_info[iter, :3]
            game_total = np.sum(info)

            logger.logkv("nupdates", update)
            logger.logkv("total_timesteps", update * nbatch)
            logger.logkv('learning_rate', lr)
            logger.logkv('returns', np.mean(returns))
            logger.logkv('rewards', np.mean(rewards))

            for (lossval, lossname) in zip(lossvals, model.loss_names):
                logger.logkv('loss' + '/' + lossname, lossval)
            logger.logkv('game_total', game_total)
            win_0 = info[0] * 1.0 / game_total
            win_1 = info[1] * 1.0 / game_total
            tie = info[2] * 1.0 / game_total
            logger.logkv('win_0', info[0] * 1.0 / game_total)
            logger.logkv('win_1', info[1] * 1.0 / game_total)
            logger.logkv('tie', info[2] * 1.0 / game_total)
            logger.dumpkvs()
            # write into txt
            fid = open(out_dir + '/Log_%d.txt ' % player_n, 'a+')
            fid.write("%d %f %f %f\n" % (update, win_0, win_1, tie))

def infer_obs_next_ph(obs_ph):
    obs_next_ph = np.zeros_like(obs_ph)
    obs_next_ph[:-1, :] = obs_ph[1:, :]
    return obs_next_ph

def infer_obs_opp_ph(obs_ph):
    abs_opp_ph = np.zeros_like(obs_ph)
    neg_sign = [-1, -1, 1, 1, -1, -1, 1, 1, -1, -1, 1, 1, 1]
    abs_opp_ph[:, :4] = obs_ph[:, 4:8]
    abs_opp_ph[:, 4:8] = obs_ph[:, 0:4]
    abs_opp_ph[:, 8:] = obs_ph[:, 8:]
    return abs_opp_ph*neg_sign

def infer_opp_action(model, obs_ph):
    a, _, _, _ = model.step(obs_ph)
    return a

def calculate_attention(obs_oppo, exp_test=None, exp_method='integratedgrad'):
    assert exp_test != None
    if exp_method == 'grad':
       grads = exp_test.grad(obs_oppo)
    elif exp_method == 'integratedgrad':
       grads = exp_test.integratedgrad(obs_oppo)
    else:
       grads = exp_test.smoothgrad(obs_oppo)

    oppo_action = exp_test.output(obs_oppo)
    grads[:, 0:4] = 0
    grads[:,8:] = 0
    new_obs_oppo = grads * obs_oppo
    new_oppo_action = exp_test.output(new_obs_oppo)
    relative_norm = np.max(abs(new_oppo_action - oppo_action), axis=1)
    # relative_action_norm = np.linalg.norm(new_oppo_action - oppo_action, axis=1)
    new_scalar = 1.0 / (1.0 + relative_norm) * 10.0
    return  new_scalar


def modeling_state(action_ph, action_noise, obs_self):
    w1 = tf.Variable(tf.truncated_normal([13, 32]), name="cur_obs_embed_w1")
    b1 = tf.Variable(tf.truncated_normal([32]), name="cur_obs_embed_b1")
    obs_embed = tf.nn.relu(tf.add(tf.matmul(obs_self, w1), b1), name="cur_obs_embed")

    w2 = tf.Variable(tf.truncated_normal([2, 4]), name="act_embed_w1")
    b2 = tf.Variable(tf.truncated_normal([4]), name="act_embed_b1")

    act_embed = tf.nn.relu(tf.add(tf.matmul(action_ph, w2), b2), name="act_embed")
    act_embed_noise = tf.nn.relu(tf.add(tf.matmul(action_noise, w2), b2), name="act_noise_embed")

    obs_act_concat = tf.concat([obs_embed, act_embed], -1, name="obs_act_concat")
    obs_act_noise_concat = tf.concat([obs_embed, act_embed_noise], -1, name="obs_act_noise_concat")


    w3 = tf.Variable(tf.truncated_normal([36, 16]), name="obs_act_embed_w1")
    b3 = tf.Variable(tf.truncated_normal([16]), name="obs_act_embed_b1")

    obs_act_embed = tf.nn.relu(tf.add(tf.matmul(obs_act_concat, w3), b3), name="obs_act_embed")
    obs_act_noise_embed = tf.nn.relu(tf.add(tf.matmul(obs_act_noise_concat, w3), b3), name="obs_act_noise_embed")


    w4 = tf.Variable(tf.truncated_normal([16, 13]), name="obs_oppo_predict_w1")
    b4 = tf.Variable(tf.truncated_normal([13]), name="obs_oppo_predict_b1")

    obs_oppo_predict = tf.nn.tanh(tf.add(tf.matmul(obs_act_embed, w4), b4), name="obs_oppo_predict_part")
    obs_oppo_predict_noise = tf.nn.tanh(tf.add(tf.matmul(obs_act_noise_embed, w4), b4), name="obs_oppo_predict_noise_part")

    return obs_oppo_predict, obs_oppo_predict_noise

# opponent be attacked
def play(env, model):
    obs = env.reset()
    done = False
    while 1:
        a, _, _, _ = model.step(obs, False, S=None, M=done)
        obs, _, done,_ = env.step(a)

def test(player_n=0, opp_model_path=None, server_id=0):
    env = gym.make("RoboschoolPong-v1")
    env.unwrapped.multiplayer(env, game_server_guid="pong_adv_"+str(server_id), player_n=player_n)
    env = DummyVecEnv([lambda: env])

    # set up the model
    sess = get_session()
    policy = build_policy(env, 'custom_mlp', value_network='copy')
    model = Act_Model(policy=policy, nbatch_act=1, sess=sess, model_index=str(player_n))
    sess.run(tf.global_variables_initializer())
    # load the model
    model.load(opp_model_path)
    # run the model
    play(env, model)

# # call learn function
parse = parse_args()
if parse.adv_party:
    learn(total_timesteps=parse.total_timesteps, out_dir=parse.out_dir, n_steps=parse.n_steps,
          ent_coef=parse.ent_coef, lr=parse.lr, gamma=parse.gamma, nminibatches=parse.nminibatches,
          noptepochs=parse.noptepochs, player_n=parse.player_n, win_info_file=parse.win_info_file,
          load_pretrain=parse.load_pretrain, pretrain_model_path=parse.pretrain_model_path, 
          server_id=parse.server_id, opp_model_path=parse.opp_model_path, mimic_opp_model_path=parse.mimic_model_path)
else:
    test(player_n=parse.player_n, opp_model_path=parse.opp_model_path, server_id=parse.server_id)
