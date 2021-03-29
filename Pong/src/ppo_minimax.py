import os
import logger
import functools
import numpy as np
import tensorflow as tf
from abc import ABC, abstractmethod
from baselines.common.tf_util import get_session
from zoo_utils import save_trainable_variables, load_trainable_variables, load_variables, parse_args, custom_mlp, SetFromFlat
from policies import build_policy
from stable_baselines.common.vec_env import SubprocVecEnv, DummyVecEnv
import gym
from os import path as osp
import roboschool


class Model(object):
    """
    Model class for the policy (player) that is trained.
    Create train and act models, PPO objective function.
    """

    def __init__(self, *, policy, nbatch_act, nbatch_train, ent_coef, vf_coef, max_grad_norm, sess,
                 microbatch_size=None, model_index='0'):

        self.sess = sess
        self.model_index = model_index

        with tf.variable_scope('ppo2_model%s' % model_index, reuse=tf.AUTO_REUSE):
            # CREATE OUR TWO MODELS
            # act_model that is used for sampling
            act_model = policy(nbatch_act, sess)

            # Train model for training
            if microbatch_size is None:
                train_model = policy(nbatch_train, sess)
            else:
                train_model = policy(microbatch_size, sess)

        self.scope = 'ppo2_model%s' % model_index
        self.A = A = train_model.pdtype.sample_placeholder([None])
        self.ADV = ADV = tf.placeholder(tf.float32, [None])
        self.R = R = tf.placeholder(tf.float32, [None])

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
        loss = pg_loss - entropy * ent_coef + vf_loss * vf_coef

        # Update the parameters.

        # 1. Get the model parameters
        params = tf.trainable_variables('ppo2_model%s' % model_index)
        self.params = params
        # print("para",model_index,params)

        # 2. Build our trainer
        self.trainer = tf.train.AdamOptimizer(learning_rate=LR, epsilon=1e-5)

        # 3. Calculate the gradients
        grads_and_var = self.trainer.compute_gradients(loss, params)
        grads, var = zip(*grads_and_var)

        if max_grad_norm is not None:
            # Clip the gradients.
            grads, _grad_norm = tf.clip_by_global_norm(grads, max_grad_norm)
        grads_and_var = list(zip(grads, var))

        self.grads = grads
        self.var = var
        self._train_op = self.trainer.apply_gradients(grads_and_var)
        self.loss_names = ['policy_loss', 'value_loss', 'policy_entropy', 'approxkl', 'clipfrac']
        self.loss_list = [pg_loss, vf_loss, entropy, approxkl, clipfrac]

        self.train_model = train_model
        self.act_model = act_model
        self.step = act_model.step
        self.value = act_model.value
        self.initial_state = act_model.initial_state

        # save and load
        self.save = functools.partial(save_trainable_variables, scope="ppo2_model%s" % model_index, sess=sess)
        self.load_pretrain = functools.partial(load_trainable_variables, scope="ppo2_model%s" % model_index, sess=sess)

        self.assign_op = SetFromFlat(params, sess=self.sess)
        self.load = functools.partial(load_variables, op=self.assign_op, variables=params)

    def train_step(self, lr, cliprange, obs, returns, actions, values, neglogpacs):
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
            self.A: actions,
            self.ADV: advs,
            self.R: returns,
            self.LR: lr,
            self.CLIPRANGE: cliprange,
            self.OLDNEGLOGPAC: neglogpacs,
            self.OLDVPRED: values
        }

        return self.sess.run(self.loss_list + [self._train_op], td_map)[:-1]


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

    def __init__(self, *, env, model, n_steps, gamma, lam, idx=0, nplayer=0, stochastic=False):
        super().__init__(env=env, model=model, n_steps=n_steps, nplayer=nplayer)
        # Lambda used in GAE (General Advantage Estimation)
        self.lam = lam
        # Discount rate
        self.gamma = gamma
        self.n_steps = n_steps
        self.idx = idx

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
            mb_obs.append(self.obs.copy())  # self.observation [nenv, nagent], self.observation[:, id] [nenv, ]
            mb_actions.append(actions)
            mb_values.append(values)
            mb_neglogpacs.append(neglogpacs)
            mb_dones.append(self.dones)  # self.dones [nenv, nagent], self.dones[:, id] [nenv, ]

            self.obs, rewards, self.dones, infos = self.env.step(
                actions)  # self.obs [nenv, nagent] (all zeros), rewards [nenv, nagent], done [nenv, nagent] (all true).
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
          lam=0.95, log_interval=2, nminibatches=64, noptepochs=6, cliprange=0.2, save_interval=1, nagents=5,
          inner_loop_party_0=1, inner_loop_party_1=1, player_n=0, win_info_file=None, load_pretrain=False, server_id=0,
          pretrain_model_path=None, **network_kwargs):
    # set logger here
    logger.configure(folder=osp.join(out_dir, 'rl'),
                     format_strs=['tensorboard', 'stdout'])
    # create environments
    env = gym.make('RoboschoolPong-v1')
    env.unwrapped.multiplayer(env, game_server_guid="pong_"+str(server_id), player_n=player_n)
    # env = SubprocVecEnv([lambda: env for _ in range(1)])
    env = DummyVecEnv([lambda: env])

    sess = get_session()
    nenvs = env.num_envs
    nbatch = nenvs * n_steps
    nbatch_train = nbatch // nminibatches

    models = []
    pre_runner_id = -1

    # initialize n number of policies for both party.
    policy = build_policy(env, 'custom_mlp', value_network='copy', **network_kwargs)

    for i in range(nagents):
        model = Model(policy=policy, nbatch_act=nenvs, nbatch_train=nbatch_train, ent_coef=ent_coef, vf_coef=vf_coef,
                      max_grad_norm=max_grad_norm, model_index=str(player_n) + '_' + str(i), sess=sess)
        models.append(model)

    sess.run(tf.global_variables_initializer())
    uninitialized = sess.run(tf.report_uninitialized_variables())
    assert len(uninitialized) == 0, 'There are uninitialized variables.'

    if load_pretrain:
        for i in range(nagents):
             models[i].load_pretrain(pretrain_model_path)

    # Define n * n runner
    # 0 0 .. 0     1 1 .. 1
    # 0 1 .. n-1   0 1 .. n-1
    runners = []
    num_runners = nagents * nagents
    for i in range(num_runners):
        if player_n == 0:
            idx = int(i / nagents)
        else:
            idx = int(i % nagents)
        runner = Runner(env=env, model=models[idx], n_steps=n_steps, gamma=gamma,
                        lam=lam, idx=idx, nplayer=player_n)
        runners.append(runner)
    # training process
    nupdates = total_timesteps // nbatch  # number of iterations
    idx_lines = 1

    # outloop
    for update in range(0, nupdates):

        print('update is %d' % update)

        for training_party in [0, 1]:
            if training_party == 0:
                inneriter = inner_loop_party_0
            else:
                inneriter = inner_loop_party_1

            assert nbatch % nminibatches == 0
            mblossvals = []
            rewards_all = []
            returns_all = []

            win_all = []
            tie_all = []
            games_all = []

            tmp_rewards_0 = []
            tmp_rewards_1 = []

            tmp_rewards_0_ = []
            tmp_rewards_1_ = []

            for i in range(num_runners):
                # print('Runner %d' % i)
                if pre_runner_id != -1:
                    runners[i].obs = runners[pre_runner_id].obs
                    runners[i].dones = runners[pre_runner_id].dones
                    pre_runner_id = i
                runners[i].stochastic = False    
                _, _, rewards, _, _, _, _, _ = runners[i].run()
                idx_lines += 1
                game_info = np.loadtxt(win_info_file)

                # print('player_%d'% player_n)
                # print(rewards)
                assert idx_lines == game_info.shape[0]
                total_rewards = game_info[-1, 3]
                tmp_rewards_0.append(game_info[-1, 4])
                tmp_rewards_1.append(game_info[-1, 5])

                if player_n == 0:
                    tmp_rewards_0_.append(np.sum(rewards))
                else:
                    tmp_rewards_0_.append(total_rewards - np.sum(rewards))

                if player_n == 1:
                    tmp_rewards_1_.append(np.sum(rewards))
                else:
                    tmp_rewards_1_.append(total_rewards - np.sum(rewards))

                print('left: %f right: %f' % (game_info[-1, 4], game_info[-1, 5]))
                print('runner: %d player: %d reward_0: %f reward_1: %f' % (
                    i, player_n, tmp_rewards_0_[i], tmp_rewards_1_[i]))

            for n in range(nagents):
                rewards_all_0 = []
                idx_all_0 = []

                rewards_all_1 = []
                idx_all_1 = []

                # find the best opponents
                for n1 in range(nagents):
                    # idx
                    idx = n * nagents + n1
                    rewards_all_0.append(tmp_rewards_0[idx])
                    idx_all_0.append(idx)

                    # -idx
                    idx = n1 * nagents + n
                    rewards_all_1.append(tmp_rewards_1[idx])
                    idx_all_1.append(idx)
                rewards_all_0 = np.array(rewards_all_0)
                min_id_0 = np.argmin(rewards_all_0)
                id_0 = idx_all_0[min_id_0]

                rewards_all_1 = np.array(rewards_all_1)
                min_id_1 = np.argmin(rewards_all_1)
                id_1 = idx_all_1[min_id_1]

                print('Player %d id_0 %d id_1 %d' % (player_n, id_0, id_1))

                for iter in range(inneriter):
                    # runners 0 | 1
                    # print('agents %d' % n)
                    if training_party == 0:
                        if pre_runner_id != -1:
                            runners[id_0].obs = runners[pre_runner_id].obs
                            runners[id_0].dones = runners[pre_runner_id].dones
                            pre_runner_id = id_0

                        runners[id_0].stochastic = True
                        obs, returns, rewards, masks, actions, values, neglogpacs, states = runners[id_0].run()
                    else:
                        if pre_runner_id != -1:
                            runners[id_1].obs = runners[pre_runner_id].obs
                            runners[id_1].dones = runners[pre_runner_id].dones
                            pre_runner_id = id_1
                        runners[id_1].stochastic = True
                        obs, returns, rewards, masks, actions, values, neglogpacs, states = runners[id_1].run()
                    idx_lines += 1
                    # print('---------', player_n)
                    if training_party == player_n:
                        if iter == inneriter - 1:
                            mblossvals_tmp = []

                        # Index of each element of batch_size
                        # Create the indices array
                        inds = np.arange(nbatch)
                        for epoch in range(noptepochs):
                            # Randomize the indexes
                            np.random.shuffle(inds)
                            # 0 to batch_size with batch_train_size step
                            for ii, start in enumerate(range(0, nbatch, nbatch_train)):
                                end = start + nbatch_train
                                mbinds = inds[start:end]
                                slices = (arr[mbinds] for arr in (obs, returns, actions, values, neglogpacs))
                                if iter == inneriter - 1:
                                    mblossvals_tmp.append(models[n].train_step(lr, cliprange, *slices))
                                else:
                                    models[n].train_step(lr, cliprange, *slices)

                        if iter == inneriter - 1:
                            lossvals = np.mean(mblossvals_tmp, axis=0)
                            mblossvals.append(lossvals)
                            rewards_all.append(np.mean(rewards))
                            returns_all.append(np.mean(returns))

                            if save_interval and (update % save_interval == 0 or update == 1):
                                checkdir = os.path.join(out_dir, 'checkpoints', 'model_%d_%d' % (player_n, n),
                                                        '%.5i' % update)
                                os.makedirs(checkdir, exist_ok=True)
                                savepath = os.path.join(checkdir, 'model')
                                models[n].save(savepath)


                if update % log_interval == 0:
                    # run test, play with the best opponent model
                    game_total = 0
                    iter = -1
                    while game_total <= 30:
                        if training_party == 0:
                            runners[id_0].stochastic = False
                            runners[id_0].run()
                        else:
                            runners[id_1].stochastic = False
                            runners[id_1].run()
                        idx_lines += 1
                        iter -= 1
                        game_info = np.loadtxt(win_info_file)
                        assert idx_lines == game_info.shape[0]
                        info = game_info[-1, :3] - game_info[iter, :3]
                        game_total = np.sum(info)

                    if training_party == player_n:
                        game_info = np.loadtxt(win_info_file)
                        assert idx_lines == game_info.shape[0]
                        info = game_info[-1, :3] - game_info[iter, :3]
                        game_total = np.sum(info)
                        games_all.append(game_total)
                        if player_n == 0:
                            win_all.append(info[0] * 1.0 / game_total)
                        else:
                            win_all.append(info[1] * 1.0 / game_total)
                        tie_all.append(info[2] * 1.0 / game_total)

            if update % log_interval == 0:

                if training_party == player_n:

                    for n in range(nagents):

                        logger.logkv('Returns: %d th in %d' % (n, player_n), np.mean(returns_all[n]))
                        logger.logkv('Rewards: %d th in %d' % (n, player_n), np.mean(rewards_all[n]))
                        logger.logkv('Game_total: %d th in %d player' % (n, player_n), games_all[n])
                        logger.logkv('Win: %d th in %d player' % (n, player_n), win_all[n])
                        logger.logkv('Tie: %d th in %d player' % (n, player_n), tie_all[n])

                        for (lossval, lossname) in zip(mblossvals[n], models[n].loss_names):
                            logger.logkv('Loss: %d th in %d' % (n, player_n) + '/' + lossname, lossval)

                    logger.logkv("nupdates_%d" % player_n, update)
                    logger.logkv("total_timesteps_%d" % player_n, update * nbatch)
                    logger.logkv('learning_rate', lr)
                    logger.dumpkvs()

    # select the final best agent
    tmp_rewards = []
    rewards_id = np.zeros((nagents, nagents))
    for i in range(num_runners):
        # print('Runner %d' % i)
        if pre_runner_id != -1:
            runners[i].obs = runners[pre_runner_id].obs
            runners[i].dones = runners[pre_runner_id].dones
            pre_runner_id = i
        _, _, rewards, _, _, _, _, _ = runners[i].run()
        idx_lines += 1
        tmp_rewards.append(np.sum(rewards))

    for n0 in range(nagents):
        # find the best opponents
        for n1 in range(nagents):
            if player_n == 0:
                idx = n0 * nagents + n1
            else:
                idx = n1 * nagents + n0
            rewards_id[n0][n1] = tmp_rewards[idx]

    rewards_id = np.mean(rewards_id, axis=1)
    best_id = np.argmax(rewards_id)
    # write into txt
    f = open(out_dir + '/best-agents-%d.txt' % player_n, 'w')
    f.write('The best agent of party %d in the agent %d.\n' % (player_n, best_id))
    f.close()

# call learn function
parse = parse_args()
learn(total_timesteps=parse.total_timesteps, out_dir=parse.out_dir, n_steps=parse.n_steps,
      ent_coef=parse.ent_coef, lr=parse.lr, gamma=parse.gamma, nminibatches=parse.nminibatches,
      noptepochs=parse.noptepochs, nagents=parse.nagents, inner_loop_party_0=parse.inner_loop_party_0,
      inner_loop_party_1=parse.inner_loop_party_1, player_n=parse.player_n, pretrain_model_path=parse.pretrain_model_path,
      win_info_file=parse.win_info_file, load_pretrain=parse.load_pretrain, server_id=parse.server_id)
