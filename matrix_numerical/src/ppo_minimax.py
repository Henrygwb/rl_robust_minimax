import os
import logger
import functools
import numpy as np
import tensorflow as tf
from abc import ABC, abstractmethod
from env import FakeSingleSpacesVec
from baselines.common.tf_util import get_session
from zoo_utils import save_trainable_variables, load_trainable_variables, build_policy
from scipy.special import softmax


class Model(object):
    """
    Model class for the policy (player) that is trained.
    Create train and act models, PPO objective function.
    """
    def __init__(self, *, policy, nbatch_act, nbatch_train, ent_coef, vf_coef, max_grad_norm,
                 microbatch_size=None, model_index='0'):

        self.sess = sess = get_session()
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
        params = tf.trainable_variables('ppo2_model%s'%model_index)
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
        self.save = functools.partial(save_trainable_variables, scope="ppo2_model%s"%model_index,sess=sess)
        self.load = functools.partial(load_trainable_variables, scope="ppo2_model%s"%model_index, sess=sess)

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

        return self.sess.run(self.loss_list + [self._train_op],td_map)[:-1]

    def log_p(self):
        params_v = self.sess.run(self.params)
        return params_v[0][0]


class Act_Model(object):
    """
    Model class for the policy (player) that is only used for acting.
    """
    def __init__(self, *, policy, nbatch_act, model_index=0):
        self.sess = sess = get_session()
        self.model_index = model_index

        with tf.variable_scope('ppo2_act_model%s'%model_index):
            # CREATE OUR TWO MODELS
            # act_model that is used for sampling
            act_model = policy(nbatch_act, sess)

        self.act_model = act_model
        self.step = act_model.step
        self.initial_state = act_model.initial_state

        self.save = functools.partial(save_trainable_variables, variables=tf.trainable_variables(scope="ppo2_act_model%s"%model_index), sess=sess)
        self.load = functools.partial(load_trainable_variables, variables=tf.trainable_variables(scope="ppo2_act_model%s"%model_index), sess=sess)


class AbstractEnvRunner(ABC):
    def __init__(self, *, env, models, n_steps, nplayer):
        self.env = env
        self.models = models
        self.nenv = nenv = self.env.num_envs
        self.nplayer = nplayer

        # These games do not take observation, make all the observations as zero.
        self.obs = np.zeros((nenv,) + (len(env.observation_space.spaces),),
                            dtype=models[0].train_model.X.dtype.name)
        self.obs[:] = env.reset()
        self.n_steps = n_steps

        # These games do not have states, mark all the states as None.
        self.states = [model.initial_state for model in self.models]

        # These games are one step games, mark all the done as False here.
        self.dones = np.array([[False for _ in range(self.nplayer)] for _ in range(self.nenv)])

    @abstractmethod
    def run(self):
        raise NotImplementedError


class Runner(AbstractEnvRunner):
    """
    Conduct minimax play using the current agents in the environment and collect trajectories.
    """
    def __init__(self, *, env, models, n_steps, nplayer, gamma, lam):
        super().__init__(env=env, models=models, n_steps=n_steps, nplayer=nplayer)
        # Lambda used in GAE (General Advantage Estimation)
        self.lam = lam
        # Discount rate
        self.gamma = gamma

    def run(self):

        def sf01(arr):
            """
            swap and then flatten axes 0 and 1
            """
            s = arr.shape
            return arr.swapaxes(0, 1).reshape(s[0] * s[1], *s[2:])

        mb_obs, mb_rewards, mb_actions, mb_values, mb_dones, mb_neglogpacs = [], [], [], [], [], []
        mb_states = self.states

        for _ in range(self.n_steps):
            all_actions = []
            act_model_0 = self.models[0].act_model
            act_model_1 = self.models[1].act_model

            # give zero observations, none states, false done -> action [nenv, ], value [nenv, ], self.states None, neglogpacs [nenv. ]
            actions_0, values_0, self.states[0], neglogpacs_0 = act_model_0.step(self.obs[:, 0],
                                                                                 S=self.states[0],
                                                                                 M=self.dones[:, 0])

            actions_1, values_1, self.states[1], neglogpacs_1 = act_model_1.step(self.obs[:, 1],
                                                                                 S=self.states[1],
                                                                                 M=self.dones[:, 1])

            mb_obs.append(self.obs.copy()) # self.observation [nenv, nplayer]
            mb_actions.append(np.stack([actions_0, actions_1], axis=1))
            mb_values.append(np.stack([values_0, values_1], axis=1))
            mb_neglogpacs.append(np.stack([neglogpacs_0, neglogpacs_1], axis=1))
            mb_dones.append(self.dones) # self.dones [nenv, nplayer]

            all_actions.append(actions_0)
            all_actions.append(actions_1)

            # Take actions in env and return the reward.
            all_actions = np.stack(all_actions, axis=1) # all_actions [nenv, nagent]
            self.obs[:], rewards, self.dones, infos = self.env.step(all_actions) # self.obs [nenv, nagent] (all zeros), rewards [nenv, nagent], done [nenv, nagent] (all true).
            mb_rewards.append(rewards)

        # batch of steps to batch of rollouts
        mb_obs = np.asarray(mb_obs, dtype=self.obs.dtype) # [n_steps, nenv, nplayer]
        mb_rewards = np.asarray(mb_rewards, dtype=np.float32) # [n_steps, nenv, nplayer]

        mb_actions = np.asarray(mb_actions) # [n_steps, nenv, nplayer]
        mb_values = np.asarray(mb_values, dtype=np.float32) # [n_steps, nenv, nplayer]
        mb_neglogpacs = np.asarray(mb_neglogpacs, dtype=np.float32) # [n_steps, nenv, nplayer]
        mb_dones = np.asarray(mb_dones, dtype=np.bool) # [n_steps, nenv, nplayer]

        mb_advs = np.zeros_like(mb_rewards)
        mb_returns = np.zeros_like(mb_rewards)

        for i in range(self.nplayer):
            # discount/bootstrap off value fn
            last_values_i = self.models[i].value(self.obs[:, i], S=self.states[i], M=self.dones[:, i])
            lastgaelam = 0
            for t in reversed(range(self.n_steps)):
                if t == self.n_steps - 1:
                    nextnonterminal = 1.0 - self.dones[:, i]
                    nextvalues = last_values_i
                else:
                    nextnonterminal = 1.0 - mb_dones[t + 1, :, i]
                    nextvalues = mb_values[t + 1, :, i]
                delta = mb_rewards[t, :, i] + self.gamma * nextvalues * nextnonterminal - mb_values[t, :, i]
                mb_advs[t, :, i] = lastgaelam = delta + self.gamma * self.lam * nextnonterminal * lastgaelam
            mb_returns[:, :, i] = mb_advs[:, :, i] + mb_values[:, :, i]

        return (*map(sf01, (mb_obs, mb_returns, mb_rewards, mb_dones, mb_actions, mb_values, mb_neglogpacs)),
                mb_states)


def learn(*, env_name, env, total_timesteps, out_dir, n_steps, ent_coef=0.0, lr=1e-3, vf_coef=0.5,
          max_grad_norm=0.5, gamma=0.995, lam=0.95, log_interval=1, nminibatches=64, noptepochs=6, cliprange=0.2,
          save_interval=1, nagents=5, inneriter=10, **network_kwargs):

    nenvs = env.num_envs
    nbatch = nenvs*n_steps
    nbatch_train = nbatch // nminibatches

    fake_env = FakeSingleSpacesVec(env)

    # train models, two players.
    models_0 = []
    models_1 = []

    # initialize n number of policies for both party.

    policy = build_policy(fake_env, env_name, **network_kwargs)

    for i in range(nagents):
        model = Model(policy=policy, nbatch_act=nenvs, nbatch_train=nbatch_train, ent_coef=ent_coef, vf_coef=vf_coef,
                      max_grad_norm=max_grad_norm, model_index='0_'+str(i))
        models_0.append(model)

    for i in range(nagents):
        model = Model(policy=policy, nbatch_act=nenvs, nbatch_train=nbatch_train, ent_coef=ent_coef, vf_coef=vf_coef,
                      max_grad_norm=max_grad_norm, model_index='1_'+str(i))
        models_1.append(model)

    sess = get_session()
    sess.run(tf.global_variables_initializer())
    uninitialized = sess.run(tf.report_uninitialized_variables())
    assert len(uninitialized) == 0, 'There are uninitialized variables.'

    # Define n * n runner.
    runners = []
    for n0 in range(nagents):
        runners_row = []
        for n1 in range(nagents):
            runner = Runner(env=env, models=[models_0[n0], models_1[n1]], n_steps=n_steps, nplayer=2, gamma=gamma, lam=lam)
            runners_row.append(runner)
        runners.append(runners_row)

    # training process
    nupdates = total_timesteps//nbatch # number of iterations

    for update in range(1, nupdates+1):

        assert nbatch % nminibatches == 0

        mblossvals_0 = []
        rewards_0 = []
        returns_0 = []

        for n in range(nagents):

            # Update the n th agent of player 0
            # play each player from party 1 with the current training player and get the idx of the agent that achieve the highest reward.
            rewards_all = []

            for n1 in range(nagents):
                runner = runners[n][n1]
                # print(runner.models[0].scope)
                # print(runner.models[1].scope)
                _, returns, rewards, _, _, _, _, _ = runner.run()  # shape [nstep*nenv, 2]
                rewards_all.append(np.mean(rewards[:, 1]))
            rewards_all = np.array(rewards_all)

            max_1 = np.argmax(rewards_all)

            # print('*******************')
            # print(max_1)
            runner_update = runners[n][max_1]
            # print(runner_update.models[0].scope)
            # print(runner_update.models[1].scope)
            # print('*******************')

            for iter in range(inneriter):

                obs, returns, rewards, masks, actions, values, neglogpacs, states = runner_update.run()  # shape [nstep*nenv, ]

                obs = obs[:, 0]
                returns = returns[:, 0]
                rewards = rewards[:, 0]
                actions = actions[:, 0]
                values = values[:, 0]
                neglogpacs = neglogpacs[:, 0]

                if iter == inneriter - 1:
                    mblossvals_0_tmp = []

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
                            mblossvals_0_tmp.append(models_0[n].train_step(lr, cliprange, *slices))
                        else:
                            models_0[n].train_step(lr, cliprange, *slices)

                if iter == inneriter - 1:
                    lossvals = np.mean(mblossvals_0_tmp, axis=0)
                    mblossvals_0.append(lossvals)
                    rewards_0.append(np.mean(rewards))
                    returns_0.append(np.mean(returns))

                    if save_interval and (update % save_interval == 0 or update == 1):
                        checkdir = os.path.join(out_dir, 'checkpoints', 'model_0_%d'%n, '%.5i'%update)
                        os.makedirs(checkdir, exist_ok=True)
                        savepath = os.path.join(checkdir, 'model')
                        models_0[n].save(savepath)

        mblossvals_1 = []
        rewards_1 = []
        returns_1 = []

        for n in range(nagents):
            # Update the n th agent of party 1
            # play each player from party 0 with the current training player and get the idx of the agent that achieve the highest reward.
            rewards_all = []

            for n0 in range(nagents):
                runner = runners[n0][n]
                # print(runner.models[0].scope)
                # print(runner.models[1].scope)
                _, returns, rewards, _, _, _, _, _ = runner.run()  # shape [nstep*nenv, 2]
                rewards_all.append(np.mean(rewards[:, 0]))
            rewards_all = np.array(rewards_all)

            max_0 = np.argmax(rewards_all) # rewards is [n, ]

            runner_update = runners[max_0][n]

            # print('*******************')
            # print(max_0)
            # print(runner_update.models[0].scope)
            # print(runner_update.models[1].scope)
            # print('*******************')

            for iter in range(inneriter):

                obs, returns, rewards, masks, actions, values, neglogpacs, states = runner_update.run()  # shape [nstep*nenv, ]

                obs = obs[:, 1]
                returns = returns[:, 1]
                rewards = rewards[:, 1]
                actions = actions[:, 1]
                values = values[:, 1]
                neglogpacs = neglogpacs[:, 1]

                if iter == inneriter - 1:
                    mblossvals_1_tmp = []

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
                            mblossvals_1_tmp.append(models_1[n].train_step(lr, cliprange, *slices))
                        else:
                            models_1[n].train_step(lr, cliprange, *slices)

                if iter == inneriter - 1:
                    lossvals = np.mean(mblossvals_1_tmp, axis=0)
                    mblossvals_1.append(lossvals)
                    rewards_1.append(np.mean(rewards))
                    returns_1.append(np.mean(returns))

                    if save_interval and (update % save_interval == 0 or update == 1):
                        checkdir = os.path.join(out_dir, 'checkpoints', 'model_1_%d' % n, '%.5i' % update)
                        os.makedirs(checkdir, exist_ok=True)
                        savepath = os.path.join(checkdir, 'model')
                        models_1[n].save(savepath)

        if update % log_interval == 0 or update == 1:
            logger.info('Done.')

            for n in range(nagents):
                logger.logkv('Returns: %d th in 0' % n, np.mean(returns_0[n]))
                logger.logkv('Rewards: %d th in 0' % n, np.mean(rewards_0[n]))
                if env_name == 'CC' or env_name == 'NCNC' or env_name=='As_CC':
                    logger.logkv('V: %d th in 0' % n, models_0[n].log_p()[0])
                else:
                    logger.logkv('Head: %d th in 0' % n, softmax(models_0[n].log_p())[0])

                for (lossval, lossname) in zip(mblossvals_0[n], models_0[n].loss_names):
                    logger.logkv('Loss: %d th in 0' % n + '/' + lossname, lossval)

            for n in range(nagents):
                logger.logkv('Returns: %d th in 1' % n, np.mean(returns_1[n]))
                logger.logkv('Rewards: %d th in 1' % n, np.mean(rewards_1[n]))
                if env_name == 'CC' or env_name == 'NCNC' or env_name=='As_CC':
                    logger.logkv('V: %d th in 1' % n, models_1[n].log_p()[0])
                else:
                    logger.logkv('Head: %d th in 1' % n, softmax(models_1[n].log_p())[0])

                for (lossval, lossname) in zip(mblossvals_1[n], models_1[n].loss_names):
                    logger.logkv('Loss: %d th in 1' % n + '/' + lossname, lossval)

            logger.logkv("nupdates", update)
            logger.logkv("total_timesteps", update * nbatch)
            logger.logkv('learning_rate', lr)
            logger.dumpkvs()

    # select the final best agent.
    rewards_0 = np.zeros((nagents, nagents))
    rewards_1 = np.zeros((nagents, nagents))

    for n0 in range(nagents):
        for n1 in range(nagents):
            runner = runners[n0][n1]
            _, returns, rewards, _, _, _, _, _ = runner.run()  # shape [nstep*nenv, 2]
            rewards_0[n0, n1] = np.mean(rewards[:, 0])
            rewards_1[n0, n1] = np.mean(rewards[:, 1])

    # compute the mean reward of each x agent - row mean of reward.
    # compute the mean reward of each y agent - column mean of reward.

    rewards_0 = np.mean(rewards_0, axis=1)
    rewards_1 = np.mean(rewards_1, axis=0)

    best_0 = np.argmax(rewards_0)
    best_1 = np.argmax(rewards_1)

    return best_0, best_1


