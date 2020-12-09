import os
import time
import logger
import functools
import numpy as np
import tensorflow as tf
from abc import ABC, abstractmethod
from env import FakeSingleSpacesVec
from baselines.common.tf_util import get_session
from baselines.common import explained_variance
from zoo_utils import save_trainable_variables, load_trainable_variables, build_policy
from scipy.special import softmax


class Model(object):
    """
    Model class for the policy (player) that is trained.
    Create train and act models, PPO objective function.
    """
    def __init__(self, *, policy, nbatch_act, nbatch_train, ent_coef, vf_coef, max_grad_norm,
                 microbatch_size=None, model_index=0):

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

    def _train_step(self, lr, cliprange, obs, returns, actions, values, neglogpacs):
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
        scope = "ppo2_act_model%s"%model_index

        self.save = functools.partial(save_trainable_variables,variables=tf.trainable_variables(scope=scope), sess=sess)
        self.load = functools.partial(load_trainable_variables,variables=tf.trainable_variables(scope=scope), sess=sess)


class AbstractEnvRunner(ABC):
    def __init__(self, *, env, model, opp_model, nagent):
        self.env = env
        self.model = model
        self.opp_model = opp_model
        self.nenv = nenv = self.env.num_envs
        self.nagent = nagent

        self.obs = np.zeros((nenv,) + (len(env.observation_space.spaces),),
                            dtype=model.train_model.X.dtype.name)
        self.obs[:] = env.reset()
        self.states = [model.initial_state, model.initial_state]
        self.dones = np.array([[False for _ in range(nagent)] for _ in range(self.nenv)])

    @abstractmethod
    def run(self):
        raise NotImplementedError


class Runner(AbstractEnvRunner):
    """
    Conduct selfploy using the current agent in the environment and collect trajectories.
    """
    def __init__(self, *, env, model, opp_model, nagent, n_steps, gamma, lam, id):
        super().__init__(env=env, model=model, opp_model=opp_model, nagent=nagent)
        self.lam = lam # Lambda used in GAE (General Advantage Estimation)
        self.gamma = gamma # Discount rate
        self.n_steps = n_steps
        self.id = id

    @staticmethod
    def sf01(arr):
        """
        swap and then flatten axes 0 and 1
        """
        s = arr.shape
        return arr.swapaxes(0, 1).reshape(s[0] * s[1], *s[2:])

    def run(self):

        mb_obs, mb_rewards, mb_actions, mb_values, mb_dones, mb_neglogpacs = [], [], [], [], [], []
        mb_states = self.states

        for _ in range(self.n_steps):
            all_actions = []
            for agt in range(self.nagent):
                if agt == self.id:
                    act_model = self.model # the training agent uses the act model in the model class.
                else:
                    act_model = self.opp_model # the act agent uses the act model in the Act_model class.

                # give zero observations, none states, false done -> action [nenv, ], value [nenv, ], self.states None, neglogpacs [nenv. ]
                actions, values, self.states[agt], neglogpacs = act_model.step(self.obs[:, agt],
                                                                               S=self.states[agt],
                                                                               M=self.dones[:, agt])
                if agt == self.id:
                    mb_obs.append(self.obs[:, agt].copy()) # self.observation [nenv, nagent], self.observation[:, id] [nenv, ]
                    mb_actions.append(actions)
                    mb_values.append(values)
                    mb_neglogpacs.append(neglogpacs)
                    mb_dones.append(self.dones[:, agt]) # self.dones [nenv, nagent], self.dones[:, id] [nenv, ]
                all_actions.append(actions) # actions of every agent in the game.

            # Take actions in env and and return the reward.
            all_actions = np.stack(all_actions, axis=1) # all_actions [nenv, nagent]
            self.obs[:], rewards, self.dones, infos = self.env.step(all_actions) # self.obs [nenv, nagent] (all zeros), rewards [nenv, nagent], done [nenv, nagent] (all true).
            mb_rewards.append(rewards[:, self.id])

        # batch of steps to batch of rollouts
        mb_obs = np.asarray(mb_obs, dtype=self.obs.dtype)  # [n_steps, nenv]
        mb_rewards = np.asarray(mb_rewards, dtype=np.float32)  # [n_steps, nenv]

        mb_actions = np.asarray(mb_actions)  # [n_steps, nenv]
        mb_values = np.asarray(mb_values, dtype=np.float32)  # [n_steps, nenv]
        mb_neglogpacs = np.asarray(mb_neglogpacs, dtype=np.float32)  # [n_steps, nenv]
        mb_dones = np.asarray(mb_dones, dtype=np.bool)  # [n_steps, nenv]
        last_values = self.model.value(self.obs[:, self.id], S=self.states[self.id], M=self.dones[:, self.id])  # [nenv, ]

        # Discount/bootstrap off value fn
        mb_advs = np.zeros_like(mb_rewards)
        lastgaelam = 0
        for t in reversed(range(self.n_steps)):
            if t == self.n_steps - 1:
                nextnonterminal = 1.0 - self.dones[:, self.id]
                nextvalues = last_values
            else:
                nextnonterminal = 1.0 - mb_dones[t + 1]
                nextvalues = mb_values[t + 1]
            delta = mb_rewards[t] + self.gamma * nextvalues * nextnonterminal - mb_values[t]
            mb_advs[t] = lastgaelam = delta + self.gamma * self.lam * nextnonterminal * lastgaelam
        mb_returns = mb_advs + mb_values

        return (*map(self.sf01, (mb_obs, mb_returns, mb_rewards, mb_dones, mb_actions, mb_values, mb_neglogpacs)),
                mb_states[self.id])


def learn(*, env_name, env, nagent=2, opp_method=0, total_timesteps=20000000, n_steps=1024, nminibatches=4, noptepochs=4,
          ent_coef=0.0, vf_coef=0.5, max_grad_norm=0.5, gamma=0.99, lam=0.95, lr=1e-3, cliprange=0.2, log_interval=1,
          save_interval=1, out_dir='', **network_kwargs):

    nenvs = env.num_envs
    nbatch = nenvs*n_steps
    nbatch_train = nbatch // nminibatches

    fake_env = FakeSingleSpacesVec(env)

    policy = build_policy(fake_env, env_name, **network_kwargs)

    # Define the policy used for training.
    model = Model(policy=policy, nbatch_act=nenvs, nbatch_train=nbatch_train, ent_coef=ent_coef, vf_coef=vf_coef,
                  max_grad_norm=max_grad_norm, model_index=0)

    # Define the opponent policy that only used for action.
    opp_model = Act_Model(policy=policy, nbatch_act=nenvs, model_index=1)

    sess = get_session()
    sess.run(tf.global_variables_initializer())
    uninitialized = sess.run(tf.report_uninitialized_variables())
    assert len(uninitialized) == 0, 'There are uninitialized variables.'

    # training process
    # number of iterations
    nupdates = total_timesteps//nbatch

    # Define the runner for the agent under training.
    runner = Runner(env=env, model=model, opp_model=opp_model, nagent=nagent, n_steps=n_steps,
                    gamma=gamma, lam=lam, id=0)

    for update in range(1, nupdates+1):
        assert nbatch % nminibatches == 0

        # Set the opponent model
        if update == 1:
            if update % log_interval == 0:
                logger.info('First iteration. Using a random agent as the opponent.')
        else:
            if opp_method == 0:
                logger.info('Select the latest model.')
                selected_opp = update - 1

            elif opp_method == 1:
                logger.info('Select the latest model.')
                selected_opp = round(np.random.uniform(1, update - 1))

            else:
                logger.info('Select the latest model.')
                selected_opp = update - 1

            model_path = os.path.join(out_dir, 'checkpoints', 'model', '%.5i'%selected_opp, 'model')
            opp_model.load(model_path)

        obs, returns, rewards, masks, actions, values, neglogpacs, states = runner.run() # shape [nstep*nenv, ]

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
                slices = (arr[mbinds] for arr in (obs, returns, actions, values, neglogpacs))
                mblossvals.append(model._train_step(lr, cliprange, *slices))

        lossvals = np.mean(mblossvals, axis=0)

        if update % log_interval == 0 or update == 1:
            # Calculates if value function is a good predicator of the returns (ev > 1)
            # or if it's just worse than predicting nothing (ev =< 0)
            ev = explained_variance(values, returns)
            logger.logkv("serial_timesteps", update)
            logger.logkv("nupdates", update)
            logger.logkv("total_timesteps", update*nbatch)
            logger.logkv("explained_variance", float(ev))
            logger.logkv('learning_rate', lr)
            logger.logkv('returns', np.mean(returns))
            logger.logkv('rewards', np.mean(rewards))
            if env_name == 'CC' or env_name == 'NCNC':
                logger.logkv('v', model.log_p()[0])
            else:
                logger.logkv('head', softmax(model.log_p())[0])
            for (lossval, lossname) in zip(lossvals, model.loss_names):
                 logger.logkv('loss' + '/' + lossname, lossval)
            logger.dumpkvs()

        if save_interval and (update % save_interval == 0 or update == 1):
            checkdir = os.path.join(out_dir, 'checkpoints', 'model_%d', '%.5i'%update)
            os.makedirs(checkdir, exist_ok=True)
            savepath = os.path.join(checkdir, 'model')
            model.save(savepath)

    return 0
