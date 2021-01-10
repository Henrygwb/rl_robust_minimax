import gym
import gym_compete
import numpy as np
from zoo_utils import load_rms
from ray.rllib.env.multi_agent_env import MultiAgentEnv
from scheduling import ConditionalAnnealer, ConstantAnnealer, LinearAnnealer, Scheduler

# Create multi-env

env_list = ["multicomp/YouShallNotPassHumans-v0", "multicomp/KickAndDefend-v0",
            "multicomp/SumoAnts-v0", "multicomp/SumoHumans-v0"]


# Note that, both the rewards and observations are normalized in the env
class MuJoCo_Env(MultiAgentEnv):
    def __init__(self, config):

        self._env = gym.make(config['env_name'])
        self.action_space = self._env.action_space.spaces[0]
        self.observation_space = self._env.observation_space.spaces[0]

        self.gamma = config['gamma']
        self.epsilon = config['epsilon']
        self.clip_reward = config['clip_reward']
        self.shaping_params = {'weights': {'dense': {'reward_move': config['reward_move']},
                               'sparse': {'reward_remaining': config['reward_remaining']}},
                               'anneal_frac': config['anneal_frac'], 'anneal_type': config['anneal_type']}
        #self.scheduler = Scheduler(annealer_dict={'lr': ConstantAnnealer(config['lr'])})
        self.scheduler = Scheduler()

        self.cnt = 0 # Current training steps (Used for linear annealing).
        self.total_step = config['total_step'] # Total number of steps (Used for linear annealing).

        self.normalize = config['normalize']
        self.load_pretrained_model = config['LOAD_PRETRAINED_MODEL']
        self.obs_norm_path = config['obs_norm_path']

        self.debug = config['debug']

        # Define return normalization for both parties.
        # Based on the implementation of VecNormal of stable baseline, normalize the return
        # rather than reward. Maintain a running mean and variance for each env.
        # Since the rewards will be highly correlated, we don't synchronize it.

        if self.load_pretrained_model:
            (mean, std, count), _ = load_rms(self.obs_norm_path)
            self.ret_rms_0 = RunningMeanStd(mean=mean, var=np.square(std), count=count, shape=())
            self.ret_rms_1 = RunningMeanStd(mean=mean, var=np.square(std), count=count, shape=())
        else:
            self.ret_rms_0 = RunningMeanStd(shape=())
            self.ret_rms_1 = RunningMeanStd(shape=())

        # Initialize the return (total discounted reward) for both parties.
        self.ret_0 = np.zeros(1)
        self.ret_1 = np.zeros(1)

        # Track wining information
        # Dimension 0: win 0
        # Dimension 1: win 1
        # Dimension 2: tie
        self.track_winner_info = np.zeros(3)

    # Return the win info, will be called in the custom_eval_function
    def get_winner_info(self):
        return self.track_winner_info

    # Reset the win info, will be called in the custom_eval_function
    def set_winner_info(self):
        self.track_winner_info = np.zeros(3)

    def step(self, action_dict):

        # action is dic
        assert isinstance(action_dict, dict)==True

        action_0 = action_dict['agent_0']
        action_1 = action_dict['agent_1']

        action = (action_0, action_1)

        obs, rewards, done, infos = self._env.step(action)
        
        if self.debug:
           self._env.render()

        # The Done given by the env is a bool variable, make it as a tuple.
        dones = (done, done)

        # Reward shapping.
        self.cnt += 20
        frac_remaining = max(1 - self.cnt / self.total_step, 0)

        reward_0 = apply_reward_shapping(infos[0], self.shaping_params, self.scheduler, frac_remaining)
        reward_1 = apply_reward_shapping(infos[1], self.shaping_params, self.scheduler, frac_remaining)

        # Reward normalization.
        if self.normalize:
            # Update return.
            self.ret_0 = self.ret_0 * self.gamma + reward_0
            self.ret_1 = self.ret_1 * self.gamma + reward_1
            reward_0, reward_1 = self._normalize_(self.ret_0, self.ret_1, reward_0, reward_1)
            if dones[0]:
                self.ret_0[0] = 0
                self.ret_1[0] = 0

        # Setup return dic and set agent IDs.
        # (Agents are indicated by the agent IDs. Return of MultiAgentEnv should be dic).
        obs_dict = {'agent_0': obs[0], 'agent_1': obs[1]}
        reward_dict = {'agent_0': reward_0, 'agent_1': reward_1}
        dones_dict = {'agent_0': done, 'agent_1': done}
        dones_dict['__all__'] = done

        # Update the wining information.
        if done:
            if 'winner' in infos[0]:
                self.track_winner_info[0] += 1
            elif 'winner' in infos[1]:
                self.track_winner_info[1] += 1
            else:
                self.track_winner_info[2] += 1

        return obs_dict, reward_dict, dones_dict, {}

    def reset(self):
        self.ret_0 = np.zeros(1)
        self.ret_1 = np.zeros(1)
        obs = self._env.reset()

        return {'agent_0': obs[0], 'agent_1': obs[1]}

    def _normalize_(self, ret_0, ret_1, reward_0, reward_1):
        """
        :param: obs: observation.
        :param: ret: return.
        :param: reward: reward.
        :return: obs: normalized and cliped observation.
        :return: reward: normalized and cliped reward.
        """
        self.ret_rms_0.update(ret_0)
        self.ret_rms_1.update(ret_1)
        #
        reward_0 = np.clip(reward_0 / np.sqrt(self.ret_rms_0.var + self.epsilon), -self.clip_reward, self.clip_reward)
        reward_1 = np.clip(reward_1 / np.sqrt(self.ret_rms_1.var + self.epsilon), -self.clip_reward, self.clip_reward)

        return reward_0, reward_1


def make_create_env(env_class):
    def create_env(config):
        return env_class(config)
    return create_env


REW_TYPES = set(('sparse', 'dense'))


def apply_reward_shapping(infos, shaping_params, scheduler, frac_remaining):
    """ Reward shaping function.
    :param: info: reward returned from the environment.
    :param: shaping_params: reward shaping parameters.
    :param: annealing factor decay schedule.
    :param: linear annealing fraction.
    :return: shaped reward.
    """

    def _anneal(reward_dict, reward_annealer, frac_remaining):
        c = reward_annealer(frac_remaining)
        assert 0 <= c <= 1
        # print('c is -----------------', c)
        sparse_weight = 1 - c
        dense_weight = c

        return (reward_dict['sparse'] * sparse_weight
                + reward_dict['dense'] * dense_weight)

    if 'metric' in shaping_params:
        rew_shape_annealer = ConditionalAnnealer.from_dict(shaping_params, get_logs=None)
        scheduler.set_conditional('rew_shape')
    else:
        anneal_frac = shaping_params.get('anneal_frac')
        if shaping_params.get('anneal_type')==0:
            rew_shape_annealer = ConstantAnnealer(anneal_frac) # Output anneal_frac as c in _anneal
        else:
            rew_shape_annealer = LinearAnnealer(1, 0, anneal_frac) # anneal_frac should be 1.

    scheduler.set_annealer('rew_shape', rew_shape_annealer)
    reward_annealer = scheduler.get_annealer('rew_shape')
    shaping_params = shaping_params['weights']

    assert shaping_params.keys() == REW_TYPES
    new_shaping_params = {}

    for rew_type, params in shaping_params.items():
        for rew_term, weight in params.items():
            new_shaping_params[rew_term] = (rew_type, weight)

    shaped_reward = {k: 0 for k in REW_TYPES}
    for rew_term, rew_value in infos.items():
        if rew_term not in new_shaping_params:
            continue
        rew_type, weight = new_shaping_params[rew_term]
        shaped_reward[rew_type] += weight * rew_value

    # Compute total shaped reward, optionally annealing
    reward = _anneal(shaped_reward, reward_annealer, frac_remaining)
    return reward


class RunningMeanStd(object):
    # https://en.wikipedia.org/wiki/Algorithms_for_calculating_variance#Parallel_algorithm
    def __init__(self, mean=None, var=None, count=None, epsilon=1e-4, shape=()):
        if mean==None:
            self.mean = np.zeros(shape, 'float64')
        else:
            self.mean = mean

        if var==None:
            self.var = np.ones(shape, 'float64')
        else:
            self.var = var

        if count==None:
            self.count = epsilon
        else:
            self.count = count

    def update(self, x):
        batch_mean = np.mean(x, axis=0)
        batch_var = np.var(x, axis=0)
        batch_count = x.shape[0]
        self.update_from_moments(batch_mean, batch_var, batch_count)

    def update_from_moments(self, batch_mean, batch_var, batch_count):
        self.mean, self.var, self.count = self.update_mean_var_count_from_moments(
            self.mean, self.var, self.count, batch_mean, batch_var, batch_count)

    def update_with_other(self, other):
        self.mean, self.var, self.count = self.update_mean_var_count_from_moments(
            self.mean, self.var, self.count, other.mean, other.var, other.count)

    @staticmethod
    def update_mean_var_count_from_moments(mean, var, count, batch_mean, batch_var, batch_count):
        delta = batch_mean - mean
        tot_count = count + batch_count

        new_mean = mean + delta * batch_count / tot_count
        m_a = var * count
        m_b = batch_var * batch_count
        M2 = m_a + m_b + np.square(delta) * count * batch_count / tot_count
        new_var = M2 / tot_count
        new_count = tot_count

        return new_mean, new_var, new_count

