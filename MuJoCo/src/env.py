import gym
import pickle
import gym_compete
import numpy as np
from zoo_utils import load_rms
from victim_agent import load_victim_agent
from ray.rllib.env.multi_agent_env import MultiAgentEnv
from scheduling import ConditionalAnnealer, ConstantAnnealer, LinearAnnealer, Scheduler

# Create multi-env
env_list = ["multicomp/YouShallNotPassHumans-v0", "multicomp/KickAndDefend-v0",
            "multicomp/SumoAnts-v0", "multicomp/SumoHumans-v0"]


# Selfplay environment. Note that, both the rewards and observations are normalized in the env
class MuJoCo_Env(MultiAgentEnv):
    def __init__(self, config):

        self._env = gym.make(config['env_name'])
        self.action_space = self._env.action_space.spaces[0]
        self.observation_space = self._env.observation_space.spaces[0]

        self.gamma = config['gamma']
        self.epsilon = config['epsilon']
        self.clip_reward = config['clip_rewards']
        self.shaping_params = {'weights': {'dense': {'reward_move': config['reward_move']},
                               'sparse': {'reward_remaining': config['reward_remaining']}},
                               'anneal_frac': config['anneal_frac'], 'anneal_type': config['anneal_type']}
        # self.scheduler = Scheduler(annealer_dict={'lr': ConstantAnnealer(config['lr'])})
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
            (mean_0, std_0, count_0), _ = load_rms(self.obs_norm_path[0])
            (mean_1, std_1, count_1), _ = load_rms(self.obs_norm_path[1])
            self.ret_rms_0 = RunningMeanStd(mean=mean_0, var=np.square(std_0), count=count_0, shape=())
            self.ret_rms_1 = RunningMeanStd(mean=mean_1, var=np.square(std_1), count=count_1, shape=())
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


# Environment for adversarial attack.
class Adv_Env(gym.Env):

    def __init__(self, config):

        self._env = gym.make(config['env_name'])

        self.env_name = config['env_name']
        # Define the action space and observation space.
        self.action_space = self._env.action_space.spaces[0]
        self.observation_space = self._env.observation_space.spaces[0]

        self.epsilon = config['epsilon']
        self.clip_reward = config['clip_rewards']
        self.normalize = config['normalization']
        self.gamma = config['gamma']
        self.debug = config['debug']

        # Params related with victim-agent.
        # Initial victim party id.
        self.victim_index = config['victim_party_id']
        # Initial victim model path.
        self.victim_model_path = config['victim_model_path']

        # load shaping_params and schedule.
        self.shaping_params = {'weights': {'dense': {'reward_move': config['reward_move']},
                               'sparse': {'reward_remaining': config['reward_remaining']}},
                               'anneal_frac': config['anneal_frac'], 'anneal_type': config['anneal_type']}

        self.scheduler = Scheduler()

        # normalize the rets.
        if self.normalize:
            self.ret_rms = RunningMeanStd(shape=())

        # return - total discounted reward.
        self.ret = np.zeros(1)

        # track wining information.
        # 0: win 0, 1: win 1, 2: tie.
        self.track_winner_info = np.zeros(3)
        self.cnt = 0
        self.total_step = config['total_step']

        # construct the victim agent.
        self.victim_agent = load_victim_agent(self.env_name, self.observation_space,
                                              self.action_space, self.victim_model_path + '/model')
        self.filter = pickle.load(open(self.victim_model_path + '/obs_rms', 'rb'))

    # return the win info, will be called in the custom_eval_function.
    def get_winner_info(self):
        return self.track_winner_info

    # reset the win info, will be called in the custom_eval_function.
    def set_winner_info(self):
        self.track_winner_info = np.zeros(3)

    def step(self, action):

        norm_ob = self.filter(self.ob, update=False) # https://github.com/ray-project/ray/blob/master/rllib/utils/filter.py
        self.action = self.victim_agent.act(stochastic=False, observation=norm_ob)[0]
        # The evaluation code of the ICLR'18 selfplay paper doesn't clip the action of both parties.
        # The existing adv attack implementation clips only the action of the adv parties.
        # Here we clip the actions of both parties to keep consistent with our selfplay/minimax implementation.
        # We test four different cases with the ICLR'18 YouShallNotPass agents:
        # Clip both parties, clip party 0, clip party 1, No clip at all:
        # The winning rate of party 0 are: 0.468, 0.486, 0.494, 0.472.
        self.action = np.clip(self.action, self.action_space.low, self.action_space.high)

        if self.victim_index == 0:
            actions = (self.action, action)
        else:
            actions = (action, self.action)

        obs, rewards, done, infos = self._env.step(actions)

        if self.victim_index == 0:
            self.ob, ob = obs
        else:
            ob, self.ob = obs

        if self.debug:
           self._env.render()

        #reward shapping
        self.cnt += 20
        frac_remaining = max(1 - self.cnt / self.total_step, 0)

        reward = apply_reward_shapping(infos[1-self.victim_index], self.shaping_params, self.scheduler, frac_remaining)

        # normalize the adversarial reward.
        if self.normalize:
            self.ret = self.ret * self.gamma + reward
            reward = self._normalize_(self.ret, reward)
            if done:
                self.ret[0] = 0

        # update the wining information
        if done:
            if 'winner' in infos[0]:
                self.track_winner_info[0] += 1
            elif 'winner' in infos[1]:
                self.track_winner_info[1] += 1
            else:
                self.track_winner_info[2] += 1
            self.victim_agent.reset()

        return ob, reward, done, {}

    def reset(self):
        self.ret = np.zeros(1)
        obs = self._env.reset()

        if self.victim_index == 0:
            self.ob, ob = obs
        else:
            ob, self.ob = obs

        self.victim_agent.reset()
        return ob

    def _normalize_(self, ret, reward):

        self.ret_rms.update(ret)
        reward = np.clip(reward / np.sqrt(self.ret_rms.var + self.epsilon), -self.clip_reward, self.clip_reward)
        return reward


# MiniMax Training
# Create one trainer for all the agents. In each iteration, update the agents in one party.
# In this environment, we define 2*"num_agents_per_party" agents, the IDs of which are defined in reset function.
# The agents are then mapped with the defined models by using the policy_mapping_fn().
# agent_i -> model_i; opp_agent_i -> opp_model_i.
# In our env, we play agent_i and opp_agent_i acts in self._envs[i] and collect trajectories for
# model_i and opp_model_i. As such, we create "num_agents_per_party" number of gym envs in the env.


class Minimax_Env(MultiAgentEnv):
    def __init__(self, config):
        self.num_agents_per_party = config['num_agents_per_party']
        self._envs = [gym.make(config['env_name']) for _ in range(self.num_agents_per_party)]

        self.action_space = self._envs[0].action_space.spaces[0]
        self.observation_space = self._envs[0].observation_space.spaces[0]

        self.gamma = config['gamma']
        self.epsilon = config['epsilon']
        self.clip_reward = config['clip_rewards']
        self.shaping_params = {'weights': {'dense': {'reward_move': config['reward_move']},
                                           'sparse': {'reward_remaining': config['reward_remaining']}},
                               'anneal_frac': config['anneal_frac'], 'anneal_type': config['anneal_type']}
        self.scheduler = Scheduler()

        self.cnt = 0  # Current training steps (Used for linear annealing).
        self.total_step = config['total_step']  # Total number of steps (Used for linear annealing).

        self.normalize = config['normalize']
        self.load_pretrained_model = config['LOAD_PRETRAINED_MODEL']
        self.obs_norm_path = config['obs_norm_path']

        self.debug = config['debug']

        # Define return normalization for each agent in each party.
        # Based on the implementation of VecNormal of stable baseline, normalize the return
        # rather than reward. Maintain a running mean and variance for each env in each worker.
        # Since the rewards will be highly correlated, we don't synchronize between workers.
        self.ret_rms_0 = []
        self.ret_rms_1 = []

        for i in range(self.num_agents_per_party):
            if self.load_pretrained_model:
                (mean_0, std_0, count_0), _ = load_rms(self.obs_norm_path[0])
                (mean_1, std_1, count_1), _ = load_rms(self.obs_norm_path[1])
                self.ret_rms_0.append(RunningMeanStd(mean=mean_0, var=np.square(std_0), count=count_0, shape=()))
                self.ret_rms_1.append(RunningMeanStd(mean=mean_1, var=np.square(std_1), count=count_1, shape=()))
            else:
                self.ret_rms_0.append(RunningMeanStd(shape=()))
                self.ret_rms_1.append(RunningMeanStd(shape=()))

        # Initialize the return (total discounted reward) for each agent in each party.
        self.ret_0 = np.zeros(self.num_agents_per_party)
        self.ret_1 = np.zeros(self.num_agents_per_party)

        # Track the wining information for each env.
        # Dimension 0: win 0
        # Dimension 1: win 1
        # Dimension 2: tie
        self.track_winner_info = np.zeros((self.num_agents_per_party, 3))
        self.dones = set()

    # Return the win info, will be called in the custom_eval_function
    def get_winner_info(self):
        return self.track_winner_info

    # Reset the win info, will be called in the custom_eval_function
    def set_winner_info(self):
        self.track_winner_info = np.zeros((self.num_agents_per_party, 3))

    def step(self, action_dict):
        # Ray gives an action_dict with keys as "agent_i" and "opp_agent_i".
        # In this function, we play agent_i and opp_agent_i acts in self._envs[i] and collect trajectories for
        # agent_i and opp_agent_i. We return observations, rewards, dones for the agents involved in ths step.
        # The, the ray will assign the collected data for agent_i/opp_agent_i to model_i/opp_model_i and use the
        # data collected for each agent to train the model.
        # Initially, action is a dict with the length of 2*self.num_agents_per_party.
        # One game episode/trajectory in one env may end earlier than the trajectory in another env.
        # In this case, the action_dict inputed here no longer contains the actions of agents in the env where the
        # game has finished.
        assert isinstance(action_dict, dict)

        # Reward shapping.
        self.cnt += 20
        frac_remaining = max(1 - self.cnt / self.total_step, 0)

        obs_dict = {}
        reward_dict = {}
        dones_dict = {}

        # The number of agents in action_dict keeps changing based on the game ending situation, here we can not
        # use self.num_agents_per_party

        num = int(len(action_dict.keys()) / 2)
        ids = [k.split('_')[-1] for k in action_dict.keys()] # get the id of agents involved in this step
        ids = list(set(ids)) # remove the duplicated numbers.
        # e.g., action_dict.keys() = ['agent_0', 'opp_agent_1', 'agent_1', 'opp_agent_1']
        # ids = ['0', '0', '1', '1'].
        # ids = ['0', '1'].
        # Note that here we fix agent_i to play with opp_agent_i. The changing in opponent can be realized by assigning
        # different models to the opponent party. For example, we can give the previous model of the opp_agent_1 to
        # opp_agent_2 as the opponent of agent_2.

        for i in range(num):
            id = int(ids[i]) # current _env ID.
            key_0 = 'agent_' + ids[i]
            key_1 = 'opp_agent_' + ids[i]

            assert key_0 in action_dict
            assert key_1 in action_dict

            action_0 = action_dict[key_0]
            action_1 = action_dict[key_1]
            action = (action_0, action_1)

            obs, rewards, done, infos = self._envs[id].step(action)

            if self.debug:
                self._envs[id].render()

            # The done given by the env is a bool variable, make it as a tuple.
            dones = (done, done)

            reward_0 = apply_reward_shapping(infos[0], self.shaping_params, self.scheduler, frac_remaining)
            reward_1 = apply_reward_shapping(infos[1], self.shaping_params, self.scheduler, frac_remaining)

            # update obs_dict, reward_dict, dones_dict
            obs_dict[key_0] = obs[0]
            obs_dict[key_1] = obs[1]

            dones_dict[key_0] = done
            dones_dict[key_1] = done

            # Reward normalization.
            if self.normalize:
                # Update return.
                reward_0, reward_1 = self._normalize_(reward_0, reward_1, id)

            reward_dict[key_0] = reward_0
            reward_dict[key_1] = reward_1

            # reset the ret
            if dones_dict[key_0]:
                self.ret_0[id] = 0
                self.ret_1[id] = 0

            # Update the wining information.
            if done:
                self.dones.add(id)
                if 'winner' in infos[0]:
                    self.track_winner_info[id, 0] += 1
                elif 'winner' in infos[1]:
                    self.track_winner_info[id, 1] += 1
                else:
                    self.track_winner_info[id, 2] += 1

        # '__all__': whether all the envs are done. Ray will check this flag to decide to whether reset env or not.
        # When one env is done, we append its id to self.dones and check its length equal to num_agents_per_party.
        dones_dict['__all__'] = len(self.dones) == self.num_agents_per_party

        return obs_dict, reward_dict, dones_dict, {}

    def reset(self):
        # Define the agent id and let agent_i and opp_agent_i run in _envs[i].
        self.dones = set()
        self.ret_0 = np.zeros(self.num_agents_per_party)
        self.ret_1 = np.zeros(self.num_agents_per_party)

        obs_dict = {}
        for i in range(self.num_agents_per_party):
            key_0 = 'agent_' + str(i)
            key_1 = 'opp_agent_' + str(i)
            obs = self._envs[i].reset()
            obs_dict[key_0] = obs[0]
            obs_dict[key_1] = obs[1]
        return obs_dict

    def _normalize_(self, reward_0, reward_1, id):

        self.ret_0[id] = self.ret_0[id] * self.gamma + reward_0
        self.ret_1[id] = self.ret_1[id] * self.gamma + reward_1

        self.ret_rms_0[id].update(self.ret_0[id:id+1])
        self.ret_rms_1[id].update(self.ret_1[id:id+1])

        reward_0 = np.clip(reward_0 / np.sqrt(self.ret_rms_0[id].var + self.epsilon),
                           -self.clip_reward, self.clip_reward)
        reward_1 = np.clip(reward_1 / np.sqrt(self.ret_rms_1[id].var + self.epsilon),
                           -self.clip_reward, self.clip_reward)

        return reward_0, reward_1


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
        #print('c is -----------------', c)
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
