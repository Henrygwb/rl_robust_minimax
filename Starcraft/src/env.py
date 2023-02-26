import gym
import pickle
import gym_compete
import numpy as np
from zoo_utils import load_rms

from gym.spaces import Discrete, Box, Dict

from victim_agent import load_victim_agent
from ray.rllib.env.multi_agent_env import MultiAgentEnv
from envs.selfplay_raw_env import SC2SelfplayRawEnv
from envs.actions.zerg_action_wrappers import ZergPlayerActionWrapper
from envs.observations.zerg_observation_wrappers import ZergPlayerObservationWrapper


def make_env(config):
    random_seed = config['game_seed']
    # Make StarCraft II env
    env = SC2SelfplayRawEnv(map_name='Flat64',
                            step_mul=config['step_mul'],
                            resolution=16,
                            agent_race='zerg',
                            opponent_race='zerg',
                            tie_to_lose=False,
                            disable_fog=config['disable_fog'],
                            game_steps_per_episode=config['game_steps_per_episode'],
                            random_seed=random_seed)

    env = ZergPlayerActionWrapper(
        player=0,
        env=env,
        game_version=config['game_version'],
        mask=config['use_action_mask'],
        use_all_combat_actions=config['use_all_combat_actions'])
    
    env = ZergPlayerObservationWrapper(
        player=0,
        env=env,
        use_spatial_features=False,
        use_game_progress=True,
        action_seq_len=8,
        use_regions=config['use_region_features'])

    env = ZergPlayerActionWrapper(
        player=1,
        env=env,
        game_version=config['game_version'],
        mask=config['use_action_mask'],
        use_all_combat_actions=config['use_all_combat_actions'])

    env = ZergPlayerObservationWrapper(
        player=1,
        env=env,
        use_spatial_features=False,
        use_game_progress=True,
        action_seq_len=8,
        use_regions=config['use_region_features'])
    print(env.observation_space, env.action_space)
    return env

# Selfplay environment. Note that, both the rewards and observations are normalized in the env
class Starcraft_Env(MultiAgentEnv):
    def __init__(self, config):

        self._env = make_env(config)
        self.observation_space = Dict(
            {
                "obs": self._env.observation_space.spaces[0], 
                "action_mask": self._env.observation_space.spaces[1]
            }
        )
        total_actions = int(np.product(self._env.observation_space.spaces[1].shape))

        self.action_space = Discrete(total_actions)
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
        
        assert 'agent_0' in action_dict
        assert 'agent_1' in action_dict

        action_0 = action_dict['agent_0']
        action_1 = action_dict['agent_1']

        action = [action_0, action_1]
        obs, reward, done, info = self._env.step(action)

        # Setup return dic and set agent IDs.
        # (Agents are indicated by the agent IDs. Return of MultiAgentEnv should be dic).
        obs_dict = {'agent_0': {"action_mask": obs[0][1], "obs": obs[0][0]}, 
                    'agent_1': {"action_mask": obs[1][1], "obs": obs[1][0]}}

        reward_dict = {'agent_0': reward, 'agent_1': info['oppo_reward']}
        dones_dict = {'agent_0': done, 'agent_1': done}
        dones_dict['__all__'] = done

        # Update the wining information.
        if done:
            if reward == 1:
                self.track_winner_info[0] += 1
            elif reward == -1:
                self.track_winner_info[1] += 1
            else:
                self.track_winner_info[2] += 1
        return obs_dict, reward_dict, dones_dict, {}

    def reset(self):

        obs = self._env.reset()
        assert type(obs) == tuple
        obs_dict = {'agent_0': {"action_mask": obs[0][1], "obs": obs[0][0]}, 
                    'agent_1': {"action_mask": obs[1][1], "obs": obs[1][0]}}
        return obs_dict
    
    def close(self):
        self._env.close()


# MiniMax Training
# Create one trainer for all the agents. In each iteration, update the agents in one party.
# In this environment, we define 2*"num_agents_per_party" agents, the IDs of which are defined in reset function.
# The agents are then mapped with the defined models by using the policy_mapping_fn().
# agent_i -> model_i; opp_agent_i -> opp_model_i.
# In our env, we play agent_i and opp_agent_i acts in self._envs[i] and collect trajectories for
# model_i and opp_model_i. As such, we create "num_agents_per_party" number of gym envs in the env.

class Minimax_Starcraft_Env(MultiAgentEnv):
    def __init__(self, config):
        self.num_agents_per_party = config['num_agents_per_party']
        self._envs = [make_env(config) for _ in range(self.num_agents_per_party)]

        self.observation_space = Dict(
            {
                "obs": self._envs[0].observation_space.spaces[0], 
                "action_mask": self._envs[0].observation_space.spaces[1]
            }
        )

        total_actions = int(np.product(self._envs[0].observation_space.spaces[1].shape))
        self.action_space = Discrete(total_actions)
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
            action = [action_0, action_1]

            obs, reward, done, info = self._envs[id].step(action)

            # The done given by the env is a bool variable, make it as a tuple.
            dones = (done, done)

            obs_dict[key_0] = {"action_mask": obs[0][1], "obs": obs[0][0]}
            obs_dict[key_1] = {"action_mask": obs[1][1], "obs": obs[1][0]}

            dones_dict[key_0] = done
            dones_dict[key_1] = done

            reward_dict[key_0] = reward
            reward_dict[key_1] = info['oppo_reward']
            # Update the wining information.
            if done:
                self.dones.add(id)
                if reward == 1:
                    self.track_winner_info[id, 0] += 1
                elif reward == -1:
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

        obs_dict = {}
        for i in range(self.num_agents_per_party):
            key_0 = 'agent_' + str(i)
            key_1 = 'opp_agent_' + str(i)
            obs = self._envs[i].reset()

            assert type(obs) == tuple
            obs_dict[key_0] = {"action_mask": obs[0][1], "obs": obs[0][0]}
            obs_dict[key_1] = {"action_mask": obs[1][1], "obs": obs[1][0]}

        return obs_dict

    def close(self):
        # Close the environment
        for i in range(len(self._envs)):
            self._envs[i].close()

# Environment for adversarial attack.

class Adv_Env(gym.Env):

    def __init__(self, config):

        self._env = make_env(config)
        self.observation_space = Dict(
            {
                "obs": self._env.observation_space.spaces[0], 
                "action_mask": self._env.observation_space.spaces[1]
            }
        )
        total_actions = int(np.product(self._env.observation_space.spaces[1].shape))

        self.action_space = Discrete(total_actions)

        # Params related with victim-agent.
        # Initial victim party id.
        self.victim_index = config['victim_party_id']
        # Initial victim model path.
        self.victim_model_path = config['victim_model_path']

        # track wining information.
        # 0: win 0, 1: win 1, 2: tie.
        self.track_winner_info = np.zeros(3)

        # construct the victim agent.
        self.victim_agent = load_victim_agent(self.env_name, self.observation_space,
                                              self.action_space, self.victim_model_path + '/model')

    # return the win info, will be called in the custom_eval_function.
    def get_winner_info(self):
        return self.track_winner_info

    # reset the win info, will be called in the custom_eval_function.
    def set_winner_info(self):
        self.track_winner_info = np.zeros(3)

    def step(self, action):

        self_action, _ = self.victim_agent.step(transform_tuple(self.ob, lambda x: np.expand_dims(x, 0)))
        self.action = self_action[0]
        if self.victim_index == 0:
            actions = [self.action, action]
        else:
            actions = [action, self.action]

        obs, reward, done, infos = self._env.step(actions)

        if self.victim_index == 0:
            self.ob, ob = obs
            reward = infos['oppo_reward']
        else:
            ob, self.ob = obs

        # update the wining information
        if done:
            if 'winner' in infos[0]:
                self.track_winner_info[0] += 1
            elif 'winner' in infos[1]:
                self.track_winner_info[1] += 1
            else:
                self.track_winner_info[2] += 1
            self.victim_agent.reset()

        return {"obs": ob[0], "action_mask": ob[1]}, reward, done, {}

    def reset(self):
        obs = self._env.reset()

        if self.victim_index == 0:
            self.ob, ob = obs
        else:
            ob, self.ob = obs

        self.victim_agent.reset()
        return {"obs": ob[0], "action_mask": ob[1]}


def transform_tuple(x, transformer):
  if isinstance(x, tuple):
    return tuple(transformer(a) for a in x)
  else:
    return transformer(x)