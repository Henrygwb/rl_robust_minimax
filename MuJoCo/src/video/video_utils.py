from copy import deepcopy
import ray
from ray.rllib.agents.ppo.ppo import PPOTrainer
from ray.rllib.models import ModelCatalog
from ray.tune.registry import register_env
from env import MuJoCo_Env
from zoo_utils import LSTM, add_prefix, MLP
import numpy as np
import pickle5 as pickle


def simulate(venv, agent):
    policy_map = agent.workers.local_worker().policy_map
    state_init = {p: m.get_initial_state() for p, m in policy_map.items()}
    use_lstm = {p: len(s) > 0 for p, s in state_init.items()}

    obs = venv.reset()
    agent_states = state_init.copy()

    # Todo init states
    while True:
        #venv.render()
        actions = []
        new_states = []
        multi_obs = {'agent_0': obs[0], 'agent_1': obs[1]}

        for agent_id, a_obs in multi_obs.items():

            policy_id = 'model' if agent_id == 'agent_0' else 'opp_model'
            
            if use_lstm[policy_id]:
                a_action, p_state, _ = agent.compute_action(a_obs,
                                             state=agent_states[policy_id],
                                             policy_id=policy_id)
                agent_states[policy_id] = p_state
            else:
                a_action = agent.compute_action(a_obs, policy_id=policy_id)
            actions.append(a_action)

        obs, rewards, dones, infos = venv.step(actions)

        if dones:
           # reset the agent_states
           obs = venv.reset()
           agent_states = state_init.copy()

        dones = np.array([dones])
        yield obs, rewards, dones, infos


def load_policy(config_path, checkpoint_0, checkpoint_1):

    ModelCatalog.register_custom_model("custom_rnn", LSTM)
    ModelCatalog.register_custom_model("custom_mlp", MLP)
    # register the custom env "MuJoCo_Env"
    register_env("mujoco", lambda config: MuJoCo_Env(config['env_config']))
    with open(config_path, 'rb') as f:
        rllib_config = pickle.load(f)
    rllib_config['num_workers'] = 0
    ray.init()
    agent = PPOTrainer(env=MuJoCo_Env, config=rllib_config)
    # load model
    model = pickle.load(open(checkpoint_0 + '/model', 'rb'))
    filter = pickle.load(open(checkpoint_0 + '/obs_rms', 'rb'))
    agent.workers.foreach_worker(lambda ev: ev.get_policy('model').set_weights(model))
    agent.workers.foreach_worker(lambda ev: ev.filters['model'].sync(filter))

    # load opp model
    opp_model = pickle.load(open(checkpoint_1 + '/model', 'rb'))
    opp_filter = pickle.load(open(checkpoint_1 + '/obs_rms', 'rb'))
    agent.workers.foreach_worker(lambda ev: ev.get_policy('opp_model').set_weights(opp_model))
    agent.workers.foreach_worker(lambda ev: ev.filters['opp_model'].sync(opp_filter))

    return agent