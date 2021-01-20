import ray
import argparse
from copy import deepcopy
from env import Minimax_Env, env_list
from ray.rllib.agents.ppo.ppo_tf_policy import PPOTFPolicy
from ray.rllib.agents.ppo.ppo import PPOTrainer, DEFAULT_CONFIG
from ray.rllib.models import ModelCatalog
from ray.tune.registry import register_env
from zoo_utils import LSTM, MLP, load_pretrain_model, setup_logger, create_mean_std
from ppo_minimax import policy_mapping_fn, minimax_learning

##################
# Hyper-parameters
##################

parser = argparse.ArgumentParser()
# Number of parallel workers/actors.
parser.add_argument("--num_workers", type=int, default=1)

# Number of environments per worker
parser.add_argument("--num_envs_per_worker", type=int, default=1)

# Number of gpus for the training worker.
parser.add_argument("--num_gpus", type=int, default=0)

# Number of gpus for the remote worker.
parser.add_argument("--num_gpus_per_worker", type=int, default=0)

# Number of parallel evaluation workers.
parser.add_argument("--eval_num_workers", type=int, default=1)

# Number of evaluation game rounds.
parser.add_argument("--num_episodes", type=int, default=2)

# Ratio between the number of workers/episodes used for evaluation and best opponent selection.
parser.add_argument("--eval_select_ratio", type=int, default=1)

# ["multicomp/YouShallNotPassHumans-v0", "multicomp/KickAndDefend-v0",
#  "multicomp/SumoAnts-v0", "multicomp/SumoHumans-v0"]
parser.add_argument("--env", type=int, default=0)

# Random seed.
parser.add_argument("--seed", type=int, default=0)

# Number of agents is trained in each party.
parser.add_argument("--num_agents_per_party", type=int, default=2)

# The order of two parties:
# 0: party_0 (model) is in the outer loop -> min_x max_y f(x, y)
# 1: party_1 (opp_model) is in the outer loop  -> max_y min_x f(x, y)
parser.add_argument('--party_order', type=int, default=0)

# Number of updating loops in outer training each iteration.
parser.add_argument('--update_loop', type=int, default=1)

# Number of inner loops for the inner agent inside the inner loops.
parser.add_argument('--inner_loop', type=int, default=2)

# Loading a pretrained model as the initial model or not.
parser.add_argument("--load_pretrained_model", type=bool, default=True)

# # Pretrained normalization and model params path .
# # YouShallNotPass: blocker (saved agent_1) -> agent_0, runner (saved agent_2) -> agent_1
parser.add_argument("--agent_0_obs_norm_path", type=str,
                    default="../initial-agents/YouShallNotPassHumans-v0/agent1-rms-v1.pkl")

parser.add_argument("--agent_0_pretrain_model_path", type=str,
                    default="../initial-agents/YouShallNotPassHumans-v0/agent1-model-v1.pkl")

parser.add_argument("--agent_1_obs_norm_path", type=str,
                    default="../initial-agents/YouShallNotPassHumans-v0/agent2-rms-v1.pkl")

parser.add_argument("--agent_1_pretrain_model_path", type=str,
                    default="../initial-agents/YouShallNotPassHumans-v0/agent2-model-v1.pkl")

# # KickAndDefend: kicker (saved agent_1) -> agent_0, keeper (saved agent_2) -> agent_1

# parser.add_argument("--agent_0_obs_norm_path", type=str,
#                     default="../initial-agents/KickAndDefend-v0/agent1-rms-v1.pkl")
#
# parser.add_argument("--agent_0_pretrain_model_path", type=str,
#                     default="../initial-agents/KickAndDefend-v0/agent1-model-v1.pkl")
#
# parser.add_argument("--agent_1_obs_norm_path", type=str,
#                     default="../initial-agents/KickAndDefend-v0/agent2-rms-v1.pkl")
#
# parser.add_argument("--agent_1_pretrain_model_path", type=str,
#                     default="../initial-agents/KickAndDefend-v0/agent2-model-v1.pkl")

# # SumoAnts.
# parser.add_argument("--agent_0_obs_norm_path", type=str,
#                     default="../initial-agents/SumoAnts-v0/agent0-rms-v1.pkl")
#
# parser.add_argument("--agent_0_pretrain_model_path", type=str,
#                     default="../initial-agents/SumoAnts-v0/agent0-model-v1.pkl")
#
# # Pretrained normalization and model params path for agent 1 (opp_model).
# parser.add_argument("--agent_1_obs_norm_path", type=str,
#                     default="../initial-agents/SumoAnts-v0/agent0-rms-v1.pkl")
#
# parser.add_argument("--agent_1_pretrain_model_path", type=str,
#                     default="../initial-agents/SumoAnts-v0/agent0-model-v1.pkl")

# SumoHumans.
# parser.add_argument("--agent_0_obs_norm_path", type=str,
#                     default="../initial-agents/SumoHumans-v0/agent0-rms-v1.pkl")
#
# parser.add_argument("--agent_0_pretrain_model_path", type=str,
#                     default="../initial-agents/SumoHumans-v0/agent0-model-v1.pkl")
#
# # Pretrained normalization and model params path for agent 1 (opp_model).
# parser.add_argument("--agent_1_obs_norm_path", type=str,
#                     default="../initial-agents/SumoHumans-v0/agent0-rms-v1.pkl")
#
# parser.add_argument("--agent_1_pretrain_model_path", type=str,
#                     default="../initial-agents/SumoHumans-v0/agent0-model-v1.pkl")

parser.add_argument('--debug', type=bool, default=False)

args = parser.parse_args()

# ======= Setting for rollout worker processes =======
# Number of parallel workers/actors.
NUM_WORKERS = args.num_workers
# Number of environments per worker.
NUM_ENV_WORKERS = args.num_envs_per_worker
# Number of gpus for the training worker.
NUM_GPUS = args.num_gpus
# Number of gpus for the remote worker.
NUM_GPUS_PER_WORKER = args.num_gpus_per_worker
# Batch size collected from each worker.
ROLLOUT_FRAGMENT_LENGTH = 100

# === Settings for the training process ===
# Number of epochs in each iteration.
NEPOCH = 5
# Training batch size.
TRAIN_BATCH_SIZE = ROLLOUT_FRAGMENT_LENGTH*NUM_WORKERS*NUM_ENV_WORKERS
# Minibatch size. Num_epoch = train_batch_size/sgd_minibatch_size.
TRAIN_MINIBATCH_SIZE = TRAIN_BATCH_SIZE/NEPOCH
# Number of iterations.
NUPDATES = 4

# Number of agents is trained in each party.
NUM_AGENTS_PER_PARTY = args.num_agents_per_party

# The order of two parties:
# 0: party_0 (model) is in the outer loop -> min_x max_y f(x, y)
# 1: party_1 (opp_model) is in the outer loop  -> max_y min_x f(x, y)
PARTY_ORDER = args.party_order

# Number of updating loops in each iteration for party 0: INNER_LOOP_PARTY_0.
# Number of updating loops in each iteration for party 1: INNER_LOOP_PARTY_1.

if PARTY_ORDER==0:
    INNER_LOOP_PARTY_0 = args.update_loop
    INNER_LOOP_PARTY_1 = args.update_loop * args.inner_loop
else:
    INNER_LOOP_PARTY_0 = args.update_loop * args.inner_loop
    INNER_LOOP_PARTY_1 = args.update_loop

SELECT_NUM_EPISODES = int(args.num_episodes / args.eval_select_ratio)
SELECT_NUM_WOEKER = int(args.eval_num_workers / args.eval_select_ratio)

# === Settings for the pretrained agent and policy network. ===
# Loading a pretrained model as the initial model or not.
LOAD_PRETRAINED_MODEL = args.load_pretrained_model

AGT_0_OBS_NORM_PATH = args.agent_0_obs_norm_path
AGT_0_MODEL_PATH = args.agent_0_pretrain_model_path

AGT_1_OBS_NORM_PATH = args.agent_1_obs_norm_path
AGT_1_MODEL_PATH = args.agent_1_pretrain_model_path

# Whether to use RNN as policy network.
if args.env == 0:
    USE_RNN = False
else:
    USE_RNN = True

# === Environment Settings ===
GAME_ENV = env_list[args.env]
GAME_SEED = args.seed
GAMMA = 0.99
# Only clip actions within the upper and lower bounds of env's action space, do not normalize actions.
NORMALIZE_ACTIONS = False
# Whether to clip rewards during Policy's postprocessing.
# None (default): Clip for Atari only (r=sign(r)).
# True: r=sign(r): Fixed rewards -1.0, 1.0, or 0.0.
# False: Never clip.
# [float value]: Clip at -value and + value.
# Tuple[value1, value2]: Clip at value1 and value2.
CLIP_REWAED = 15.0
# Whether to clip actions to the action space's low/high range spec.
# Default is true and clip according to the action boundary
CLIP_ACTIONS = True
# The default learning rate.
LR = 1e-3

# === PPO Settings ===
# kl_coeff: Additional loss term in ray implementation (ppo_tf_policy.py).  policy.kl_coeff * action_kl
KL_COEFF = 0
# If specified, clip the global norm of gradients by this amount.
GRAD_CLIP = 0.5
# PPO clip parameter.
CLIP_PARAM = 0.2 # [1-CLIP_PARAM, 1+CLIP_PARAM]
# clip param for the value function
VF_CLIP_PARAM = 0.2 # [-VF_CLIP_PARAM, VF_CLIP_PARAM]
# coefficient of the value function loss
VF_LOSS_COEF = 0.5
# The GAE (General advantage estimation) (lambda): self.gamma * self.lambda.
LAMBDA = 0.95

# === Evaluation Settings ===
# Number of evaluation game rounds.
EVAL_NUM_EPISODES = args.num_episodes
# Number of parallel evaluation workers.
EVAL_NUM_WOEKER = args.eval_num_workers


SAVE_DIR = '../agent-zoo/' + GAME_ENV.split('/')[1] + '_outer_party_id_' + str(PARTY_ORDER) \
           + '_party_0_loop_' + str(INNER_LOOP_PARTY_0) + '_party_1_loop_' + str(INNER_LOOP_PARTY_1) + '_' + str(LR)

EXP_NAME = str(GAME_SEED)
out_dir = setup_logger(SAVE_DIR, EXP_NAME)

if __name__ == '__main__':

    config = deepcopy(DEFAULT_CONFIG)

    # ======= Setting for rollout worker processes =======
    # Number of parallel workers/actors.
    config['num_workers'] = NUM_WORKERS
    # Number of environments per worker.
    config['num_envs_per_worker'] = NUM_ENV_WORKERS
    # Batch size collected from each worker (similar to n_steps).
    config['rollout_fragment_length'] = ROLLOUT_FRAGMENT_LENGTH
    # Number of gpus for the training worker.
    config['num_gpus'] = NUM_GPUS
    # Number of gpus for the remote worker.
    config['num_gpus_per_worker'] = NUM_GPUS_PER_WORKER

    # === Settings for the training process ===
    # Training batch size (similar to n_steps*nenv).
    config['train_batch_size'] = TRAIN_BATCH_SIZE
    # Minibatch size. Num_epoch = train_batch_size/sgd_minibatch_size.
    config['sgd_minibatch_size'] = TRAIN_MINIBATCH_SIZE
    config['lr'] = LR
    config['gamma'] = GAMMA
    # Number of epochs per iteration.
    config['num_sgd_iter'] = NEPOCH

    # === Environment Settings ===
    # Hyper-parameters that passed to the environment defined in env.py
    # Set debug as True for video playing.
    config['env_config'] = {'env_name': GAME_ENV, # Environment name.
                            'gamma': GAMMA, # Discount factor.
                            'num_agents_per_party': NUM_AGENTS_PER_PARTY, # number of agents in each party
                            'clip_rewards': CLIP_REWAED, # Reward clip boundary.
                            'epsilon': 1e-8, # Small value used for normalization.
                            'normalize': False, # Reward normalization.
                            'obs_norm_path': (AGT_0_OBS_NORM_PATH, AGT_1_OBS_NORM_PATH),
                            'reward_move': 0.1, # Dense reward fraction. (Reward move contains all the dense rewards.)
                            'reward_remaining': 0.01, # Sparse reward fraction.
                            'anneal_frac': 0, # Constant: (0: only use sparse reward. 1: only use dense reward).
                            'anneal_type': 0, # Anneal type: 0: Constant anneal, 1: Linear anneal (set anneal_frac as 1).
                            'total_step': TRAIN_BATCH_SIZE * NUPDATES, # total time steps.
                            'LOAD_PRETRAINED_MODEL': False, # True only if 'reward_move': 1 and 'reward_remaining': 1.
                            'debug': args.debug}

    # Add mean_std_filter of the observation. This normalization supports synchronization among workers.
    config['observation_filter'] = "MeanStdFilter"

    # Register the custom env "MuJoCo_Env"
    register_env('mujoco', lambda config: Minimax_Env(config['env_config']))
    env = Minimax_Env(config['env_config'])
    config['env'] = 'mujoco'

    # === PPO Settings ===
    # warning: kl_coeff
    config['kl_coeff'] = KL_COEFF
    # If specified, clip the global norm of gradients by this amount.
    config['grad_clip'] = GRAD_CLIP
    # PPO clip parameter.
    config['clip_param'] = CLIP_PARAM
    # clip param for the value function
    config['vf_clip_param'] = VF_CLIP_PARAM
    # coefficient of the value function loss
    config['vf_loss_coeff'] = VF_LOSS_COEF
    # The GAE (General advantage estimation) (lambda): self.gamma * self.lambda.
    config['lambda'] = LAMBDA

    # === Policy Settings === # ppo_ft_policy.py: define ppo loss functions.
    # Policy network settings.
    if USE_RNN:
        config['model']['fcnet_hiddens'] = [128]
        config['model']['lstm_cell_size'] = 128

        # LSTM rollout length. In our single worker implementation, it is set as the batch size.
        # In ray's implementation, the max_seq_len will dynamically change according to the actual trajectories length.
        # eg: trajectories collected from three envs: [x x x y y y y z z z], max_seq_len = 4,
        # because the actual max seq len in the input is 4 (y sequence)
        # config['model']['max_seq_len'] = 200

        # Register the custom model 'LSTM'.
        ModelCatalog.register_custom_model('custom_rnn', LSTM)
        config['model']['custom_model'] = 'custom_rnn'
    else:
        config['model']['fcnet_hiddens'] = [64, 64]

        # Register the custom model 'MLP'.
        ModelCatalog.register_custom_model('custom_mlp', MLP)
        config['model']['custom_model'] = 'custom_mlp'

    # Specify action distribution, Without this parameter, the distribution is set as Gaussian
    # based on the action space type (catalog.py: 213) catalog.py - get action distribution and policy model.
    # config['dist_type'] = 'DiagGaussian'

    # Define models (model, opp_model) and use the default PPO loss to train these models.

    policy_graphs = {}
    for i in range(NUM_AGENTS_PER_PARTY):
        policy_graphs['model_'+str(i)] = (PPOTFPolicy, env.observation_space, env.action_space, {})
        policy_graphs['opp_model_'+str(i)] = (PPOTFPolicy, env.observation_space, env.action_space, {})

    # Multi-agent settings.
    config.update({
        'multiagent': {
            'policies': policy_graphs,
            'policy_mapping_fn': policy_mapping_fn,
        },
    })

    # === Evaluation Settings ===
    config['evaluation_num_episodes'] = EVAL_NUM_EPISODES
    config['evaluation_num_workers'] = EVAL_NUM_WOEKER
    config['evaluation_config'] = {'out_dir': out_dir}

    # Initialize the ray.
    ray.init(local_mode=True)
    trainer = PPOTrainer(env=Minimax_Env, config=config)
    # This instruction will build a trainer class and setup the trainer. The setup process will make workers, which will
    # call DynamicTFPolicy in dynamic_tf_policy.py. DynamicTFPolicy will define the action distribution based on the
    # action space type (line 151) and build the model.

    if LOAD_PRETRAINED_MODEL:
        init_model, init_opp_model = load_pretrain_model(AGT_0_MODEL_PATH, AGT_1_MODEL_PATH)
        init_filter_0 = create_mean_std(AGT_0_OBS_NORM_PATH)
        init_filter_1 = create_mean_std(AGT_1_OBS_NORM_PATH)

        if not USE_RNN:
            init_model['model/logstd'] = init_model['model/logstd'].flatten()
            init_opp_model['opp_model/logstd'] = init_opp_model['opp_model/logstd'].flatten()

        for i in range(NUM_AGENTS_PER_PARTY):
            init_model_tmp = {k.replace('model', 'model_'+str(i)): v for k, v in init_model.items()}
            init_opp_model_tmp = {k.replace('opp_model', 'opp_model_'+str(i)):v for k, v in init_opp_model.items()}
            trainer.workers.foreach_worker(lambda ev: ev.get_policy('model_'+str(i)).set_weights(init_model_tmp))
            trainer.workers.foreach_worker(lambda ev: ev.get_policy('opp_model_'+str(i)).set_weights(init_opp_model_tmp))
            trainer.workers.foreach_worker(lambda ev: ev.filters['model_'+str(i)].sync(init_filter_0))
            trainer.workers.foreach_worker(lambda ev: ev.filters['opp_model_'+str(i)].sync(init_filter_1))

        # print(trainer.get_weights()['model_0']['model_0/logstd'])
        # print(trainer.get_weights()['model_1']['model_1/logstd'])
        # print(trainer.get_weights()['opp_model_0']['opp_model_0/logstd'])
        # print(trainer.get_weights()['opp_model_1']['opp_model_1/logstd'])
        # print(trainer.workers.foreach_worker(lambda ev: ev.get_policy('model_0').get_weights())[0]['model_0/logstd'])
        # print(trainer.workers.foreach_worker(lambda ev: ev.get_policy('model_0').get_weights())[1]['model_0/logstd'])
        # print(trainer.workers.foreach_worker(lambda ev: ev.get_policy('opp_model_1').get_weights())[0]['opp_model_1/logstd'])
        # print(trainer.workers.foreach_worker(lambda ev: ev.get_policy('opp_model_1').get_weights())[1]['opp_model_1/logstd'])
        # print(trainer.workers.foreach_worker(lambda ev: ev.filters)[0])
        # print(trainer.workers.foreach_worker(lambda ev: ev.filters)[1])

    minimax_learning(trainer=trainer, num_workers=NUM_WORKERS, num_agent_per_party=NUM_AGENTS_PER_PARTY,
                     inner_loop_party_0=INNER_LOOP_PARTY_0, inner_loop_party_1=INNER_LOOP_PARTY_1,
                     select_num_episodes=SELECT_NUM_EPISODES,  select_num_worker=SELECT_NUM_WOEKER,
                     nupdates=NUPDATES, out_dir=out_dir)

