import os
os.environ["CUDA_VISIBLE_DEVICES"] = ' '
import argparse
from copy import deepcopy
from env import Adv_Env, env_list
from ray.rllib.agents.ppo.ppo import DEFAULT_CONFIG
from ray.rllib.models import ModelCatalog
from ray.tune.registry import register_env
from zoo_utils import LSTM, MLP, setup_logger
from ppo_adv import custom_eval_function, adv_attacking, iterative_adv_training


##################
# Hyper-parameters
##################
parser = argparse.ArgumentParser()
# Number of parallel workers/actors.
parser.add_argument("--num_workers", type=int, default=70)

# Number of environments per worker
parser.add_argument("--num_envs_per_worker", type=int, default=8)

# Number of parallel evaluation workers.
parser.add_argument("--eval_num_workers", type=int, default=10)

# Number of evaluation game rounds.
parser.add_argument("--num_episodes", type=int, default=50)

# Number of gpus for the training worker.
parser.add_argument("--num_gpus", type=int, default=0)

# Number of gpus for the remote worker.
parser.add_argument("--num_gpus_per_worker", type=int, default=0)

# ["multicomp/YouShallNotPassHumans-v0", "multicomp/KickAndDefend-v0",
#  "multicomp/SumoAnts-v0", "multicomp/SumoHumans-v0"]
parser.add_argument("--env", type=int, default=0)

# (Initial) victim party id.
# YouShallNotPass: blocker -> agent_0, runner -> agent_1. 1
# KickAndDefend: kicker -> agent_0, keeper -> agent_1. 0
# SumoGames: 0
parser.add_argument("--victim_party_id", type=int, default=1)

# (Initial) selfplay victim model path.
# You Shall Not Pass
# parser.add_argument("--victim_model_path", type=str, default="../victim-agents/minimax/YouShallNotPassHumans-v0/party_1/lr_1e-4_0.32_0.68/")
# parser.add_argument("--victim_model_path", type=str, default="../victim-agents/selfplay/YouShallNotPassHumans-v0/party_1/lr_5e-3_0.08_0.92/")

# SumoHumans
# parser.add_argument("--victim_model_path", type=str, default="../victim-agents/selfplay/SumoHumans-v0/lr_1e-4_0.3_0.3_0.4/")
# parser.add_argument("--victim_model_path", type=str, default="../victim-agents/selfplay/SumoHumans-v0/lr_5e-3_0.17_0.16_0.67/")
# parser.add_argument("--victim_model_path", type=str, default="../victim-agents/selfplay/SumoHumans-v0/lr_5e-3_0.4_0.38_0.22/")

# SumoAnts
#parser.add_argument("--victim_model_path", type=str, default="../victim-agents/selfplay/SumoAnts-v0/lr_1e-4_0.44_0.44_0.12/")
#parser.add_argument("--victim_model_path", type=str, default="../victim-agents/selfplay/SumoAnts-v0/lr_5e-3_0.08_0.08_0.84/")

#########
# (Initial) minimax victim model path.
# You Shall Not Pass
# parser.add_argument("--victim_model_path", type=str, default="../victim-agents/minimax/YouShallNotPassHumans-v0/party_1/lr_1e-4_0.25_0.87/")

# SumoHumans
# parser.add_argument("--victim_model_path", type=str, default="../victim-agents/minimax/SumoHumans-v0/party_0/lr_1e-4_0.35_0.54/")

# SumoAnts
# parser.add_argument("--victim_model_path", type=str, default="../victim-agents/minimax/SumoAnts-v0/party_0/lr_1e-4_0.45_0.59/")

# Whether to load a pretrained adversarial model in the first iteration (attack).
parser.add_argument("--load_pretrained_model_first", type=bool, default=False)

# (Initial) pretrained adversarial model path.
# YouShallNotPass: blocker (saved agent_1) -> agent_0, runner (saved agent_2) -> agent_1
# parser.add_argument("--pretrained_obs_path", type=str,
#                     default="../initial-agents/YouShallNotPassHumans-v0/agent1-rms-v1.pkl")
#
# parser.add_argument("--pretrained_model_path", type=str,
#                     default="../initial-agents/YouShallNotPassHumans-v0/agent1-model-v1.pkl")

# KickAndDefend: kicker (saved agent_1) -> agent_0, keeper (saved agent_2) -> agent_1
# parser.add_argument("--pretrained_obs_path", type=str,
#                     default="../initial-agents/KickAndDefend-v0/agent2-rms-v3.pkl")
#
# parser.add_argument("--pretrained_model_path", type=str,
#                     default="../initial-agents/KickAndDefend-v0/agent2-model-v3.pkl")

# SumoAnts.
# parser.add_argument("--pretrained_obs_path", type=str,
#                     default="../initial-agents/SumoAnts-v0/agent0-rms-v1.pkl")
#
# parser.add_argument("--pretrained_model_path", type=str,
#                     default="../initial-agents/SumoAnts-v0/agent0-model-v1.pkl")

# SumoHumans.
# parser.add_argument("--pretrained_obs_path", type=str,
#                     default="../initial-agents/SumoHumans-v0/agent0-rms-v3.pkl")
#
# parser.add_argument("--pretrained_model_path", type=str,
#                     default="../initial-agents/SumoHumans-v0/agent0-model-v3.pkl")

# Whether to apply iteratively adversarial training.
parser.add_argument("--iterative", type=bool, default=True)

# Number of iterative
parser.add_argument("--outer_loop", type=int, default=10)

# Whether to load a pretrained model for each party [party_0, party_1] except the first iteration.
# You Shall Not Pass: [False, True].
# Blocker (party 0) is harder to be attacked. Runner (party 1) need to load a pretrained model.
LOAD_PRETRAINED_MODEL = [False, True]

# Always load the initial victim model path.
LOAD_INITIAL = [False, True]

# LR.
parser.add_argument('--lr', type=float, default=3e-4)

# Debug or not.
parser.add_argument('--debug', type=bool, default=False)

# Random seed.
parser.add_argument("--seed", type=int, default=0)

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
ROLLOUT_FRAGMENT_LENGTH = 200

# === Settings for the training process ===
# Number of epochs in each iteration.
NEPOCH = 4
# Training batch size.
TRAIN_BATCH_SIZE = ROLLOUT_FRAGMENT_LENGTH*NUM_WORKERS*NUM_ENV_WORKERS
# Minibatch size. Num_epoch = train_batch_size/sgd_minibatch_size.
TRAIN_MINIBATCH_SIZE = TRAIN_BATCH_SIZE/NEPOCH
# Number of iterations.
NUPDATES = int(30000000/TRAIN_BATCH_SIZE)

# === Settings for the (iterative) adversarial training process ===
# Whether to use RNN as policy network.
if args.env == 0:
    USE_RNN = False
else:
    USE_RNN = True

# (Initial) victim party id.
VICTIM_PARTY_ID = args.victim_party_id
# Initial victim model path.
VICTIM_MODEL_PATH = args.victim_model_path

# Whether to load a pretrained adversarial model in the first iteration (attack).
LOAD_PRETRAINED_MODEL_FIRST = args.load_pretrained_model_first

# (Initial) pretrained adversarial model path.
PRETRAINED_MODEL_PATH = args.pretrained_model_path
PRETRAINED_OBS_PATH = args.pretrained_obs_path
# Whether to apply iteratively adversarial training.
ITERATIVE = args.iterative
# Number of outer iterative
OUTER_LOOP = args.outer_loop

# Whether to load a pretrained model for each party [party_0, party_1] except the first iteration.
# You Shall Not Pass: [False, True].
# Blocker (party 0) is harder to be attacked. Runner (party 1) need to load a pretrained model.
LOAD_PRETRAINED_MODEL = LOAD_PRETRAINED_MODEL

# Always load the initial model [True, True].
LOAD_INITIAL = LOAD_INITIAL

print('====================================')
print(env_list[args.env])
print('Use RNN:')
print(USE_RNN)
print('VICTIM_PARTY_ID:')
print(VICTIM_PARTY_ID)
print('VICTIM_MODEL_PATH:')
print(VICTIM_MODEL_PATH)
print('LOAD_PRETRAINED_MODEL_FIRST:')
print(LOAD_PRETRAINED_MODEL_FIRST)
print('PRETRAINED_MODEL_PATH:')
print(PRETRAINED_MODEL_PATH)
print('ITERATIVE:')
print(ITERATIVE)
print('LOAD_PRETRAINED_MODEL:')
print(LOAD_PRETRAINED_MODEL)
print('LOAD_INITIAL:')
print(LOAD_INITIAL)
print('====================================')


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
# Default is true and clip according to the action boundary.
CLIP_ACTIONS = True
# The default learning rate.
LR = args.lr

# === PPO Settings ===
# kl_coeff: Additional loss term in ray implementation (ppo_tf_policy.py).  policy.kl_coeff * action_kl.
KL_COEFF = 0
# If specified, clip the global norm of gradients by this amount.
GRAD_CLIP = 0.5
# PPO clip parameter.
CLIP_PARAM = 0.2  # [1-CLIP_PARAM, 1+CLIP_PARAM]
# clip param for the value function
VF_CLIP_PARAM = 0.2  # [-VF_CLIP_PARAM, VF_CLIP_PARAM]
# coefficient of the value function loss
VF_LOSS_COEF = 0.2
# The GAE (General advantage estimation) (lambda): self.gamma * self.lambda.
LAMBDA = 0.95

# === Evaluation Settings ===

EVAL_NUM_EPISODES = args.num_episodes
EVAL_NUM_WOEKER = args.eval_num_workers


if ITERATIVE:
    SAVE_DIR = '../iterative-adv-training/' + GAME_ENV.split('/')[1] + '_initial_victim_id_' + str(VICTIM_PARTY_ID)\
               + '_load_pretrain_' + str(LOAD_PRETRAINED_MODEL[0]) + '_' + str(LOAD_PRETRAINED_MODEL[0])\
               + '_always_load_initial_' + str(LOAD_INITIAL[0]) + '_' + str(LOAD_INITIAL[1]) \
               + '_load_pretrain_first_' + str(LOAD_PRETRAINED_MODEL_FIRST) + '_' +str(LR)
else:
    SAVE_DIR = '../adv-agent-zoo/' + GAME_ENV.split('/')[1] + '_victim_id_' + str(VICTIM_PARTY_ID) \
               + '_load_pretrain_first_' + str(LOAD_PRETRAINED_MODEL_FIRST) + '_' +str(LR)

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
    config['env_config'] = {'env_name': GAME_ENV,  # Environment name.
                            'gamma': GAMMA,  # Discount factor.
                            'clip_rewards': CLIP_REWAED,  # Reward clip boundary.
                            'epsilon': 1e-8,  # Small value used for normalization.
                            'normalization': True,  # Reward normalization.
                            'victim_party_id': VICTIM_PARTY_ID, # Initial victim party id.
                            'victim_model_path': VICTIM_MODEL_PATH, # Initial victim model path.
                            'reward_move': 0.1,  # Dense reward fraction.
                            'reward_remaining': 0.01,  # Sparse reward fraction.
                            'anneal_frac': 0,  # Constant: (0: only use sparse reward. 1: only use dense reward).
                            'anneal_type': 0, # Anneal type: 0: Constant anneal, 1: Linear anneal (set anneal_frac as 1).
                            'total_step': TRAIN_BATCH_SIZE * NUPDATES,  # total time steps.
                            'debug': args.debug}

    # Add mean_std_filter of the observation. This normalization supports synchronization among workers.
    config['observation_filter'] = "MeanStdFilter"

    # Register the custom env "MuJoCo_Env"
    register_env('MuJoCo_adv_env', lambda config: Adv_Env(config['env_config']))
    config['env'] = 'MuJoCo_adv_env'

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

    # === Evaluation Settings ===
    # Test 50 episodes, use 10 eval workers to do parallel test.
    config['custom_eval_function'] = custom_eval_function
    config['evaluation_interval'] = 1
    config['evaluation_num_episodes'] = EVAL_NUM_EPISODES
    config['evaluation_num_workers'] = EVAL_NUM_WOEKER

    if ITERATIVE:
        iterative_adv_training(config, NUPDATES, OUTER_LOOP, VICTIM_PARTY_ID, USE_RNN, LOAD_PRETRAINED_MODEL,
                               LOAD_INITIAL, LOAD_PRETRAINED_MODEL_FIRST, PRETRAINED_MODEL_PATH, PRETRAINED_OBS_PATH,
                               out_dir)
    else:
        adv_attacking(config, NUPDATES, LOAD_PRETRAINED_MODEL_FIRST, PRETRAINED_MODEL_PATH, out_dir)
