import os
os.environ["CUDA_VISIBLE_DEVICES"] = ' '
import ray
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
parser.add_argument("--num_workers", type=int, default=2)

# Number of environments per worker
parser.add_argument("--num_envs_per_worker", type=int, default=2)

# Number of parallel evaluation workers.
parser.add_argument("--eval_num_workers", type=int, default=1)

# Number of evaluation game rounds.
parser.add_argument("--num_episodes", type=int, default=2)

# Number of gpus for the training worker.
parser.add_argument("--num_gpus", type=int, default=0)

# Number of gpus for the remote worker.
parser.add_argument("--num_gpus_per_worker", type=int, default=0)

# ["multicomp/YouShallNotPassHumans-v0", "multicomp/KickAndDefend-v0",
#  "multicomp/SumoAnts-v0", "multicomp/SumoHumans-v0"]
parser.add_argument("--env", type=int, default=0)

# (Initial) victim party id.
# YouShallNotPass: blocker -> agent_0, runner -> agent_1.
# KickAndDefend: kicker -> agent_0, keeper -> agent_1.
parser.add_argument("--victim_party_id", type=int, default=0)

# (Initial) victim model path.
parser.add_argument("--victim_model_path", type=str, default="../adv_agent/you")

# Whether to apply iteratively adversarial training.
parser.add_argument("--iterative", type=bool, default=False)

# Number of iterative
parser.add_argument("--outer_loop", type=int, default=20)

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
ROLLOUT_FRAGMENT_LENGTH = 100

# === Settings for the Trainer process ===
# Number of epochs in each iteration.
NEPOCH = 4
# Training batch size.
TRAIN_BATCH_SIZE = ROLLOUT_FRAGMENT_LENGTH*NUM_WORKERS*NUM_ENV_WORKERS
# Minibatch size. Num_epoch = train_batch_size/sgd_minibatch_size.
TRAIN_MINIBATCH_SIZE = TRAIN_BATCH_SIZE/4
# Loading a pretrained model as the initial model or not.
LOAD_PRETRAINED_MODEL = args.load_pretrained_model
# Number of iterations.
NUPDATES = int(20000000/TRAIN_BATCH_SIZE)

# === Settings for the (iterative) adversarial training process ===
# Whether to use RNN as policy network.
if args.env == 0:
    USE_RNN = False
else:
    USE_RNN = True

# Whether to apply iteratively adversarial training.
ITERATIVE = args.iterative
# Number of outer iterative
OUTER_LOOP = args.outer_loop
# Whether to load pretrained model for each party [party_0, party_1].
LOAD_PRETRAINED_MODEL = [True, True]
# (Initial) victim party id.
VICTIM_PARTY_ID = args.victim_party_id
# Initial victim model path.
VICTIM_MODEL_PATH = args.victim_model_path

# === Environment Settings ===
GAME_ENV = env_list[args.env]
GAME_SEED = args.seed
GAMMA = 0.99
# Clip actions to the upper and lower bounds of env's action space.
NORMALIZE_ACTIONS = True
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
    SAVE_DIR = '../adv-agent-zoo/' + GAME_ENV.split('/')[1] + '_' + LOAD_PRETRAINED_MODEL + '_' + str(LR)
else:
    SAVE_DIR = '../iterative-adv-training/' + GAME_ENV.split('/')[1] + '_victim_id_' + VICTIM_PARTY_ID + '_' \
               + LOAD_PRETRAINED_MODEL + '_' + str(LR)

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

    # === Settings for the Trainer process ===
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
        iterative_adv_training(config, NUPDATES, OUTER_LOOP, USE_RNN, VICTIM_PARTY_ID, LOAD_PRETRAINED_MODEL, out_dir)
    else:
        adv_attacking(config, NUPDATES, out_dir)
