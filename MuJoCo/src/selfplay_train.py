import os
import ray
import argparse
from copy import deepcopy
from env import MuJoCo_Env, env_list
from ray.rllib.agents.ppo.ppo_tf_policy import PPOTFPolicy
from ray.rllib.agents.ppo.ppo import PPOTrainer, DEFAULT_CONFIG
from ray.rllib.models import ModelCatalog
from ray.tune.registry import register_env
from zoo_utils import LSTM, MLP, load_pretrain_model, setup_logger, create_mean_std
from ppo_selfplay import custom_symmtric_eval_function, custom_assymmtric_eval_function, \
    symmtric_learning, assymmtric_learning, policy_mapping_fn


##################
# Hyper-parameters
##################

parser = argparse.ArgumentParser()
# Number of parallel workers/actors.
parser.add_argument("--num_workers", type=int, default=1)

# Number of environments per worker
parser.add_argument("--num_envs_per_worker", type=int, default=1)

# ["multicomp/YouShallNotPassHumans-v0", "multicomp/KickAndDefend-v0",
#  "multicomp/SumoAnts-v0", "multicomp/SumoHumans-v0"]
parser.add_argument("--env", type=int, default=1)

# Random seed.
parser.add_argument("--seed", type=int, default=0)

# The model used as the opponent. latest, random.
parser.add_argument("--opp_model", type=str, default='random')

# Loading a pretrained model as the initial model or not.
parser.add_argument("--load_pretrained_model", type=bool, default=True)

# # Pretrained normalization and model params path .
# YouShallNotPass: blocker (saved agent_1) -> agent_0, runner (saved agent_2) -> agent_1
# parser.add_argument("--agent_0_obs_norm_path", type=str,
#                     default="../initial-agents/YouShallNotPassHumans-v0/agent1-rms-v1.pkl")
#
# parser.add_argument("--agent_0_pretrain_model_path", type=str,
#                     default="../initial-agents/YouShallNotPassHumans-v0/agent1-model-v1.pkl")
#
# parser.add_argument("--agent_1_obs_norm_path", type=str,
#                     default="../initial-agents/YouShallNotPassHumans-v0/agent2-rms-v1.pkl")
#
# parser.add_argument("--agent_1_pretrain_model_path", type=str,
#                     default="../initial-agents/YouShallNotPassHumans-v0/agent2-model-v1.pkl")

# KickAndDefend: kicker (saved agent_1) -> agent_0, keeper (saved agent_2) -> agent_1
# Pretrained normalization and model params path.

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

# SumoAnts.
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


parser.add_argument('--debug', type=bool, default=True)

args = parser.parse_args()

# ======= Setting for rollout worker processes =======
# Number of parallel workers/actors.
NUM_WORKERS = args.num_workers
# Number of environments per worker.
NUM_ENV_WORKERS = args.num_envs_per_worker
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
NUPDATES = int(200000000/TRAIN_BATCH_SIZE)

AGT_0_OBS_NORM_PATH = args.agent_0_obs_norm_path
AGT_0_MODEL_PATH = args.agent_0_pretrain_model_path

AGT_1_OBS_NORM_PATH = args.agent_1_obs_norm_path
AGT_1_MODEL_PATH = args.agent_1_pretrain_model_path

if args.opp_model == 'latest':
    OPP_MODEL = 0
elif args.opp_model == 'random':
    OPP_MODEL = 1
else:
    print('unknown option of which model to be used as the opponent model, default as the latest model.')
    OPP_MODEL = 0

# SYMM_TRAIN or not.
if args.env == 0 or args.env == 1:
    SYMM_TRAIN = False
else:
    SYMM_TRAIN = True

# Whether to use RNN as policy network.
if args.env == 0:
    USE_RNN = False
else:
    USE_RNN = True

print('====================================')
print(USE_RNN)
print(SYMM_TRAIN)
print(AGT_0_MODEL_PATH)
print(AGT_1_MODEL_PATH)
print(AGT_0_OBS_NORM_PATH)
print(AGT_1_OBS_NORM_PATH)
print('====================================')

# === Environment Settings ===
GAME_ENV = env_list[args.env]
GAME_SEED = args.seed
GAMMA = 0.99
# Unsquash actions to the upper and lower bounds of env's action space
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
LR = 1e-7

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

EVAL_NUM_EPISODES = 2
EVAL_NUM_WOEKER = 25


SAVE_DIR = '../agent-zoo/' + GAME_ENV.split('/')[1] + '_' + args.opp_model
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
    config['env_config'] = {'env_name': GAME_ENV, # Environment name.
                            'gamma': GAMMA, # Discount factor.
                            'clip_rewards': CLIP_REWAED, # Reward clip boundary.
                            'epsilon': 1e-8, # Small value used for normalization.
                            'normalize': True, # Reward normalization.
                            'obs_norm_path': (AGT_0_OBS_NORM_PATH, AGT_1_OBS_NORM_PATH),
                            'reward_move': 0.1, # Dense reward fraction.
                            'reward_remaining': 0.01, # Sparse reward fraction.
                            'anneal_frac': 0, # Constant: (0: only use sparse reward. 1: only use dense reward).
                            'anneal_type': 0, # Anneal type: 0: Constant anneal, 1: Linear anneal (set anneal_frac as 1).
                            'total_step': TRAIN_BATCH_SIZE * NUPDATES, # total time steps.
                            'LOAD_PRETRAINED_MODEL': False, # True only if 'reward_move': 1 and 'reward_remaining': 1.
                            'debug': args.debug}

    # Add mean_std_filter of the observation. This normalization supports synchronization among workers.
    config['observation_filter'] = "MeanStdFilter"

    # Register the custom env "MuJoCo_Env"
    register_env('mujoco', lambda config: MuJoCo_Env(config['env_config']))
    env = MuJoCo_Env(config['env_config'])
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

    # Define two models (model, opp_model) and use the default PPO loss to train these models.
    policy_graphs = {'model': (PPOTFPolicy, env.observation_space, env.action_space, {}),
                     'opp_model': (PPOTFPolicy, env.observation_space, env.action_space, {})}

    # Multi-agent settings.
    config.update({
        'multiagent': {
            'policies': policy_graphs,
            'policy_mapping_fn': policy_mapping_fn,
        },
    })

    # === Evaluation Settings ===
    # Test 50 episodes, use 10 eval workers to do parallel test.
    if SYMM_TRAIN:
        config['custom_eval_function'] = custom_symmtric_eval_function
    else:
        config['custom_eval_function'] = custom_assymmtric_eval_function
    config['evaluation_interval'] = 1
    config['evaluation_num_episodes'] = EVAL_NUM_EPISODES
    config['evaluation_num_workers'] = EVAL_NUM_WOEKER
    config['evaluation_config'] = {
        'out_dir': out_dir,
    }

    # Initialize the ray.
    ray.init(local_mode=True)
    trainer = PPOTrainer(env=MuJoCo_Env, config=config)
    # This instruction will build a trainer class and setup the trainer. The setup process will make workers, which will
    # call DynamicTFPolicy in dynamic_tf_policy.py. DynamicTFPolicy will define the action distribution based on the
    # action space type (line 151) and build the model.

    if LOAD_PRETRAINED_MODEL:
        init_model, init_opp_model = load_pretrain_model(AGT_0_MODEL_PATH, AGT_1_MODEL_PATH)

        # Load the pretrained model as the initial model.
        if not USE_RNN:
            init_model['model/logstd'] = init_model['model/logstd'].flatten()
            init_opp_model['opp_model/logstd'] = init_opp_model['opp_model/logstd'].flatten()

        trainer.workers.foreach_worker(lambda ev: ev.get_policy('model').set_weights(init_model))
        trainer.workers.foreach_worker(lambda ev: ev.get_policy('opp_model').set_weights(init_opp_model))

        # Check model loading for the local worker.
        # para = trainer.get_policy('model').get_weights()
        # for i in para.keys():
        #     print(np.count_nonzero(init_model[i] - para[i]))
        # para = trainer.workers.local_worker().get_weights()['opp_model']
        # for i in para.keys():
        #     print(np.count_nonzero(init_opp_model[i] - para[i]))

        # Check model loading for the remote worker.
        # ww = trainer.workers.foreach_worker(lambda ev: ev.get_policy('model').get_weights())
        # ww_opp = trainer.workers.foreach_worker(lambda ev: ev.get_policy('opp_model').get_weights())
        # Check the length of ww/ww_opp. The first one is local worker, the others are remote works.

        # Load mean/std of the pretrained model and copy them to the current rms filter.
        # Load mean/std of the pretrained model and copy them to the current rms filter.
        init_filter_0 = create_mean_std(AGT_0_OBS_NORM_PATH)
        init_filter_1 = create_mean_std(AGT_1_OBS_NORM_PATH)

        trainer.workers.foreach_worker(lambda ev: ev.filters['model'].sync(init_filter_0))
        trainer.workers.foreach_worker(lambda ev: ev.filters['opp_model'].sync(init_filter_1))

        # Check obs_rms loading.
        # trainer.workers.local_worker().get_filters()['opp_model']
        # trainer.workers.local_worker().get_filters()['model']
        # All fitlers: trainer.workers.foreach_worker(lambda ev: ev.get_filters())

    # pickle.dump(trainer.config, open(out_dir+'/config.pkl', 'wb'))

    if SYMM_TRAIN:
        symmtric_learning(trainer=trainer, num_workers=NUM_WORKERS, nupdates=NUPDATES,
                          opp_method=OPP_MODEL, out_dir=out_dir)
    else:
        assymmtric_learning(trainer=trainer, num_workers=NUM_WORKERS, nupdates=NUPDATES,
                            opp_method=OPP_MODEL, out_dir=out_dir)

    folder_time = out_dir.split('/')[-1]
    folder_time = folder_time[0:4] + '-' + folder_time[4:6] + '-' + folder_time[6:8] + '_' + \
                  folder_time[9:11] + '-' + folder_time[11:13]
    default_log_folder = '/Users/Henryguo/ray_results'
    log_folders = os.listdir(default_log_folder)
    target_log_folder = [f for f in log_folders if folder_time in f]
    if len(target_log_folder) == 0:
        folder_time = folder_time[:-1] + str(int(folder_time[-1])+1)
        target_log_folder = [f for f in log_folders if folder_time in f]
    for folder in target_log_folder:
        os.system('cp -r '+os.path.join(default_log_folder, folder)+' '+out_dir+'/'+folder)
        os.system('rm -r '+os.path.join(default_log_folder, folder))
