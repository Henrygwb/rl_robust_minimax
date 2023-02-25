import os
import sys
os.environ["CUDA_VISIBLE_DEVICES"]=' '
import ray
import argparse
import random
from absl import app, flags, logging

from copy import deepcopy
from os.path import expanduser
from env import Starcraft_Env
from ray.rllib.agents.ppo.ppo_tf_policy import PPOTFPolicy
from ray.rllib.agents.ppo.ppo import PPOTrainer, DEFAULT_CONFIG
from ray.rllib.models import ModelCatalog
from ray.tune.registry import register_env
from zoo_utils import LSTM, MLP, load_pretrain_model, setup_logger
from ppo_selfplay import custom_symmtric_eval_function, custom_assymmtric_eval_function, \
    symmtric_learning, assymmtric_learning, policy_mapping_fn


##################
# Hyper-parameters
##################

parser = argparse.ArgumentParser()
# Number of parallel workers/actors.
parser.add_argument("--num_workers", type=int, default=20)

# Number of environments per worker
parser.add_argument("--num_envs_per_worker", type=int, default=1)

# Number of parallel evaluation workers.
parser.add_argument("--eval_num_workers", type=int, default=10)

# Number of evaluation game rounds.
parser.add_argument("--num_episodes", type=int, default=50)

parser.add_argument("--eval_select_ratio", type=int, default=2)

# Number of gpus for the training worker.
parser.add_argument("--num_gpus", type=int, default=0)

# Number of gpus for the remote worker.
parser.add_argument("--num_gpus_per_worker", type=int, default=0)

# Random seed.
parser.add_argument("--seed", type=int, default=0)

# The model used as the opponent. latest, random.
parser.add_argument("--opp_model", type=str, default='latest')

# Loading a pretrained model as the initial model or not.
parser.add_argument("--load_pretrained_model", type=bool, default=True)


# # Pretrained normalization and model params path. 
parser.add_argument("--agent_0_pretrain_model_path", type=str,
                    default="../initial-agents/starcraft")
parser.add_argument("--agent_1_pretrain_model_path", type=str,
                    default="../initial-agents/starcraft")

# # Options for Starcraft
parser.add_argument("--game_version", type=str, default='4.6')

parser.add_argument("--game_steps_per_episode", type=int, default=43200)

parser.add_argument("--step_mul", type=int, default=32)

parser.add_argument("--disable_fog", type=bool, default=True)

parser.add_argument("--use_all_combat_actions", type=bool, default=False)

parser.add_argument("--use_region_features", type=bool, default=False)

parser.add_argument("--use_action_mask", type=bool, default=True)

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
ROLLOUT_FRAGMENT_LENGTH = 512

SELECT_NUM_EPISODES = int(args.num_episodes / args.eval_select_ratio)

# === Settings for the training process ===
# Number of epochs in each iteration.
NEPOCH = 4
# Training batch size.
TRAIN_BATCH_SIZE = ROLLOUT_FRAGMENT_LENGTH*NUM_WORKERS*NUM_ENV_WORKERS
# Minibatch size. Num_epoch = train_batch_size/sgd_minibatch_size.
TRAIN_MINIBATCH_SIZE = TRAIN_BATCH_SIZE/NEPOCH
# Loading a pretrained model as the initial model or not.
LOAD_PRETRAINED_MODEL = args.load_pretrained_model
# Number of iterations.
NUPDATES = int(10000000/TRAIN_BATCH_SIZE)

AGT_0_MODEL_PATH = args.agent_0_pretrain_model_path
AGT_1_MODEL_PATH = args.agent_1_pretrain_model_path

if args.opp_model == 'latest':
    OPP_MODEL = 0
elif args.opp_model == 'random':
    OPP_MODEL = 1
else:
    OPP_MODEL = 2

SYMM_TRAIN = True
USE_RNN = False

print('====================================')
print(USE_RNN)
print(SYMM_TRAIN)
print(AGT_0_MODEL_PATH)
print(AGT_1_MODEL_PATH)
print('====================================')

# === Environment Settings ===
GAME_ENV = 'StarCraft2'
GAME_SEED = random.randint(0, 2 ** 32 - 1)
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
LR = 0.0

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

EVAL_NUM_EPISODES = args.num_episodes
EVAL_NUM_WOEKER = args.eval_num_workers


SAVE_DIR = '../agent-zoo/' + GAME_ENV + '_' + args.opp_model + '_' + str(LR)
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
    config['env_config'] = {'env_name': GAME_ENV, # Environment name.
                            'gamma': GAMMA, # Discount factor.
                            'clip_rewards': CLIP_REWAED, # Reward clip boundary.
                            'epsilon': 1e-8, # Small value used for normalization.
                            'total_step': TRAIN_BATCH_SIZE * NUPDATES, # total time steps.
                            'LOAD_PRETRAINED_MODEL': False, # True only if 'reward_move': 1 and 'reward_remaining': 1.
                            'debug': args.debug, 
                            # env setting
                            'game_version': args.game_version,
                            'game_seed': GAME_SEED,
                            'game_steps_per_episode': args.game_steps_per_episode,
                            'step_mul': args.step_mul,
                            'disable_fog': args.disable_fog,
                            'use_all_combat_actions': args.use_all_combat_actions,
                            'use_region_features': args.use_region_features,
                            'use_action_mask': args.use_action_mask}
    
    config['observation_filter'] = 'NoFilter'
    
    # Register the custom env "Starcraft_Env"
    register_env('starcraft', lambda config: Starcraft_Env(config['env_config']))
    env = Starcraft_Env(config['env_config'])
    config['env'] = 'starcraft'

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
   
    config['model']['fcnet_hiddens'] = [128, 128, 128]

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
    config['evaluation_interval'] = 1
    config['evaluation_num_episodes'] = EVAL_NUM_EPISODES
    config['evaluation_num_workers'] = EVAL_NUM_WOEKER
    config['evaluation_config'] = {
        'out_dir': out_dir,
    }

    # Initialize the ray.
    ray.init()
    trainer = PPOTrainer(env=Starcraft_Env, config=config)
    # This instruction will build a trainer class and setup the trainer. The setup process will make workers, which will
    # call DynamicTFPolicy in dynamic_tf_policy.py. DynamicTFPolicy will define the action distribution based on the
    # action space type (line 151) and build the model.

    if LOAD_PRETRAINED_MODEL:
        init_model, init_opp_model = load_pretrain_model(AGT_0_MODEL_PATH, AGT_1_MODEL_PATH)

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

    # pickle.dump(trainer.config, open(out_dir+'/config.pkl', 'wb'))

    if SYMM_TRAIN:
       symmtric_learning(trainer=trainer, num_workers=NUM_WORKERS, nupdates=NUPDATES, 
                         select_num_episodes=SELECT_NUM_EPISODES, opp_method=OPP_MODEL, out_dir=out_dir)

    # Move log in ray_results to the current output folder.
    folder_time = out_dir.split('/')[-1]
    folder_time = folder_time[0:4] + '-' + folder_time[4:6] + '-' + folder_time[6:8] + '_' + \
                  folder_time[9:11] + '-' + folder_time[11:13]
    default_log_folder = expanduser("~") + '/ray_results'
    log_folders = os.listdir(default_log_folder)
    target_log_folder = [f for f in log_folders if folder_time in f]
    if len(target_log_folder) == 0:
        folder_time = folder_time[:-1] + str(int(folder_time[-1])+1)
        target_log_folder = [f for f in log_folders if folder_time in f]
    for folder in target_log_folder:
        os.system('cp -r '+os.path.join(default_log_folder, folder)+' '+out_dir+'/'+folder)
        os.system('rm -r '+os.path.join(default_log_folder, folder))
