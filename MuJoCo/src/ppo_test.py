from copy import deepcopy
import ray
from ray.rllib.agents.ppo.ppo_tf_policy import PPOTFPolicy
from ray.rllib.agents.ppo.ppo import PPOTrainer, DEFAULT_CONFIG
from ray.rllib.models import ModelCatalog
from ray.tune.logger import pretty_print
from ray.tune.registry import register_env
from env import MuJoCo_Env
from zoo_utils import LSTM, load_model, load_pretrain_model, setup_logger, add_prefix, remove_prefix
import numpy as np
import pickle
import os
import argparse
from common import env_list
from ray.rllib.evaluation.metrics import collect_episodes, summarize_episodes
from zoo_utils import create_mean_std


##################
# Hyper-parameters
##################

parser = argparse.ArgumentParser()
parser.add_argument("--env", type=int, default=5)
# random seed
parser.add_argument("--seed", type=int, default=0)
# The model used as the opponent. latest, random, best
parser.add_argument("--opp_model", type=str, default='random')
# whether train from scratch or not 
parser.add_argument("--finetune", type=bool, default=False)
# obs normalization params (mean / std)
parser.add_argument("--obs_norm_path", type=str, default='/Users/Henryguo/Desktop/rl_robustness/MuJoCo/initial-agents/SumoAnts-v0/agent0-model-v1.pkl')
# pretrain model path
parser.add_argument("--pretrain_model_path", type=str, default='/Users/Henryguo/Desktop/rl_robustness/MuJoCo/initial-agents/SumoAnts-v0/agent0-rms-v1.pkl')

# number of rollout worker actors to create for parallel sampling
parser.add_argument("--num_workers", type=int, default=20)

# number of environments to evaluate vectorwise per worker
parser.add_argument("--num_envs_per_worker", type=int, default=2)

# number of steps for rollout, similar to baselines' param n_steps
parser.add_argument("--rollout_fragment_length", type=int, default=2048)

parser.add_argument('--debug', type=bool, default=False)

args = parser.parse_args()


GAME_ENV = env_list[args.env]
GAME_SEED = args.seed
if args.opp_model == 'latest':
    OPP_METHOD = 0
elif args.opp_model == 'random':
    OPP_METHOD = 1
elif args.opp_model == 'best':
    OPP_METHOD = 2
else:
    print('unknown option of which model to be used as the opponent model, default as the latest model.')
    OPP_METHOD = 0

LR = 1e-3
NUPDATES = 2442
ENT_COEF = 0.0
VF_COEF = 0.5
GAMMA = 0.99

FINETUNE = args.finetune

SAVE_DIR = '../agent-zoo/'+ GAME_ENV + '_OPPO_Model_' + str(OPP_METHOD)
EXP_NAME = str(GAME_SEED)


# Custom evaluation during training
def custom_eval_function(trainer, eval_workers):
    """
    Args:
        trainer (Trainer): trainer class to evaluate.
        eval_workers (WorkerSet): evaluation workers.
    Returns:
        metrics (dict): evaluation metrics dict.
    """
    model = trainer.get_policy().get_weights()
    weights = ray.put({'default_policy': model})
    # sync the weights 
    for w in eval_workers.remote_workers():
        w.set_weights.remote(weights)
        w.foreach_env.remote(lambda env: env.set_winner_info())

    for i in range(5):
        print("Custom evaluation round", i)
        # Calling .sample() runs exactly one episode per worker due to how the
        # eval workers are configured.
        ray.get([w.sample.remote() for w in eval_workers.remote_workers()])

    # Collect the accumulated episodes on the workers, and then summarize the
    # episode stats into a metrics dict.
    episodes, _ = collect_episodes(
        remote_workers=eval_workers.remote_workers(), timeout_seconds=99999)
    # You can compute metrics from the episodes manually, or use the
    # convenient `summarize_episodes()` utility:
    metrics = summarize_episodes(episodes)

    game_info = []
    for w in eval_workers.remote_workers():
        out_info = ray.get(w.foreach_env.remote(lambda env: env.get_winner_info()))
        for out in out_info:
            game_info.append(out)

    game_results = np.zeros((3,))
    for game_res in game_info:
        game_results += game_res

    num_games = np.sum(game_results)
    win_0 = game_results[0] * 1.0 / num_games
    win_1 = game_results[1] * 1.0 / num_games
    tie = game_results[2] * 1.0 / num_games

    metrics['win_0'] = win_0
    metrics['win_1'] = win_1
    metrics['tie'] = tie

    # write into csv
    if out_dir is not None:
        os.makedirs(out_dir, exist_ok=True)
        fid = open(out_dir + '/Log.txt', 'a+')
        fid.write("%d %f %f %f\n" % (trainer.iteration, win_0, win_1, tie))
        fid.close()

    return metrics


if __name__ == '__main__':

    config = deepcopy(DEFAULT_CONFIG)

    # number of rollout worker actors to create for parallel sampling.
    config['num_workers'] = args.num_workers
    # number of environments to evaluate vectorwise per worker.
    config['num_envs_per_worker'] = args.num_envs_per_worker
    # number of timesteps collected for each SGD round. This defines the size of each SGD epoch.
    config['train_batch_size'] = 40 * 2048
    # total SGD batch size across all devices for SGD. This defines the minibatch size within each epoch.
    config['sgd_minibatch_size'] = 10 * 2048
    # number of steps for rollout, similar to baselines' param: n_steps.
    config['rollout_fragment_length'] = args.rollout_fragment_length
    config['lr'] = LR
    # Discount factor of the MDP
    config['gamma'] = GAMMA
    
    # warning: kl_coeff
    config['kl_coeff'] = 0
    # If specified, clip the global norm of gradients by this amount.
    config['grad_clip'] = 0.5
    # number of epochs to execute per train batch.
    config['num_sgd_iter'] = 4
    # PPO clip parameter.
    config['clip_param'] = 0.1
    # clip param for the value function
    config['vf_clip_param'] = 0.1
    # coefficient of the value function loss
    config['vf_loss_coeff'] = VF_COEF
    # The GAE (lambda) parameter
    config['lambda'] = 0.95


    # set the env params
    # note that if debug is True, we will play the video.
    config['env_config'] = {'env_name': GAME_ENV, 'epsilon': 1e-8, 
                            'clip_reward': 10.0, 'norm': True,
                            'gamma': 0.99, 'coef': 0.01,
                            'obs_norm_path': args.obs_norm_path,
                            'reward_move': 0.1, 'reward_remaining': 0.01,
                            'anneal_frac': 1, 'anneal_type': 1, 'lr': LR,
                            'total_step': config['train_batch_size'] * NUPDATES,
                            'finetune': FINETUNE, 'debug': args.debug}

    
    # Add mean_std filter
    config['observation_filter'] = "MeanStdFilter"

    # set the model
    config['model']['fcnet_hiddens'] = [128]
    config['model']['lstm_cell_size'] = 128

    ModelCatalog.register_custom_model("rnn", LSTM)
    register_env("mujoco", lambda config: MuJoCo_Env(config))
    env = MuJoCo_Env(config['env_config'])

    config['env'] = 'mujoco'
    config['model']['custom_model'] = 'rnn'
    # evaluation setting
    config['custom_eval_function'] = custom_eval_function
    config['evaluation_interval'] = 1
    config['evaluation_num_episodes'] = 50
    config['evaluation_num_workers'] = 10

    ray.init()

    trainer = PPOTrainer(env=MuJoCo_Env, config=config)

    # load pre-train model
    if FINETUNE:
        init_model = load_model(args.pretrain_model_path)
        init_filter = create_mean_std(args.obs_norm_path)
        trainer.workers.foreach_worker(lambda ev: ev.get_policy().set_weights(init_model))
        trainer.workers.foreach_worker(lambda ev: ev.filters.sync(init_filter))

    out_dir = setup_logger(SAVE_DIR, EXP_NAME)

    for i in range(1, NUPDATES):
        results = trainer.train()
