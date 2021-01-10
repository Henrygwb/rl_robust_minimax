import os
import ray
import pickle
import argparse
import random
import numpy as np
from copy import deepcopy
from env import MuJoCo_Env, env_list
from ray.rllib.agents.ppo.ppo_tf_policy import PPOTFPolicy
from ray.rllib.agents.ppo.ppo import PPOTrainer, DEFAULT_CONFIG
from ray.rllib.models import ModelCatalog
from ray.tune.registry import register_env
from ray.rllib.evaluation.metrics import collect_episodes, summarize_episodes
from zoo_utils import LSTM, MLP, load_pretrain_model, setup_logger, add_prefix, remove_prefix, create_mean_std


##################
# Hyper-parameters
##################

parser = argparse.ArgumentParser()
# Number of parallel workers/actors.
parser.add_argument("--num_workers", type=int, default=1)

# Number of environments per worker
parser.add_argument("--num_envs_per_worker", type=int, default=2)

# ["multicomp/YouShallNotPassHumans-v0", "multicomp/KickAndDefend-v0",
#  "multicomp/SumoAnts-v0", "multicomp/SumoHumans-v0"]
parser.add_argument("--env", type=int, default=2)

# Random seed.
parser.add_argument("--seed", type=int, default=0)

# The model used as the opponent. latest, random.
parser.add_argument("--opp_model", type=str, default='random')

# Loading a pretrained model as the initial model or not.
parser.add_argument("--load_pretrained_model", type=bool, default=True)

# Pretrained normalization params path.
parser.add_argument("--obs_norm_path", type=str,
                    default="/Users/Henryguo/Desktop/rl_robustness/MuJoCo/initial-agents/SumoAnts-v0/agent0-rms-v1.pkl")

# Pretrained model path.
parser.add_argument("--pretrain_model_path", type=str,
                    default="/Users/Henryguo/Desktop/rl_robustness/MuJoCo/initial-agents/SumoAnts-v0/agent0-model-v1.pkl")

parser.add_argument('--debug', type=bool, default=False)

args = parser.parse_args()

# ======= Setting for rollout worker processes =======
# Number of parallel workers/actors.
NUM_WORKERS = args.num_workers
# Number of environments per worker.
NUM_ENV_WORKERS = args.num_envs_per_worker
# Batch size collected from each worker.
ROLLOUT_FRAGMENT_LENGTH = 100

# === Settings for the Trainer process ===
# Number of iterations.
NUPDATES = 2442
# Number of epochs in each iteration.
NEPOCH = 4
# Training batch size.
TRAIN_BATCH_SIZE = 200
# Minibatch size. Num_epoch = train_batch_size/sgd_minibatch_size.
TRAIN_MINIBATCH_SIZE = 100
# Loading a pretrained model as the initial model or not.
LOAD_PRETRAINED_MODEL = args.load_pretrained_model
# Pretrained normalization params path.
OBS_NORM_PATH = args.obs_norm_path
# Pretrained model path.
MODEL_PATH = args.pretrain_model_path

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
CLIP_REWAED = 10.0
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

EVAL_NUM_EPISODES = 4
EVAL_NUM_WOEKER = 2


SAVE_DIR = '../agent-zoo/' + GAME_ENV.split('/')[1] + '_' + args.opp_model
EXP_NAME = str(GAME_SEED)


# Custom evaluation during training. This function is called when trainer.train() function ends
def custom_eval_function(trainer, eval_workers):
    """
    Args:
        trainer (Trainer): trainer class to evaluate.
        eval_workers (WorkerSet): evaluation workers.
    Returns:
        metrics (dict): evaluation metrics dict.
    """
    model = trainer.get_policy('model').get_weights()
    opp_model = trainer.get_policy('opp_model').get_weights()

    tmp_model = {}

    # Warning: eval_workers should load latest model 

    # In even iteration, use model as the current policy.
    # In odd iteration, use opp_model as the current policy.

    if trainer.iteration % 2 == 0:
        for (k1, v1), (k2, _) in zip(model.items(), opp_model.items()):
            # todo: logstd = 0 # deterministic distribution.
             tmp_model[k2] = v1
        weights = ray.put({'model': model, 'opp_model': tmp_model})
    else:
        for (k1, _), (k2, v2) in zip(model.items(), opp_model.items()):
             tmp_model[k1] = v2
        weights = ray.put({'model': tmp_model, 'opp_model': opp_model})

    # Copy the current policy to eval_workers' weight.
    for w in eval_workers.remote_workers():
        w.set_weights.remote(weights)
        w.foreach_env.remote(lambda env: env.set_winner_info())

    # Check the weights of each eval worker.
    # w_eval_model = eval_workers.foreach_worker(lambda ev: ev.get_policy('model').get_weights())
    # w_eval_opp_model = eval_workers.foreach_worker(lambda ev: ev.get_policy('opp_model').get_weights())
    # local_worker_model: w_eval_model[0]['model/fully_connected_1/bias']
    # remote_eval_i_worker_model: w_eval_model[i]['model/fully_connected_1/bias']
    # local_worker_opp_model: w_eval_opp_model[0]['opp_model/fully_connected_1/bias']
    # remote_eval_i_worker_opp_model: w_eval_opp_model[i]['opp_model/fully_connected_1/bias']
    # If using model/opp_model as the current policy,
    # all remote workers should have the same parameters with model/opp_model.

    for i in range(int(EVAL_NUM_EPISODES/EVAL_NUM_WOEKER)):
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

    # For each worker, get its parallel envs' win information and concate them.
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

    # write the winning information into txt.
    if out_dir is not None:
        os.makedirs(out_dir, exist_ok=True)
        fid = open(out_dir + '/Log.txt', 'a+')
        fid.write("%d %f %f %f\n" % (trainer.iteration, win_0, win_1, tie))
        fid.close()

    return metrics


# Define the policy_mapping_function.
# 'agent_0'  ---->   model
# 'agent_1'  ---->   opp_model
def policy_mapping_fn(agent_id):
    # get agent id via env.reset()
    ret = 'model' if agent_id == 'agent_0' else 'opp_model'
    return ret


def symmtric_learning(out_dir):
    # Symmtric Training algorithm
    # We define two trainable models, one for each party: mapping relation: agent_0 -> model  agent_1 -> opp_model.

    # Split the total training iteration into two halves.
    # In the even iterations, we load a previous (random/latest) model for agent_1 and train agent 0, save the model
    # of agent 0 as the trained policy after this iteration.
    # Copy the weights of model to the opp_model.
    # In the odd iterations, we load a previous (random/latest) model for agent_0 and train agent 1, save the opp_model
    # of agent 1 as the trained policy after this iteration.
    # Copy the weights of opp_model to the model.

    # In the even iterations, update model, sample a previous policy for opp_model.
    # In the odd iterations, update opp_model, sample a previous policy for model.

    for update in range(1, NUPDATES + 1):
        if update == 1:
            print('Use the initial agent as the opponent.')
        else:
            if OPP_MODEL == 0:
                print('Select the latest model')
                selected_opp_model = update - 1
            elif OPP_MODEL == 1:
                print('Select the random model')
                selected_opp_model = round(np.random.uniform(1, update - 1))
            else:
                print('Select the random model')
                selected_opp_model = round(np.random.uniform(1, update - 1))

            # In the even iteration, sample a previous policy for opp_model.
            # In the odd iteration, sample a previous policy for model.
            model_path = os.path.join(out_dir, 'checkpoints', 'model', '%.5i'%selected_opp_model, 'model')
            tmp_model = pickle.load(open(model_path, 'rb'))
            if update % 2 == 0:
                prefix = 'opp_model'
            else:
                prefix = 'model'
            tmp_model = add_prefix(tmp_model, prefix)
            trainer.workers.foreach_worker(lambda ev: ev.get_policy(prefix).set_weights(tmp_model))

        # Update both model and opp_model.
        result = trainer.train()

        # Ray will implicitly call custom_eval_function.

        # Sync model parameters.
        # In the even iteration, save model as the current policy and copy the weights of model to opp_model.
        # In the odd iteration, save opp_model as the current policy and copy the weights of opp_model to model.

        model = trainer.get_policy('model').get_weights()
        opp_model = trainer.get_policy('opp_model').get_weights()
        tmp_model = {}

        if update % 2 == 0:
            for (k1, v1), (k2, _) in zip(model.items(), opp_model.items()):
                tmp_model[k2] = v1
            trainer.workers.foreach_worker(lambda ev: ev.get_policy('opp_model').set_weights(tmp_model))
        else:
            for (k1, _), (k2, v2) in zip(model.items(), opp_model.items()):
                tmp_model[k1] = v2
            trainer.workers.foreach_worker(lambda ev: ev.get_policy('model').set_weights(tmp_model))

        # Check model parameters.
        # ww = trainer.workers.foreach_worker(lambda ev: ev.get_policy('model').get_weights())
        # ww_opp = trainer.workers.foreach_worker(lambda ev: ev.get_policy('opp_model').get_weights())
        # Check the length of ww/ww_opp. The first one is local worker, the others are remote works.

        # After sync the model weights, save the current policy and rms parameters.
        m = trainer.get_policy('model').get_weights()
        m = remove_prefix(m)
        checkdir = os.path.join(out_dir, 'checkpoints', 'model', '%.5i' % update)
        os.makedirs(checkdir, exist_ok=True)
        savepath = os.path.join(checkdir, 'model')
        pickle.dump(m, open(savepath, 'wb'))

        # Save the running mean std of the observations.
        if update % 2 == 0:
            obs_filter = trainer.workers.local_worker().get_filters()['model']
        else:
            obs_filter = trainer.workers.local_worker().get_filters()['opp_model']
        savepath = os.path.join(checkdir, 'obs_rms')
        pickle.dump(obs_filter, open(savepath, 'wb'))

        # Save the running mean std of the rewards.
        for r in range(NUM_WORKERS):
            remote_worker = trainer.workers.remote_workers()[r]
            if update % 2 == 0:
                rt_rms_all = ray.get(remote_worker.foreach_env.remote(lambda env: env.ret_rms_0))
            else:
                rt_rms_all = ray.get(remote_worker.foreach_env.remote(lambda env: env.ret_rms_1))
            rt_rms_tmp = rt_rms_all[0]
            for l in range(len(rt_rms_all)):
                rt_rms_tmp.update_with_other(rt_rms_all[l])

            if r == 0:
                rt_rms = rt_rms_tmp
            else:
                rt_rms.update_with_other(rt_rms_tmp)

        rt_rms = {'rt_rms': rt_rms}
        savepath = os.path.join(checkdir, 'rt_rms')
        pickle.dump(rt_rms, open(savepath, 'wb'))
    return 0


def assymmtric_learning(out_dir):
    # Assymmtric Training algorithm
    # We define two trainable models, one for each party: mapping relation: agent_0 -> model  agent_1 -> opp_model.

    # Split the total training iteration into two halves.
    # In the even iterations, we load a previous (random/latest) model for agent_1 and train agent 0.
    # Save the weights of model as the current policy of model
    # Load the weights of opp_model at the last iteration to opp_model.
    # In the odd iterations, we load a previous (random/latest) model for agent_0 and train agent 1.
    # Save the weights of opp_model as the current policy of opp_model
    # Load the weights of model at the last iteration to model.

    # In the even iterations, update model, sample a previous policy for opp_model.
    # In the odd iterations, update opp_model, sample a previous policy for model.

    for update in range(1, NUPDATES + 1):
        if update % 2 == 0:
            load_idx = 'opp_model'
            save_idx = 'model'
        else:
            load_idx = 'model'
            save_idx = 'opp_model'

        if update == 1:
            print('Use the initial agent as the opponent.')
        else:
            if OPP_MODEL == 0:
                print('Select the latest model')
                selected_opp_model = update - 1
            elif OPP_MODEL == 1:
                print('Select the random model')
                selected_opp_model = round(np.random.uniform(1, update - 1))
            else:
                print('Select the random model')
                if update % 2 == 0:
                    selected_opp_model = random.randrange(1, update-1, 2)
                else:
                    selected_opp_model = random.randrange(2, update-1, 2)

            # In the even iteration, sample a previous policy for opp_model (Only be saved in the odd iterations).
            # In the odd iteration, sample a previous policy for model (Only be saved in the even iterations).

            model_path = os.path.join(out_dir, 'checkpoints', load_idx, '%.5i'%selected_opp_model, 'model')
            tmp_model = pickle.load(open(model_path, 'rb'))
            tmp_model = add_prefix(tmp_model, load_idx)
            trainer.workers.foreach_worker(lambda ev: ev.get_policy(load_idx).set_weights(tmp_model))

        # Update both model and opp_model.
        result = trainer.train()

        # Ray will implicitly call custom_eval_function.

        # In the even iteration, save model as the current policy and load the opp_model weights in the last iteration.
        # In the odd iteration, save opp_model as the current policy and load the model weights in the last iteration.

        latest_model_path = os.path.join(out_dir, 'checkpoints', load_idx, '%.5i'%(update-1), 'model')
        tmp_model = pickle.load(open(latest_model_path, 'rb'))
        tmp_model = add_prefix(tmp_model, load_idx)
        trainer.workers.foreach_worker(lambda ev: ev.get_policy(load_idx).set_weights(tmp_model))

        # Check model parameters.
        # ww = trainer.workers.foreach_worker(lambda ev: ev.get_policy('model').get_weights())
        # ww_opp = trainer.workers.foreach_worker(lambda ev: ev.get_policy('opp_model').get_weights())
        # Check the length of ww/ww_opp. The first one is local worker, the others are remote works.

        m = trainer.get_policy(save_idx).get_weights()
        m = remove_prefix(m)
        checkdir = os.path.join(out_dir, 'checkpoints', save_idx, '%.5i' % update)
        os.makedirs(checkdir, exist_ok=True)
        savepath = os.path.join(checkdir, 'model')
        pickle.dump(m, open(savepath, 'wb'))

        # Save the running mean std of the observations.
        obs_filter = trainer.workers.local_worker().get_filters()[save_idx]
        savepath = os.path.join(checkdir, 'obs_rms')
        pickle.dump(obs_filter, open(savepath, 'wb'))

        # Save the running mean std of the rewards.
        if update%2 == 0:
            obs_filter = trainer.workers.local_worker().get_filters()['model']
        else:
            obs_filter = trainer.workers.local_worker().get_filters()['opp_model']
        savepath = os.path.join(checkdir, 'obs_rms')
        pickle.dump(obs_filter, open(savepath, 'wb'))

        # Save the running mean std of the rewards.
        for r in range(NUM_WORKERS):
            remote_worker = trainer.workers.remote_workers()[r]
            if update % 2 == 0:
                rt_rms_all = ray.get(remote_worker.foreach_env.remote(lambda env: env.ret_rms_0))
            else:
                rt_rms_all = ray.get(remote_worker.foreach_env.remote(lambda env: env.ret_rms_1))
            rt_rms_tmp = rt_rms_all[0]
            for l in range(len(rt_rms_all)):
                rt_rms_tmp.update_with_other(rt_rms_all[l])

            if r == 0:
                rt_rms = rt_rms_tmp
            else:
                rt_rms.update_with_other(rt_rms_tmp)

        rt_rms = {'rt_rms': rt_rms}
        savepath = os.path.join(checkdir, 'rt_rms')
        pickle.dump(rt_rms, open(savepath, 'wb'))

    return 0


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
                            'clip_reward': CLIP_REWAED, # Reward clip boundary.
                            'epsilon': 1e-8, # Small value used for normalization.
                            'normalize': True, # Reward normalization.
                            'obs_norm_path': OBS_NORM_PATH,
                            'reward_move': 0.1, # Dense reward fraction.
                            'reward_remaining': 0.01, # Sparse reward fraction.
                            'anneal_frac': 0, # Constant: (0: only use sparse reward. 1: only use dense reward).
                            'anneal_type': 0, # Anneal type: 0: Constant anneal, 1: Linear anneal (set anneal_frac as 1).
                            'total_step': TRAIN_BATCH_SIZE * NUPDATES, # total time steps.
                            'LOAD_PRETRAINED_MODEL': False, # Whether to load a pretrained model.
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

    # === Policy Settings ===
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
    # based on the action space type (catalog: 213).
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
    config['custom_eval_function'] = custom_eval_function
    config['evaluation_interval'] = 1
    config['evaluation_num_episodes'] = EVAL_NUM_EPISODES
    config['evaluation_num_workers'] = EVAL_NUM_WOEKER

    # Initialize the ray.
    ray.init(local_mode=True)
    trainer = PPOTrainer(env=MuJoCo_Env, config=config)
    # This instruction will build a trainer class and setup the trainer. The setup process will make workers, which will
    # call DynamicTFPolicy in dynamic_tf_policy.py. DynamicTFPolicy will define the action distribution based on the
    # action space type (line 151) and build the model.

    if LOAD_PRETRAINED_MODEL:
        init_model, init_opp_model = load_pretrain_model(args.pretrain_model_path)

        # Load the pretrained model as the initial model.
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
        init_filter = create_mean_std(args.obs_norm_path)
        trainer.workers.foreach_worker(lambda ev: ev.filters['model'].sync(init_filter))
        trainer.workers.foreach_worker(lambda ev: ev.filters['opp_model'].sync(init_filter))

        # Check obs_rms loading.
        # trainer.workers.local_worker().get_filters()['opp_model']
        # trainer.workers.local_worker().get_filters()['model']
        # All fitlers: trainer.workers.foreach_worker(lambda ev: ev.get_filters())

    out_dir = setup_logger(SAVE_DIR, EXP_NAME)

    if SYMM_TRAIN:
        symmtric_learning(out_dir)
    else:
        assymmtric_learning(out_dir)
